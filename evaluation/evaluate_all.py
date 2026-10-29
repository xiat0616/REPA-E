r"""Evaluate one checkpoint with shared images for identity, Inception FID and DINO FD.

    .venv/bin/python evaluation/evaluate_all.py \
        --checkpoint checkpoints/EXPERIMENT/checkpoints/0200000.pt \
        --num-samples 5000 --num-steps 50 --seed 1234

Results: evaluation/results/EXPERIMENT/step_0200000/
    test_n5000_seed1234_steps50_ema/{identity,fid_inception,fd_dino}/
The run also saves config.json, indices.json and a combined summary.csv.
evaluation/results/comparison.csv collects completed runs with their protocol IDs.
Reuse --indices-file across checkpoints. Existing result directories are preserved.

Default: all metrics/tasks; EMA SiT; checkpoint VAE for REPA-E; FP32; posterior mode.
Generation, VAE reconstruction, flow cycle and each intervention are computed once
per batch, then scored by the enabled metrics. No subprocess evaluators are run.
Identity scores are similarity proxies. CF FIDs use observational target groups;
age uses fixed bins. Low-support groups remain unavailable, and the weighted CF
summary is labeled as a mean of target-group distances rather than a pooled FID.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from data_handle.embed import DatasetConfig, build_image_transform
from evaluate_identity import ATTRIBUTES, EvaluationDataset, IdentityMetrics, load_models, make_intervention, write_summaries
from evaluate_frechet import DinoFeatures, InceptionFeatures, group_label, sha256_file, update_groups
from frechet_metrics import FeatureMoments, frechet_distance
from plot_utils import comparison_figure_to_array, decode_flow_latent, flow_euler
from samplers import euler_sampler
from utils import preprocess_imgs_vae


def distribution_summary(stats, args):
    """Assemble the same global/target-group distances as the standalone evaluator."""
    rows = []
    for task, values in stats['outputs'].items():
        if values.count:
            rows.append({'comparison': task, 'attribute': 'none', 'target': 'all',
                         **frechet_distance(stats['real'], values)})
    for (attribute, group), values in sorted(stats['counterfactual'].items()):
        reference = stats['targets'].get((attribute, group), FeatureMoments(stats['dimension']))
        rows.append({'comparison': 'counterfactual_target', 'attribute': attribute,
                     'target': group_label(attribute, group, args.age_bin_edges),
                     **frechet_distance(reference, values)})
    aggregates = []
    if 'counterfactual' in args.tasks:
        for attribute in args.attributes:
            groups = [row for row in rows if row['comparison'] == 'counterfactual_target' and row['attribute'] == attribute]
            if not groups:
                unavailable = frechet_distance(stats['real'], FeatureMoments(stats['dimension']))
                rows.append({'comparison': 'counterfactual_target', 'attribute': attribute,
                             'target': 'none', **dict(unavailable, status='no_interventions')})
                continue
            valid = [row for row in groups if row['score'] is not None]
            n = sum(row['n_generated'] for row in groups)
            aggregates.append({
                'attribute': attribute, 'summary_type': 'generated-count-weighted mean of target-group FDs; not a pooled FD',
                'score': sum(row['score'] * row['n_generated'] for row in valid) / n if len(valid) == len(groups) else None,
                'status': 'ok' if len(valid) == len(groups) else 'insufficient_target_support',
                'n_generated': n, 'n_scored': sum(row['n_generated'] for row in valid),
                'target_groups': len(groups), 'scored_target_groups': len(valid),
                'rank_deficient': any(row['rank_deficient'] for row in groups),
            })
    return rows, aggregates


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def update_comparison(output_root, rows):
    """Register only a completed run; lock and atomically replace the shared CSV."""
    path = output_root / 'comparison.csv'
    evaluation_id = rows[0]['evaluation_id']
    with (output_root / '.comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = []
        if path.exists():
            with path.open(newline='') as handle:
                previous = [row for row in csv.DictReader(handle) if row['evaluation_id'] != evaluation_id]
        with tempfile.NamedTemporaryFile(dir=output_root, prefix='.comparison-', suffix='.csv', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            write_csv(temporary, previous + rows)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@torch.no_grad()
def main(args):
    if args.batch_size < 1 or args.num_steps < 1 or args.cpu_threads < 1 or args.num_workers < 0 or args.num_samples < 0 or args.save_examples < 0:
        raise ValueError('Invalid batch, step, sample, example or worker count.')
    if not np.isfinite(args.cfg_scale) or args.cfg_scale < 1:
        raise ValueError('--cfg-scale must be finite and >=1 (generation only).')
    if args.target_age is not None and not 0 <= args.target_age <= 100:
        raise ValueError('--target-age must be in [0,100] years.')
    edges = np.asarray(args.age_bin_edges)
    if len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError('--age-bin-edges must be finite and strictly increasing.')
    identity_enabled = 'identity' in args.evaluations
    if identity_enabled and not set(args.tasks) & {'reconstruction', 'counterfactual'}:
        raise ValueError('Identity metrics require reconstruction or counterfactual tasks.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    before = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    model, vae, config, vae_source = load_models(checkpoint, args.weights, device, args.vae_ckpt)
    step = checkpoint.get('steps', checkpoint_path.stem)
    del checkpoint

    # 1). Select one common held-out cohort for every enabled metric.
    data_dir = Path(args.data_dir or config['data_dir']).expanduser()
    split_dir = Path(args.split_dir or config['split_dir']).expanduser()
    if not args.data_dir and not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not args.split_dir and not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    split_file = split_dir / f'{args.split}.csv'
    split_hash = sha256_file(split_file)
    frame = pd.read_csv(split_file, low_memory=False)
    if args.indices_file:
        selection = json.loads(Path(args.indices_file).read_text())
        if selection['split'] != args.split or selection['split_csv_sha256'] != split_hash:
            raise ValueError('Indices file belongs to a different split CSV or row ordering.')
        indices = selection['indices']
    else:
        order = torch.randperm(len(frame), generator=torch.Generator().manual_seed(args.seed))
        indices = order[:min(args.num_samples, len(frame))].tolist() if args.num_samples else order.tolist()
    minimum = 2 if set(args.evaluations) & {'inception', 'dino'} else 1
    if len(indices) < minimum or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(frame) for i in indices):
        raise ValueError(f'Evaluation needs at least {minimum} unique, valid split indices.')
    dataset = EvaluationDataset(data_dir, frame, args.split, parents=list(ATTRIBUTES),
                                transform=build_image_transform(DatasetConfig(img_height=config['height'], img_width=config['width'], img_channels=1)))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False, pin_memory=device.type == 'cuda')
    experiment = args.run_name or config.get('exp_name') or (checkpoint_path.parent.parent.name if checkpoint_path.parent.name == 'checkpoints' else checkpoint_path.parent.name)
    if Path(experiment).name != experiment or experiment in ('.', '..'):
        raise ValueError('--run-name/experiment must be one folder name.')
    step_label = f'{int(step):07d}' if str(step).isdigit() else checkpoint_path.stem
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_root / experiment / f'step_{step_label}' / f'{args.split}_n{len(indices)}_seed{args.seed}_steps{args.num_steps}_{args.weights}'
    if output_dir.exists():
        raise FileExistsError(f'Results already exist: {output_dir}. Choose another --output-dir.')

    # Resolve the Inception cache before LPIPS can change torch.hub's cache directory.
    inception_name = 'weights-inception-2015-12-05-6726825d.pth'
    candidates = [Path(torch.hub.get_dir()) / 'checkpoints' / inception_name,
                  ROOT / 'cache/torch/hub/checkpoints' / inception_name,
                  Path.home() / '.cache/torch/hub/checkpoints' / inception_name]
    inception_weights = Path(args.inception_weights).expanduser().resolve() if args.inception_weights else next((p for p in candidates if p.is_file()), candidates[0])
    extractors, feature_metadata = {}, {}
    if 'inception' in args.evaluations:
        extractors['fid_inception'] = InceptionFeatures(inception_weights, device).eval()
        feature_metadata['fid_inception'] = {
            'metric': 'FID', 'feature_model': 'inception_v3_tf_2048', 'dimension': 2048,
            'weights': str(inception_weights), 'weights_sha256': sha256_file(inception_weights),
            'library': 'torch-fidelity', 'library_version': importlib.metadata.version('torch-fidelity'),
            'preprocessing': 'RGB uint8; backend TF-compatible bilinear299 and scaling; pool3',
        }
    if 'dino' in args.evaluations:
        dino_repo = Path(args.dino_repo).expanduser().resolve()
        dino_weights = Path(args.dino_weights or ROOT / f'cache/torch/hub/checkpoints/{args.feature_model}_pretrain.pth').expanduser().resolve()
        extractors['fd_dino'] = DinoFeatures(args.feature_model, dino_repo, dino_weights, device).eval()
        feature_metadata['fd_dino'] = {
            'metric': 'Frechet DINO distance', 'feature_model': args.feature_model,
            'dimension': extractors['fd_dino'].dimension, 'weights': str(dino_weights),
            'weights_sha256': sha256_file(dino_weights), 'repo': str(dino_repo),
            'repo_source_sha256': {str(p.relative_to(dino_repo)): sha256_file(p) for p in sorted(dino_repo.rglob('*.py'))},
            'preprocessing': 'RGB uint8 grid; bicubic224 antialias; ImageNet normalization; CLS without extra L2 normalization',
        }
    identity = IdentityMetrics(args.identity_metrics, device, args.torch_hub_dir) if identity_enabled else None
    statistics = {name: {
        'dimension': extractor.dimension, 'real': FeatureMoments(extractor.dimension),
        'outputs': {task: FeatureMoments(extractor.dimension) for task in ['generation', 'reconstruction_vae', 'reconstruction_cycle']},
        'targets': {}, 'counterfactual': {},
    } for name, extractor in extractors.items()}
    metadata = {
        'status': 'running', 'checkpoint': str(checkpoint_path), 'step': step, 'experiment': experiment,
        'checkpoint_bytes': before.st_size, 'checkpoint_mtime_ns': before.st_mtime_ns,
        'weights': args.weights, 'vae_source': vae_source, 'model': config['model'],
        'resolution': [config['height'], config['width']], 'split': args.split,
        'split_csv': str(split_file), 'split_csv_sha256': split_hash, 'num_images': len(indices),
        'indices_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        'arguments': vars(args), 'feature_extractors': feature_metadata, 'torch_version': torch.__version__,
        'inference': 'FP32, TF32 off, posterior mode, selected saved BN; CF/cycle CFG1; generation requested CFG/Heun',
        'notes': ['Identity metrics measure image similarity; low distance alone does not demonstrate a successful counterfactual.',
                  'CF distribution references are observational target categories or age bins; other covariates may differ.',
                  'Sample-size effects are flagged. Target-weighted FD means are not pooled FIDs.',
                  'DINOv2 is also used as a REPA teacher; compare with Inception and identity scores.'],
        'source_sha256': {name: sha256_file(ROOT / name) for name in [
            'evaluation/evaluate_all.py', 'evaluation/evaluate_identity.py', 'evaluation/evaluate_frechet.py',
            'evaluation/frechet_metrics.py', 'plot_utils.py', 'samplers.py', 'models/sit.py',
            'models/embedder.py', 'models/autoencoder.py', 'data_handle/embed.py', 'utils.py', 'loss/lpips.py']},
    }
    signature = dict(metadata)
    signature.pop('status')
    signature['arguments'] = {key: value for key, value in vars(args).items() if key not in ('output_root', 'output_dir', 'run_name', 'indices_file', 'save_examples')}
    metadata['evaluation_id'] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = {'split': args.split, 'split_csv_sha256': split_hash, 'indices': indices}
    (output_dir / 'indices.json').write_text(json.dumps(selection, indent=2) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    identity_rows, eligible_counts = [], {attribute: 0 for attribute in args.attributes}
    example_counts = {attribute: 0 for attribute in args.attributes}

    for batch in tqdm(loader, desc='Combined evaluation'):
        original = batch['x'].float().to(device)
        parents = {key: value.float().to(device) for key, value in batch['pa'].items()}
        batch_indices = batch['index'].tolist()
        # 2). Each reference image is encoded once by each feature extractor.
        for name, extractor in extractors.items():
            features = extractor(original)
            statistics[name]['real'].update(features)
            if 'counterfactual' in args.tasks:
                for attribute in args.attributes:
                    update_groups(statistics[name]['targets'], features, parents[attribute], attribute, args)
        # 3). Generate once, then send the same images to both distribution metrics.
        if extractors and 'generation' in args.tasks:
            factor = 8 if config['vae'] == 'f8d4' else 16
            shape = (model.in_channels, config['height'] // factor, config['width'] // factor)
            noise = torch.stack([torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(args.seed + i)) for i in batch_indices])
            z = euler_sampler(model, noise, parents, num_steps=args.num_steps, heun=args.heun,
                              cfg_scale=args.cfg_scale, path_type=config.get('path_type', 'linear')).float()
            images = decode_flow_latent(model, vae, z)
            for name, extractor in extractors.items():
                statistics[name]['outputs']['generation'].update(extractor(images))
        if not set(args.tasks) & {'reconstruction', 'counterfactual'}:
            continue
        # 4). One VAE encode, one factual inversion and one no-edit return per batch.
        raw_z = vae.encode(preprocess_imgs_vae(original.repeat(1, 3, 1, 1) * 255.).float()).mode().float()
        z0 = model.bn(raw_z)
        zT = flow_euler(model, z0, parents, 0., 1., args.num_steps)
        baseline_images = {}
        if identity_enabled or 'reconstruction' in args.tasks:
            baseline_images['vae_vs_input'] = ((vae.decode(raw_z).sample.mean(1, keepdim=True) + 1) / 2).clamp(0, 1)
            cycle_z = flow_euler(model, zT, parents, 1., 0., args.num_steps)
            baseline_images['cycle_vs_input'] = decode_flow_latent(model, vae, cycle_z)
        baseline_values = {}
        for comparison, images in baseline_images.items():
            if identity_enabled:
                values = identity(original, images)
                baseline_values[comparison] = values
                for i, sample_index in enumerate(batch_indices):
                    identity_rows.append({'index': sample_index, 'comparison': comparison, 'attribute': 'none', 'source': '', 'target': '',
                                          **{key: value[i] for key, value in values.items()}})
            if 'reconstruction' in args.tasks:
                task = 'reconstruction_vae' if comparison == 'vae_vs_input' else 'reconstruction_cycle'
                for name, extractor in extractors.items():
                    statistics[name]['outputs'][task].update(extractor(images))
        # 5). Compute each intervention once; all metrics consume that same CF.
        if 'counterfactual' in args.tasks:
            for attribute in args.attributes:
                cf_parents, eligible = make_intervention(parents, attribute, args)
                selected = torch.where(eligible)[0]
                positions = selected.cpu().tolist()
                eligible_counts[attribute] += len(positions)
                if not positions:
                    continue
                target_parents = {key: value[selected] for key, value in cf_parents.items()}
                cf_z = flow_euler(model, zT[selected], target_parents, 1., 0., args.num_steps)
                cf = decode_flow_latent(model, vae, cf_z)
                for name, extractor in extractors.items():
                    update_groups(statistics[name]['counterfactual'], extractor(cf), target_parents[attribute], attribute, args)
                if identity_enabled:
                    sources, targets = parents[attribute][selected], target_parents[attribute]
                    sources = sources[:, 0] * 100 if attribute == 'age' else sources.argmax(1)
                    targets = targets[:, 0] * 100 if attribute == 'age' else targets.argmax(1)
                    values_by_comparison = {comparison: {key: [value[p] for p in positions] for key, value in values.items()}
                                            for comparison, values in baseline_values.items()}
                    values_by_comparison['cf_vs_input'] = identity(original[selected], cf)
                    values_by_comparison['cf_vs_cycle'] = identity(baseline_images['cycle_vs_input'][selected], cf)
                    for comparison, values in values_by_comparison.items():
                        for i, position in enumerate(positions):
                            identity_rows.append({'index': batch_indices[position], 'comparison': comparison, 'attribute': attribute,
                                                  'source': float(sources[i]), 'target': float(targets[i]),
                                                  **{key: value[i] for key, value in values.items()}})
                count = min(args.save_examples - example_counts[attribute], len(positions))
                if count > 0:
                    from PIL import Image
                    examples = output_dir / 'examples'
                    examples.mkdir(exist_ok=True)
                    outputs, labels = [cf[:count]], [f'{attribute} CF']
                    if 'cycle_vs_input' in baseline_images:
                        outputs.insert(0, baseline_images['cycle_vs_input'][selected][:count])
                        labels.insert(0, 'No-edit cycle')
                    grid = comparison_figure_to_array(original[selected][:count], outputs, labels)
                    Image.fromarray(grid).save(examples / f'{attribute}_{example_counts[attribute]:06d}.png')
                    example_counts[attribute] += count
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during evaluation; this run will not be registered.')

    # 6). Save metric folders and publish one completed comparison record.
    metadata['eligible_interventions'] = eligible_counts
    metadata['status'] = 'complete'
    common = {key: metadata[key] for key in ('evaluation_id', 'experiment', 'step', 'checkpoint', 'weights', 'split', 'num_images')}
    common.update(seed=args.seed, num_steps=args.num_steps, results_dir=str(output_dir))
    combined_rows = []
    if identity_enabled:
        folder = output_dir / 'identity'
        folder.mkdir()
        write_summaries(folder, identity_rows, metadata)
        summary = json.loads((folder / 'summary.json').read_text())
        for row in summary['metrics']:
            combined_rows.append(dict(common, family='identity', comparison=row['comparison'], attribute=row['attribute'], target='all',
                                      metric=row['metric'], value=row['mean'], std=row['std'], n_real=row['n'], n_generated=row['n'],
                                      status='ok', low_sample_count='', rank_deficient=''))
    for name, stats in statistics.items():
        folder = output_dir / name
        folder.mkdir()
        rows, aggregates = distribution_summary(stats, args)
        backend_metadata = dict(metadata, active_feature_extractor=feature_metadata[name])
        (folder / 'summary.json').write_text(json.dumps({'metadata': backend_metadata, 'distances': rows, 'counterfactual_target_weighted': aggregates}, indent=2, allow_nan=False) + '\n')
        write_csv(folder / 'summary.csv', rows)
        metric = 'fid_inception' if name == 'fid_inception' else 'fd_' + args.feature_model
        for row in rows:
            combined_rows.append(dict(common, family=name, comparison=row['comparison'], attribute=row['attribute'], target=row['target'],
                                      metric=metric, value=row['score'], std='', n_real=row['n_real'], n_generated=row['n_generated'],
                                      status=row['status'], low_sample_count=row['low_sample_count'], rank_deficient=row['rank_deficient']))
        for row in aggregates:
            combined_rows.append(dict(common, family=name, comparison='counterfactual_target_weighted_mean', attribute=row['attribute'], target='all',
                                      metric=metric, value=row['score'], std='', n_real='', n_generated=row['n_generated'],
                                      status=row['status'], low_sample_count='', rank_deficient=row['rank_deficient']))
    write_csv(output_dir / 'summary.csv', combined_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    update_comparison(output_root, combined_rows)
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(f'Completed evaluation: {output_dir}')
    print(f'Comparison table: {output_root / "comparison.csv"}')
    return output_dir


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=['ema', 'model'], default='ema')
    parser.add_argument('--vae-ckpt', default=None)
    parser.add_argument('--run-name', default=None, help='Optional experiment folder name.')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--split-dir', default=None)
    parser.add_argument('--split', choices=['test', 'valid'], default='test')
    parser.add_argument('--num-samples', type=int, default=5000, help='0 means the whole split; ignored with --indices-file.')
    parser.add_argument('--indices-file', default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--evaluations', nargs='+', choices=['identity', 'inception', 'dino'], default=['identity', 'inception', 'dino'])
    parser.add_argument('--identity-metrics', nargs='+', choices=['mae', 'mse', 'ssim', 'lpips'], default=['mae', 'mse', 'ssim', 'lpips'])
    parser.add_argument('--tasks', nargs='+', choices=['generation', 'reconstruction', 'counterfactual'], default=['generation', 'reconstruction', 'counterfactual'])
    parser.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=list(ATTRIBUTES))
    parser.add_argument('--target-view', type=int, choices=range(2), default=None)
    parser.add_argument('--target-density', type=int, choices=range(4), default=None)
    parser.add_argument('--target-scanner', type=int, choices=range(5), default=None)
    parser.add_argument('--target-age', type=float, default=None)
    parser.add_argument('--age-bin-edges', nargs='+', type=float, default=[0, 40, 50, 60, 70, 80, 101])
    parser.add_argument('--feature-model', choices=['dinov2_vitb14', 'dinov2_vitl14'], default='dinov2_vitb14')
    parser.add_argument('--dino-repo', default=str(ROOT / 'cache/torch/hub/facebookresearch_dinov2_main'))
    parser.add_argument('--dino-weights', default=None)
    parser.add_argument('--inception-weights', default=None, help='Converted FID weights; automatically finds an existing cache file.')
    parser.add_argument('--torch-hub-dir', default=str(ROOT / 'cache/torch/hub'), help='VGG16 cache for identity LPIPS.')
    parser.add_argument('--cfg-scale', type=float, default=1, help='Fresh generation only; CF and cycle use CFG1.')
    parser.add_argument('--heun', action='store_true', help='Heun correction for fresh generation only.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--save-examples', type=int, default=0, help='Example CF images per attribute, saved on this machine.')
    parser.add_argument('--output-root', default=str(ROOT / 'evaluation/results'))
    parser.add_argument('--output-dir', default=None, help='Optional new directory for this run; comparison.csv stays under --output-root.')
    args = parser.parse_args(input_args)
    for name in ('evaluations', 'identity_metrics', 'tasks', 'attributes'):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


if __name__ == '__main__':
    main(parse_args())
