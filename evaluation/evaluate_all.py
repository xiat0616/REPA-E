r"""Evaluate saved images without loading the diffusion model, VAE or raw dataset.

    .venv/bin/python evaluation/evaluate_all.py \
        --samples-dir sample_results/EXPERIMENT/step_0065000/test_n5000_seed1234_steps50_ema

Sampling is a separate step, performed by sample_all.py. This evaluator reads the
completed samples.csv and its 16-bit grayscale PNGs. It checks each image checksum
and uses fixed [0,1] scaling; it never rescales an image by its own intensity range.
The saved-image protocol includes uint16 quantization for both references and
outputs. Existing result directories are preserved; choose another --output-dir
to evaluate the same image cache with additional metrics or settings.
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
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from identity_metrics import IdentityMetrics, write_summaries
from feature_metrics import DinoFeatures, InceptionFeatures, group_label, sha256_file, update_groups
from frechet_metrics import FeatureMoments, frechet_distance
from sample.sample_io import load_samples, read_image

ATTRIBUTES = ('view', 'density', 'scanner', 'age')
TASKS = ('generation', 'reconstruction', 'counterfactual')
BASELINES = {'reconstruction_vae': 'vae_vs_input', 'reconstruction_cycle': 'cycle_vs_input'}


class SavedImages(Dataset):
    """Load only requested saved images, with one factual reference per row."""
    def __init__(self, root, config, selection, records, tasks, attributes, identity_enabled):
        self.root = root
        self.resolution = tuple(config['resolution'])
        self.indices = selection['indices']
        self.rows = {index: {} for index in self.indices}
        for row in records:
            index = int(row['index'])
            key = (row['task'], row['attribute'])
            self.rows[index][key] = row
        self.keys = [('real', 'none')]
        if 'generation' in tasks:
            self.keys.append(('generation', 'none'))
        if 'reconstruction' in tasks or ('counterfactual' in tasks and identity_enabled):
            self.keys.extend((task, 'none') for task in BASELINES)
        self.cf_attributes = attributes if 'counterfactual' in tasks else []
        for index, rows in self.rows.items():
            for key in self.keys:
                if key not in rows:
                    raise ValueError(f'Missing required {key[0]} image for split row {index}.')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        records = self.rows[index]
        keys = self.keys + [('counterfactual', attribute) for attribute in self.cf_attributes
                            if ('counterfactual', attribute) in records]
        images = {}
        for key in keys:
            row = records[key]
            image = read_image(self.root, row['file_path'], row['sha256'])
            if tuple(image.shape) != (1, *self.resolution):
                raise ValueError(f'Cached image dimensions differ from config: {row["file_path"]}')
            images[key] = image
        return {'index': index, 'records': records, 'images': images}


def keep_items(items):
    return items


def stack_images(items, key, device):
    return torch.stack([item['images'][key] for item in items]).to(device)


def parent_values(rows, attribute, device):
    values = torch.tensor([float(row[attribute]) for row in rows], device=device)
    if attribute == 'age':
        return values[:, None] / 100.
    count = {'view': 2, 'density': 4, 'scanner': 5}[attribute]
    return torch.nn.functional.one_hot(values.long(), count).float()


def save_example(original, cycle, counterfactual, destination, attribute):
    """A compact inspection grid; the authoritative saved images remain uint16."""
    from PIL import Image, ImageDraw
    columns = [('Input', original), ('No-edit cycle', cycle), (f'{attribute} CF', counterfactual)]
    height, width = original.shape[-2:]
    canvas = Image.new('RGB', (3 * width, len(original) * height + 24), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (label, images) in enumerate(columns):
        draw.text((column * width + 4, 4), label, fill='black')
        for row, image in enumerate(images):
            pixels = (image[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).convert('RGB'), (column * width, 24 + row * height))
    canvas.save(destination)

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


def resolve_tasks(args, config, records):
    cached = config.get('arguments', {})
    available = cached.get('tasks')
    if available is None:
        available = []
        recorded = {row['task'] for row in records}
        if 'generation' in recorded:
            available.append('generation')
        if set(BASELINES).issubset(recorded):
            available.append('reconstruction')
        if 'counterfactual' in recorded:
            available.append('counterfactual')
    requested = args.tasks if args.tasks is not None else [task for task in TASKS if task in available]
    if not requested or not set(requested).issubset(available):
        raise ValueError(f'Requested tasks are not in the saved cache. Available: {available}')
    available_attributes = cached.get('attributes', list(ATTRIBUTES))
    attributes = args.attributes if args.attributes is not None else list(available_attributes)
    if 'counterfactual' in requested and not set(attributes).issubset(available_attributes):
        raise ValueError(f'Requested CF attributes are not cached. Available: {available_attributes}')
    return requested, attributes


@torch.no_grad()
def main(args):
    if min(args.batch_size, args.cpu_threads) < 1 or min(args.num_workers, args.save_examples) < 0:
        raise ValueError('Batch size/CPU threads must be positive; workers/examples nonnegative.')
    edges = np.asarray(args.age_bin_edges)
    if len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError('--age-bin-edges must be finite and strictly increasing.')
    samples_dir = Path(args.samples_dir).expanduser().resolve()
    sample_config, selection, records = load_samples(samples_dir)
    args.tasks, args.attributes = resolve_tasks(args, sample_config, records)
    identity_enabled = 'identity' in args.evaluations
    if identity_enabled and not set(args.tasks) & {'reconstruction', 'counterfactual'}:
        raise ValueError('Identity metrics require reconstruction or counterfactual tasks.')
    indices = selection['indices']
    if len(indices) < (2 if set(args.evaluations) & {'inception', 'dino'} else 1):
        raise ValueError('Distribution evaluation requires at least two cached images.')
    seed = int(sample_config['arguments'].get('seed', 1234))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    dataset = SavedImages(samples_dir, sample_config, selection, records, args.tasks,
                          args.attributes, identity_enabled or args.save_examples > 0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=keep_items,
                        pin_memory=device.type == 'cuda')
    step = sample_config['step']
    experiment = sample_config['experiment']
    step_label = f'{int(step):07d}' if str(step).isdigit() else str(step)
    if Path(experiment).name != experiment or experiment in ('.', '..'):
        raise ValueError('Cached experiment must be a single folder name.')
    output_root = Path(args.output_root).expanduser().resolve()
    run_label = f"{sample_config['split']}_n{len(indices)}_seed{seed}_steps{sample_config['arguments'].get('num_steps', 50)}_{sample_config['weights']}"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_root / experiment / f'step_{step_label}' / run_label
    if output_dir.exists():
        raise FileExistsError(f'Results already exist: {output_dir}. Choose another --output-dir.')
    if output_dir == samples_dir or samples_dir in output_dir.parents:
        raise ValueError('Write metrics outside the immutable sampling cache.')

    # Feature models are metric dependencies only. No checkpoint/raw-data paths are opened.
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
            'preprocessing': 'saved uint16 grayscale->RGB uint8; backend TF-compatible bilinear299 and scaling; pool3',
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
            'preprocessing': 'saved uint16 grayscale->RGB uint8 grid; bicubic224 antialias; ImageNet normalization; CLS without extra L2 normalization',
        }
    identity = IdentityMetrics(args.identity_metrics, device, args.torch_hub_dir) if identity_enabled else None
    statistics = {name: {
        'dimension': extractor.dimension, 'real': FeatureMoments(extractor.dimension),
        'outputs': {task: FeatureMoments(extractor.dimension) for task in ['generation', *BASELINES]},
        'targets': {}, 'counterfactual': {},
    } for name, extractor in extractors.items()}
    arguments = dict(sample_config['arguments'])
    arguments.update(vars(args))
    for key in ('weights', 'split'):
        arguments[key] = sample_config[key]
    for key, default in (('num_samples', len(indices)), ('num_steps', 50), ('seed', seed),
                         ('cfg_scale', 1.), ('heun', False), ('target_view', None),
                         ('target_density', None), ('target_scanner', None), ('target_age', None)):
        arguments[key] = sample_config['arguments'].get(key, default)
    metadata = {key: sample_config[key] for key in (
        'checkpoint', 'step', 'experiment', 'weights', 'vae_source', 'model', 'resolution',
        'split', 'split_csv_sha256', 'num_images', 'indices_sha256')}
    for key in ('checkpoint_bytes', 'checkpoint_mtime_ns', 'split_csv', 'image_storage',
                'sampling_source_sha256', 'reference_images_sha256'):
        if key in sample_config:
            metadata[key] = sample_config[key]
    cache_hashes = {name: sha256_file(samples_dir / name) for name in ('config.json', 'indices.json', 'samples.csv')}
    metadata.update({
        'status': 'running', 'arguments': arguments, 'feature_extractors': feature_metadata,
        'torch_version': torch.__version__, 'samples_dir': str(samples_dir),
        'sampling_files_sha256': cache_hashes,
        'inference': 'Saved-image evaluation; ' + sample_config.get('inference', 'sampling protocol recorded in sampling config'),
        'notes': ['Metrics consume saved uint16 grayscale PNGs using fixed /65535 scaling.',
                  'Identity metrics measure image similarity; low distance alone does not demonstrate a successful counterfactual.',
                  'CF references are observational target groups; other covariates can differ.',
                  'Target-weighted FD means are not pooled FIDs; low sample support is flagged.',
                  'DINOv2 is also the REPA teacher; compare with Inception and identity scores.'],
        'source_sha256': {name: sha256_file(ROOT / name) for name in [
            'evaluation/evaluate_all.py', 'evaluation/identity_metrics.py', 'evaluation/feature_metrics.py',
            'evaluation/frechet_metrics.py', 'sample/sample_io.py', 'loss/lpips.py'] if (ROOT / name).is_file()},
    })
    signature = dict(metadata)
    signature.pop('status')
    signature['arguments'] = {key: value for key, value in arguments.items()
                              if key not in ('output_root', 'output_dir', 'samples_dir', 'save_examples')}
    metadata['evaluation_id'] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'indices.json').write_text(json.dumps(selection, indent=2) + '\n')
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    identity_rows, eligible_counts = [], {attribute: 0 for attribute in args.attributes}
    example_counts = {attribute: 0 for attribute in args.attributes}

    for items in tqdm(loader, desc='Evaluating saved images'):
        original = stack_images(items, ('real', 'none'), device)
        factual_rows = [item['records'][('real', 'none')] for item in items]
        for name, extractor in extractors.items():
            features = extractor(original)
            statistics[name]['real'].update(features)
            if 'counterfactual' in args.tasks:
                for attribute in args.attributes:
                    update_groups(statistics[name]['targets'], features,
                                  parent_values(factual_rows, attribute, device), attribute, args)
        if 'generation' in args.tasks:
            if extractors:
                generated = stack_images(items, ('generation', 'none'), device)
                for name, extractor in extractors.items():
                    statistics[name]['outputs']['generation'].update(extractor(generated))
        baseline_images, baseline_values = {}, {}
        if 'reconstruction' in args.tasks or ('counterfactual' in args.tasks and identity_enabled):
            for task, comparison in BASELINES.items():
                images = stack_images(items, (task, 'none'), device)
                baseline_images[comparison] = images
                if identity_enabled:
                    values = identity(original, images)
                    baseline_values[comparison] = values
                    for i, item in enumerate(items):
                        identity_rows.append({'index': item['index'], 'comparison': comparison, 'attribute': 'none',
                                              'source': '', 'target': '', **{key: value[i] for key, value in values.items()}})
                if 'reconstruction' in args.tasks:
                    for name, extractor in extractors.items():
                        statistics[name]['outputs'][task].update(extractor(images))
        if 'counterfactual' not in args.tasks:
            continue
        for attribute in args.attributes:
            positions = [i for i, item in enumerate(items) if ('counterfactual', attribute) in item['images']]
            eligible_counts[attribute] += len(positions)
            if not positions:
                continue
            chosen = [items[i] for i in positions]
            cf_rows = [item['records'][('counterfactual', attribute)] for item in chosen]
            cf = stack_images(chosen, ('counterfactual', attribute), device)
            for name, extractor in extractors.items():
                update_groups(statistics[name]['counterfactual'], extractor(cf),
                              parent_values(cf_rows, attribute, device), attribute, args)
            if identity_enabled:
                values_by_comparison = {comparison: {key: [value[p] for p in positions] for key, value in values.items()}
                                        for comparison, values in baseline_values.items()}
                values_by_comparison['cf_vs_input'] = identity(original[positions], cf)
                values_by_comparison['cf_vs_cycle'] = identity(baseline_images['cycle_vs_input'][positions], cf)
                for comparison, values in values_by_comparison.items():
                    for i, item in enumerate(chosen):
                        identity_rows.append({'index': item['index'], 'comparison': comparison, 'attribute': attribute,
                                              'source': float(cf_rows[i]['source']), 'target': float(cf_rows[i]['target']),
                                              **{key: value[i] for key, value in values.items()}})
            count = min(args.save_examples - example_counts[attribute], len(positions))
            if count > 0:
                examples = output_dir / 'examples'
                examples.mkdir(exist_ok=True)
                # A cycle image is cached even when only CF metrics were requested.
                cycle = stack_images(chosen, ('reconstruction_cycle', 'none'), device) if ('reconstruction_cycle', 'none') in chosen[0]['images'] else original[positions]
                save_example(original[positions][:count], cycle[:count], cf[:count],
                             examples / f'{attribute}_{example_counts[attribute]:06d}.png', attribute)
                example_counts[attribute] += count
    if any(sha256_file(samples_dir / name) != digest for name, digest in cache_hashes.items()):
        raise RuntimeError('Sampling metadata changed during evaluation; results will not be registered.')
    metadata['eligible_interventions'] = eligible_counts
    metadata['status'] = 'complete'
    common = {key: metadata[key] for key in ('evaluation_id', 'experiment', 'step', 'checkpoint', 'weights', 'split', 'num_images')}
    common.update(seed=seed, num_steps=arguments['num_steps'], results_dir=str(output_dir))
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
        (folder / 'summary.json').write_text(json.dumps({'metadata': dict(metadata, active_feature_extractor=feature_metadata[name]),
                                                       'distances': rows, 'counterfactual_target_weighted': aggregates}, indent=2, allow_nan=False) + '\n')
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
    print(f'Completed saved-image evaluation: {output_dir}')
    return output_dir


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples-dir', required=True)
    parser.add_argument('--evaluations', nargs='+', choices=['identity', 'inception', 'dino'], default=['identity', 'inception', 'dino'])
    parser.add_argument('--identity-metrics', nargs='+', choices=['mae', 'mse', 'ssim', 'lpips'], default=['mae', 'mse', 'ssim', 'lpips'])
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=None, help='Subset of cached tasks; default evaluates all cached tasks.')
    parser.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=None, help='Subset of cached CF attributes; default uses all cached attributes.')
    parser.add_argument('--age-bin-edges', nargs='+', type=float, default=[0, 40, 50, 60, 70, 80, 101])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--feature-model', choices=['dinov2_vitb14', 'dinov2_vitl14'], default='dinov2_vitb14')
    parser.add_argument('--dino-repo', default=str(ROOT / 'cache/torch/hub/facebookresearch_dinov2_main'))
    parser.add_argument('--dino-weights', default=None)
    parser.add_argument('--inception-weights', default=None)
    parser.add_argument('--torch-hub-dir', default=str(ROOT / 'cache/torch/hub'))
    parser.add_argument('--save-examples', type=int, default=0)
    parser.add_argument('--output-root', default=str(ROOT / 'evaluation/results'))
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args(input_args)
    for name in ('evaluations', 'identity_metrics', 'tasks', 'attributes'):
        if getattr(args, name) is not None:
            setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


if __name__ == '__main__':
    main(parse_args())
