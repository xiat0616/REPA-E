r"""Generation, reconstruction and counterfactual Inception FID or Frechet DINO distance.

Example from the repository root (Python only; no job submission):
    .venv/bin/python evaluation/evaluate_frechet.py \
        --checkpoint checkpoints/EXPERIMENT/checkpoints/0200000.pt \
        --num-samples 5000 --num-steps 50

Use --indices-file from the first evaluation for subsequent checkpoints.
Use --feature-extractor inception for standard FID with TensorFlow-compatible
InceptionV3 pool3 features (2048 dimensions). The default remains frozen
DINOv2-B/14 CLS features (768 dimensions). Select DINOv2-L/14 and its weights
for a different DINO feature protocol. Each backend has separate output metadata.
Sources: https://github.com/NVlabs/edm2/blob/main/calculate_metrics.py
         https://github.com/layer6ai-labs/dgm-eval
         https://github.com/toshas/torch-fidelity

Generation is conditioned on the selected held-out metadata mixture. Reconstruction
includes both direct VAE reconstruction and the no-edit flow cycle. Counterfactual
FD is stratified by intervention/target, using real images from that target group;
age targets use the fixed --age-bin-edges. The target-weighted summary is a mean
of group distances, not a pooled FD. These group references are observational,
not paired ground-truth counterfactuals. Small-sample FD estimates are biased;
scores with n<=feature_dim are explicitly flagged. Keep sample counts, selection,
feature model, sampling settings and preprocessing fixed across model comparisons.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
from data_handle.embed import DatasetConfig, build_image_transform
from evaluate_identity import ATTRIBUTES, EvaluationDataset, load_models, make_intervention
from frechet_metrics import FeatureMoments, frechet_distance
from plot_utils import decode_flow_latent, flow_euler
from samplers import euler_sampler
from utils import preprocess_imgs_vae


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class DinoFeatures(torch.nn.Module):
    """Local frozen DINOv2 CLS features; no teacher wrappers or downloads."""
    def __init__(self, model_name, repo, weights, device):
        super().__init__()
        if not Path(repo).is_dir() or not Path(weights).is_file():
            raise FileNotFoundError('DINO requires a local repository and pretrained weights; see --dino-repo/--dino-weights.')
        self.encoder = torch.hub.load(str(repo), model_name, source='local', pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder = self.encoder.float().to(device).eval().requires_grad_(False)
        self.dimension = self.encoder.embed_dim
        self.register_buffer('mean', torch.tensor([.485, .456, .406], device=device)[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225], device=device)[None, :, None, None])

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or not torch.isfinite(images).all():
            raise ValueError('Feature images must be finite Bx1xHxW grayscale tensors.')
        if images.min() < 0 or images.max() > 1:
            raise ValueError('Feature images must be in [0,1].')
        # Quantize real and generated images identically before feature resizing.
        images = (images.float() * 255).round().clamp(0, 255) / 255
        images = F.interpolate(images.repeat(1, 3, 1, 1), size=(224, 224), mode='bicubic',
                               align_corners=False, antialias=True)
        features = self.encoder.forward_features((images - self.mean) / self.std)['x_norm_clstoken']
        # Layer-normalized CLS is used directly, with no extra L2 normalization.
        if features.ndim != 2 or features.shape[1] != self.dimension or not torch.isfinite(features).all():
            raise ValueError('Invalid DINO CLS features.')
        return features.float()


class InceptionFeatures(torch.nn.Module):
    """Conventional FID pool3 features from the TensorFlow-compatible Inception."""
    dimension = 2048

    def __init__(self, weights, device):
        super().__init__()
        try:
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        except ImportError as error:
            raise ImportError('Inception FID requires torch-fidelity==0.3.0; install requirements-uv.txt.') from error
        if not Path(weights).is_file():
            raise FileNotFoundError(f'FID Inception weights not found: {weights}. Set --inception-weights to the converted FID checkpoint.')
        self.encoder = FeatureExtractorInceptionV3(
            name='inception-v3-compat', features_list=['2048'],
            feature_extractor_weights_path=str(weights),
        ).float().to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or not torch.isfinite(images).all():
            raise ValueError('Feature images must be finite Bx1xHxW grayscale tensors.')
        if images.min() < 0 or images.max() > 1:
            raise ValueError('Feature images must be in [0,1].')
        images = (images.float().repeat(1, 3, 1, 1) * 255).round().clamp(0, 255).to(torch.uint8)
        # The backend handles TF-compatible bilinear299 resizing and scaling.
        # No external resize or ImageNet normalization is applied here.
        features = self.encoder(images)[0]
        if features.shape != (images.shape[0], self.dimension) or not torch.isfinite(features).all():
            raise ValueError('Invalid Inception pool3 features.')
        return features.float()


def target_groups(values, attribute, age_bin_edges):
    if attribute != 'age':
        return values.argmax(1).long()
    years = values[:, 0] * 100
    edges = torch.tensor(age_bin_edges, device=years.device, dtype=years.dtype)
    groups = torch.bucketize(years.contiguous(), edges, right=True) - 1
    if (groups < 0).any() or (groups >= len(age_bin_edges) - 1).any():
        raise ValueError('An age is outside --age-bin-edges; extend the edges to cover it.')
    return groups


def group_label(attribute, group, edges):
    return f'[{edges[group]:g},{edges[group+1]:g})years' if attribute == 'age' else str(group)


def update_groups(container, features, values, attribute, args):
    groups = target_groups(values, attribute, args.age_bin_edges)
    for group in groups.unique().tolist():
        key = (attribute, int(group))
        container.setdefault(key, FeatureMoments(features.shape[1])).update(features[groups == group])


@torch.no_grad()
def main(args):
    if args.num_steps < 1 or args.batch_size < 1 or args.num_samples < 0 or args.cpu_threads < 1 or args.num_workers < 0:
        raise ValueError('Invalid evaluation sample, step, thread or batch count.')
    if args.target_age is not None and not 0 <= args.target_age <= 100:
        raise ValueError('--target-age must be in [0,100] years.')
    if args.cfg_scale < 1 or not np.isfinite(args.cfg_scale):
        raise ValueError('--cfg-scale must be finite and >=1.')
    edges = np.asarray(args.age_bin_edges)
    if len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError('--age-bin-edges must be finite and strictly increasing.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    path = Path(args.checkpoint).expanduser().resolve()
    before = path.stat()
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model, vae, config, vae_source = load_models(checkpoint, args.weights, device, args.vae_ckpt)
    step = checkpoint.get('steps', path.stem)
    del checkpoint
    data_dir = Path(args.data_dir or config['data_dir']).expanduser()
    split_dir = Path(args.split_dir or config['split_dir']).expanduser()
    if args.data_dir is None and not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if args.split_dir is None and not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    split_file = split_dir / f'{args.split}.csv'
    split_hash = hashlib.sha256(split_file.read_bytes()).hexdigest()
    frame = pd.read_csv(split_file, low_memory=False)
    if args.indices_file:
        selection = json.loads(Path(args.indices_file).read_text())
        if selection['split'] != args.split or selection['split_csv_sha256'] != split_hash:
            raise ValueError('Indices file belongs to a different split CSV/row ordering.')
        indices = selection['indices']
    else:
        order = torch.randperm(len(frame), generator=torch.Generator().manual_seed(args.seed))
        count = min(args.num_samples, len(frame)) if args.num_samples else len(frame)
        indices = order[:count].tolist()
    if len(indices) < 2 or len(set(indices)) != len(indices) or any(type(index) is not int or not 0 <= index < len(frame) for index in indices):
        raise ValueError('FD evaluation requires at least two unique, valid split indices.')
    dataset_config = DatasetConfig(img_height=config['height'], img_width=config['width'], img_channels=1)
    dataset = EvaluationDataset(data_dir, frame, args.split, transform=build_image_transform(dataset_config), parents=list(ATTRIBUTES))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False, pin_memory=device.type == 'cuda')
    if args.feature_extractor == 'inception':
        import importlib.metadata
        import torch_fidelity.feature_extractor_inceptionv3 as inception_source
        feature_weights = Path(args.inception_weights).expanduser().resolve()
        extractor = InceptionFeatures(feature_weights, device).eval()
        feature_name = 'inception_v3_tf_2048'
        metric_name = 'FID (TensorFlow-compatible InceptionV3 pool3)'
        feature_protocol = 'grayscale->RGB uint8 round/clamp; torch-fidelity TF-compatible bilinear299 resize and (x-128)/128; pool3; no ImageNet normalization'
        feature_provenance = {
            'feature_library': 'torch-fidelity',
            'feature_library_version': importlib.metadata.version('torch-fidelity'),
            'feature_source_sha256': {str(inception_source.__file__): sha256_file(inception_source.__file__)},
        }
    else:
        dino_repo = Path(args.dino_repo).expanduser().resolve()
        feature_weights = Path(args.dino_weights or ROOT / f'cache/torch/hub/checkpoints/{args.feature_model}_pretrain.pth').expanduser().resolve()
        extractor = DinoFeatures(args.feature_model, dino_repo, feature_weights, device).eval()
        feature_name = args.feature_model
        metric_name = 'Frechet DINO feature distance (not Inception FID)'
        feature_protocol = 'grayscale->RGB, round to uint8 grid, bicubic224x224 antialias, ImageNet mean/std, layer-normalized CLS, no vector L2 normalization'
        feature_provenance = {
            'feature_repo': str(dino_repo),
            'feature_source_sha256': {str(source.relative_to(dino_repo)): sha256_file(source)
                                      for source in sorted(dino_repo.rglob('*.py'))},
        }
    dimension = extractor.dimension
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / 'evaluation/results' / f'fd_{path.parent.parent.name}_{step}_{args.weights}_{feature_name}_{args.split}_s{args.seed}_n{len(indices)}_steps{args.num_steps}'
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'indices.json').write_text(json.dumps({'split': args.split, 'split_csv_sha256': split_hash, 'indices': indices}, indent=2) + '\n')
    metadata = {
        'checkpoint': str(path), 'checkpoint_step': step, 'weights': args.weights, 'vae_source': vae_source,
        'checkpoint_bytes': before.st_size, 'checkpoint_mtime_ns': before.st_mtime_ns,
        'model': config['model'], 'resolution': [config['height'], config['width']],
        'split_csv': str(split_file), 'split_csv_sha256': split_hash, 'num_images': len(indices),
        'metric': metric_name, 'feature_extractor': args.feature_extractor,
        'feature_model': feature_name, 'feature_dimension': dimension,
        'feature_weights': str(feature_weights), 'feature_weights_sha256': sha256_file(feature_weights),
        'feature_preprocessing': feature_protocol, **feature_provenance,
        'sampling': 'generation: existing Euler/optional Heun ODE sampler with requested CFG; inversion and CF: FP32 Euler with CFG1, as in training plots',
        'num_steps_per_direction': args.num_steps, 'seed': args.seed,
        'covariance': 'streaming float64, sample covariance n-1; symmetric PSD sandwich square root',
        'cf_reference': 'real held-out images from the target category or fixed target-age bin; observational quality proxy',
        'age_bin_edges_years': args.age_bin_edges, 'status': 'running', 'arguments': vars(args),
        'torch_version': torch.__version__,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            'evaluation/evaluate_frechet.py', 'evaluation/frechet_metrics.py', 'evaluation/evaluate_identity.py',
            'plot_utils.py', 'samplers.py', 'models/sit.py', 'models/autoencoder.py', 'data_handle/embed.py']},
        'notes': ['Keep references, budgets, preprocessing and seeds fixed across models.',
                  'FD is sample-size biased. Rank-deficient estimates are flagged; groups with fewer than two images have no score.',
                  'Target-matched CF reference groups can differ in the other attributes; scores may also reflect those observational covariate differences.',
                  'Distribution scores should accompany identity and intervention-effectiveness checks.'],
    }
    if args.feature_extractor == 'dino':
        metadata['notes'].append('DINOv2 is also the REPA teacher; consider the Inception backend for a different feature space.')
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    real = FeatureMoments(dimension)
    generated = {task: FeatureMoments(dimension) for task in ['generation', 'reconstruction_vae', 'reconstruction_cycle']}
    reference_groups, cf_groups = {}, {}
    for batch in tqdm(loader, desc=f'{args.feature_extractor} distribution evaluation'):
        original = batch['x'].float().to(device)
        parents = {key: value.float().to(device) for key, value in batch['pa'].items()}
        # 1). Real references and target-specific reference distributions.
        real_features = extractor(original)
        real.update(real_features)
        if 'counterfactual' in args.tasks:
            for attribute in args.attributes:
                update_groups(reference_groups, real_features, parents[attribute], attribute, args)
        # 2). Fresh generation uses the same held-out metadata mixture, independent noise.
        if 'generation' in args.tasks:
            noise_shape = (model.in_channels, config['height'] // (8 if config['vae'] == 'f8d4' else 16),
                           config['width'] // (8 if config['vae'] == 'f8d4' else 16))
            # Per-row RNG keeps input noise fixed across checkpoints and batch sizes.
            noise = torch.stack([torch.randn(noise_shape, device=device,
                                  generator=torch.Generator(device=device).manual_seed(args.seed + int(index)))
                                 for index in batch['index']])
            z = euler_sampler(model, noise, parents, num_steps=args.num_steps, heun=args.heun,
                              cfg_scale=args.cfg_scale, path_type=config.get('path_type', 'linear')).float()
            generated['generation'].update(extractor(decode_flow_latent(model, vae, z)))
        if 'reconstruction' not in args.tasks and 'counterfactual' not in args.tasks:
            continue
        # 3). Encode once: direct VAE reconstruction and factual inverse/return cycle.
        processed = preprocess_imgs_vae(original.repeat(1, 3, 1, 1) * 255.)
        raw_z = vae.encode(processed.float()).mode().float()
        z0 = model.bn(raw_z)
        zT = flow_euler(model, z0, parents, 0., 1., args.num_steps)
        if 'reconstruction' in args.tasks:
            reconstruction = ((vae.decode(raw_z).sample.mean(1, keepdim=True) + 1) / 2).clamp(0, 1)
            generated['reconstruction_vae'].update(extractor(reconstruction))
            cycle = flow_euler(model, zT, parents, 1., 0., args.num_steps)
            generated['reconstruction_cycle'].update(extractor(decode_flow_latent(model, vae, cycle)))
        # 4). Attribute CFs reuse the same factual zT; score only actual interventions.
        if 'counterfactual' in args.tasks:
            for attribute in args.attributes:
                cf_parents, eligible = make_intervention(parents, attribute, args)
                selected = torch.where(eligible)[0]
                if not len(selected):
                    continue
                target_parents = {key: value[selected] for key, value in cf_parents.items()}
                cf_z = flow_euler(model, zT[selected], target_parents, 1., 0., args.num_steps)
                features = extractor(decode_flow_latent(model, vae, cf_z))
                update_groups(cf_groups, features, target_parents[attribute], attribute, args)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during evaluation.')
    rows = []
    for task, moments in generated.items():
        if moments.count:
            rows.append({'comparison': task, 'attribute': 'none', 'target': 'all',
                         **frechet_distance(real, moments)})
    for (attribute, group), moments in sorted(cf_groups.items()):
        reference = reference_groups.get((attribute, group), FeatureMoments(dimension))
        rows.append({'comparison': 'counterfactual_target', 'attribute': attribute,
                     'target': group_label(attribute, group, args.age_bin_edges),
                     **frechet_distance(reference, moments)})
    if 'counterfactual' in args.tasks:
        for attribute in args.attributes:
            if not any(key[0] == attribute for key in cf_groups):
                unavailable = frechet_distance(real, FeatureMoments(dimension))
                unavailable['status'] = 'no_interventions'
                rows.append({'comparison': 'counterfactual_target', 'attribute': attribute,
                             'target': 'none', **unavailable})
    # A clearly labeled weighted mean provides one summary per attribute.
    # It remains unavailable if any generated target group has insufficient support.
    aggregates = []
    for attribute in args.attributes:
        group_rows = [row for row in rows if row['comparison'] == 'counterfactual_target' and row['attribute'] == attribute
                      and row['status'] != 'no_interventions']
        if not group_rows:
            continue
        valid = [row for row in group_rows if row['score'] is not None]
        complete = len(valid) == len(group_rows)
        count = sum(row['n_generated'] for row in group_rows)
        aggregates.append({'attribute': attribute, 'summary_type': 'generated-count-weighted mean of target-group FDs; not a pooled FD',
                           'score': sum(row['score'] * row['n_generated'] for row in valid) / count if complete else None,
                           'status': 'ok' if complete else 'insufficient_target_support',
                           'n_generated': count, 'n_scored': sum(row['n_generated'] for row in valid),
                           'target_groups': len(group_rows), 'scored_target_groups': len(valid),
                           'rank_deficient': any(row['rank_deficient'] for row in group_rows)})
    metadata['status'] = 'complete'
    metadata['small_sample_estimates_present'] = any(row['score'] is not None and row['low_sample_count'] for row in rows)
    metadata['insufficient_groups_present'] = any(row['score'] is None for row in rows)
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output_dir / 'summary.json').write_text(json.dumps({'metadata': metadata, 'distances': rows, 'counterfactual_target_weighted': aggregates}, indent=2, allow_nan=False) + '\n')
    with (output_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_stats:
        torch.save({'real': real.state_dict(), 'outputs': {key: value.state_dict() for key, value in generated.items()},
                    'real_targets': {str(key): value.state_dict() for key, value in reference_groups.items()},
                    'cf_targets': {str(key): value.state_dict() for key, value in cf_groups.items()}}, output_dir / 'feature_stats.pt')
    for row in rows:
        print(f"{row['comparison']:24s} {row['attribute']:8s} {row['target']:18s} FD={row['score']} real={row['n_real']} generated={row['n_generated']} status={row['status']}")
    print(f'Saved {metric_name} to {output_dir}')


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=['ema', 'model'], default='ema')
    parser.add_argument('--vae-ckpt', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--split-dir', default=None)
    parser.add_argument('--split', choices=['test', 'valid'], default='test')
    parser.add_argument('--num-samples', type=int, default=5000, help='0 selects the whole split. Ignored with --indices-file.')
    parser.add_argument('--indices-file', default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--tasks', nargs='+', choices=['generation', 'reconstruction', 'counterfactual'], default=['generation', 'reconstruction', 'counterfactual'])
    parser.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=list(ATTRIBUTES))
    parser.add_argument('--target-view', type=int, choices=range(2), default=None)
    parser.add_argument('--target-density', type=int, choices=range(4), default=None)
    parser.add_argument('--target-scanner', type=int, choices=range(5), default=None)
    parser.add_argument('--target-age', type=float, default=None)
    parser.add_argument('--age-bin-edges', nargs='+', type=float, default=[0, 40, 50, 60, 70, 80, 101])
    parser.add_argument('--feature-extractor', choices=['dino', 'inception'], default='dino', help='DINO feature distance or conventional Inception FID.')
    parser.add_argument('--inception-weights', default=str(Path(torch.hub.get_dir()) / 'checkpoints/weights-inception-2015-12-05-6726825d.pth'), help='Local TensorFlow-compatible FID weights; no download is performed.')
    parser.add_argument('--feature-model', choices=['dinov2_vitb14', 'dinov2_vitl14'], default='dinov2_vitb14')
    parser.add_argument('--dino-repo', default=str(ROOT / 'cache/torch/hub/facebookresearch_dinov2_main'))
    parser.add_argument('--dino-weights', default=None, help='Default: cache/torch/hub/checkpoints/FEATURE_MODEL_pretrain.pth')
    parser.add_argument('--cfg-scale', type=float, default=1, help='Generation CFG only; CF/cycle match plotting with CFG1.')
    parser.add_argument('--heun', action='store_true', help='Heun correction for fresh generation only.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--save-stats', action='store_true', help='Optionally retain feature means/covariances on this machine.')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args(input_args)
    args.tasks = list(dict.fromkeys(args.tasks))
    args.attributes = list(dict.fromkeys(args.attributes))
    return args


if __name__ == '__main__':
    main(parse_args())
