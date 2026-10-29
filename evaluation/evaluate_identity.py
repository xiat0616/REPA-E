r"""Evaluate input/counterfactual similarity for a saved SiT, REPA or REPA-E model.

From the repository root, for example:
    .venv/bin/python evaluation/evaluate_identity.py \
        --checkpoint checkpoints/EXPERIMENT/checkpoints/0200000.pt \
        --split test --num-samples 256 --num-steps 50

Use the same --indices-file from the first evaluation for subsequent models.
Defaults: EMA SiT, checkpoint VAE when present, FP32, posterior mode, no CFG,
and the same Euler inversion/intervention rules as the training plots.

Outputs: per_sample.csv, summary.csv/json, indices.json, config.json, and optional
example grids. MAE, MSE, SSIM loss (1-SSIM) and LPIPS are image-change proxies:
lower distance alone does not establish patient identity or a successful edit.
SSIM/LPIPS are reported separately rather than combined into an arbitrary score.
LPIPS uses the repository's frozen VGG implementation and cached weights only.
Reference: https://github.com/richzhang/PerceptualSimilarity
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_handle.embed import EMBED, DatasetConfig, build_image_transform, get_sample
from models.autoencoder import vae_models
from models.sit import SiT_models
from plot_utils import (
    comparison_figure_to_array, decode_flow_latent,
    flow_euler, make_counterfactual_parents,
)
from utils import preprocess_imgs_vae

ATTRIBUTES = ('view', 'density', 'scanner', 'age')


class EvaluationDataset(EMBED):
    """Use training preprocessing, but fail instead of silently replacing an image."""
    def _load_image_item(self, idx):
        row = self.df.iloc[idx]
        image, metadata = get_sample(self.root, row, normalize_age=self.normalize_age)
        return image, metadata, row

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # CSV row position is stable across evaluations with identical split files.
        sample['index'] = idx
        return sample


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key in cleaned:
            raise ValueError(f'Duplicate checkpoint key after removing wrappers: {key}')
        cleaned[key] = value
    return cleaned


def load_models(checkpoint, weights, device, vae_path=None):
    """Strict-load SiT and its saved BN; pair a joint checkpoint with its tuned VAE."""
    if 'args' not in checkpoint or weights not in checkpoint:
        raise KeyError(f'Checkpoint must contain args and {weights!r}.')
    config = checkpoint['args']
    config = dict(config) if isinstance(config, dict) else vars(config).copy()
    state = clean_state_dict(checkpoint[weights])
    height, width = int(config['height']), int(config['width'])
    vae_name = config['vae']
    downsample = {'f8d4': 8, 'f16d32': 16}[vae_name]
    if height % downsample or width % downsample:
        raise ValueError('Checkpoint image dimensions do not match the VAE downsampling.')
    # Projectors are needed for strict state loading, but no teacher is needed at inference.
    projector_indices = sorted({int(match.group(1)) for key in state
                                if (match := re.fullmatch(r'projectors\.(\d+)\.4\.weight', key))})
    if projector_indices != list(range(len(projector_indices))):
        raise ValueError('Checkpoint projector indices must be contiguous.')
    z_dims = [state[f'projectors.{index}.4.weight'].shape[0] for index in projector_indices]
    projector_dim = state['projectors.0.0.weight'].shape[0] if z_dims else 2048
    model = SiT_models[config['model']](
        input_size=(height // downsample, width // downsample),
        in_channels=state['x_embedder.proj.weight'].shape[1], z_dims=z_dims,
        projector_dim=projector_dim, encoder_depth=config.get('encoder_depth', 8),
        class_dropout_prob=config.get('cfg_prob', .1),
        bn_momentum=config.get('bn_momentum', .1),
        fused_attn=config.get('fused_attn', True), qk_norm=config.get('qk_norm', False),
    )
    model.load_state_dict(state, strict=True)
    model = model.float().to(device).eval().requires_grad_(False)
    if not torch.isfinite(model.bn.running_var).all() or (model.bn.running_var <= 0).any():
        raise ValueError('Checkpoint has invalid latent BatchNorm variances.')
    vae = vae_models[vae_name]()
    if 'vae' in checkpoint:
        if vae_path is not None:
            raise ValueError('--vae-ckpt applies only to checkpoints without a saved VAE.')
        vae.load_state_dict(clean_state_dict(checkpoint['vae']), strict=True)
        vae_source = 'checkpoint[vae] (jointly trained VAE from this step)'
    else:
        vae_path = Path(vae_path or config['vae_ckpt']).expanduser()
        if not vae_path.is_absolute():
            vae_path = ROOT / vae_path
        # All encode/decode parameters must be present. Extra unused pretrained keys are allowed.
        vae_state = clean_state_dict(torch.load(vae_path, map_location='cpu', weights_only=False))
        loaded = vae.load_state_dict(vae_state, strict=False)
        if loaded.missing_keys:
            raise ValueError(f'Pretrained VAE is missing parameters: {loaded.missing_keys}')
        vae_source = str(vae_path.resolve())
    vae = vae.float().to(device).eval().requires_grad_(False)
    for name, value in list(model.state_dict().items()) + list(vae.state_dict().items()):
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint tensor: {name}')
    return model, vae, config, vae_source


class IdentityMetrics:
    def __init__(self, names, device, torch_hub_dir):
        self.names = names
        self.lpips = None
        if 'lpips' in names:
            # Check both files before constructing LPIPS: evaluation never downloads weights.
            from loss.lpips import LPIPS, MD5_MAP, md5_hash
            lpips_path = ROOT / 'pretrained/lpips/vgg.pth'
            vgg_path = Path(torch_hub_dir) / 'checkpoints/vgg16-397923af.pth'
            if not lpips_path.is_file() or not vgg_path.is_file():
                raise FileNotFoundError(
                    f'LPIPS requires cached {lpips_path} and {vgg_path}. '
                    'Use --metrics mae mse ssim to evaluate without LPIPS.')
            if md5_hash(lpips_path) != MD5_MAP['vgg_lpips']:
                raise ValueError(f'LPIPS weights failed their checksum: {lpips_path}')
            torch.hub.set_dir(str(torch_hub_dir))
            workspace = os.environ.get('WORKSPACE')
            os.environ['WORKSPACE'] = str(ROOT)
            try:
                self.lpips = LPIPS().float().to(device).eval().requires_grad_(False)
            finally:
                if workspace is None:
                    os.environ.pop('WORKSPACE', None)
                else:
                    os.environ['WORKSPACE'] = workspace

    @torch.no_grad()
    def __call__(self, reference, output):
        if reference.shape != output.shape or reference.ndim != 4 or reference.shape[1] != 1:
            raise ValueError('Metrics require matching Bx1xHxW grayscale tensors.')
        for image in (reference, output):
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Metric images must be finite and in [0,1].')
        reference, output = reference.float(), output.float()
        values = {}
        difference = output - reference
        if 'mae' in self.names:
            values['mae'] = difference.abs().flatten(1).mean(1)
        if 'mse' in self.names:
            values['mse'] = difference.square().flatten(1).mean(1)
        if 'ssim' in self.names:
            # Wang et al. convention: Gaussian11/sigma1.5, population covariance,
            # data_range1, valid spatial window (no padded boundary pixels).
            if min(reference.shape[-2:]) < 11:
                raise ValueError('SSIM requires image height/width >=11.')
            coordinates = torch.arange(11, device=reference.device, dtype=torch.float32) - 5
            gaussian = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
            gaussian /= gaussian.sum()
            kernel = torch.outer(gaussian, gaussian)[None, None]
            mu_x, mu_y = F.conv2d(reference, kernel), F.conv2d(output, kernel)
            var_x = F.conv2d(reference.square(), kernel) - mu_x.square()
            var_y = F.conv2d(output.square(), kernel) - mu_y.square()
            covariance = F.conv2d(reference * output, kernel) - mu_x * mu_y
            score = ((2 * mu_x * mu_y + .01 ** 2) * (2 * covariance + .03 ** 2)
                     / ((mu_x.square() + mu_y.square() + .01 ** 2) * (var_x + var_y + .03 ** 2)))
            values['ssim'] = score.flatten(1).mean(1)
            values['ssim_loss'] = 1 - values['ssim']
        if self.lpips is not None:
            # The repository LPIPS implementation expects RGB in [-1,1].
            values['lpips'] = self.lpips(reference.repeat(1, 3, 1, 1) * 2 - 1,
                                         output.repeat(1, 3, 1, 1) * 2 - 1).flatten()
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError('A similarity metric produced NaN/Inf.')
        return {key: value.cpu().tolist() for key, value in values.items()}


def make_intervention(parents, attribute, args):
    target = getattr(args, 'target_' + attribute)
    if target is None:
        # Matches training plots: next category, or age+10 years (age-10 above90).
        changed = make_counterfactual_parents(parents, attribute)
    else:
        changed = {key: value.clone() for key, value in parents.items()}
        if attribute == 'age':
            changed[attribute].fill_(target / 100.)
        else:
            changed[attribute].zero_()
            changed[attribute][:, target] = 1
    eligible = (changed[attribute] - parents[attribute]).abs().flatten(1).amax(1) > 1e-6
    return changed, eligible


def write_summaries(output_dir, rows, metadata):
    metric_names = [name for name in rows[0] if name not in ('index', 'comparison', 'attribute', 'source', 'target')]
    groups = {}
    for row in rows:
        groups.setdefault((row['comparison'], row['attribute']), []).append(row)
    summary_rows = []
    for (comparison, attribute), group in groups.items():
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            summary_rows.append({
                'comparison': comparison, 'attribute': attribute, 'metric': metric,
                'n': len(values), 'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if len(values) > 1 else None,
                'median': float(np.median(values)),
                'direction': 'higher_is_more_similar' if metric == 'ssim' else 'lower_is_more_similar',
            })
    for name, values in [('per_sample.csv', rows), ('summary.csv', summary_rows)]:
        with (output_dir / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output_dir / 'summary.json').write_text(json.dumps({'metadata': metadata, 'metrics': summary_rows}, indent=2, allow_nan=False) + '\n')
    for row in summary_rows:
        if row['comparison'] == 'cf_vs_input':
            print(f"{row['attribute']:8s} {row['metric']:10s} mean={row['mean']:.6f} n={row['n']}")


@torch.no_grad()
def main(args):
    if args.batch_size < 1 or args.num_steps < 1 or args.num_samples < 0 or args.save_examples < 0 or args.num_workers < 0 or args.cpu_threads < 1:
        raise ValueError('Batch size/num-steps must be positive; num-samples/save-examples must be nonnegative.')
    if args.target_age is not None and not 0 <= args.target_age <= 100:
        raise ValueError('--target-age must be in [0,100] years.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    # Use the same inference arithmetic across checkpoints, independent of training AMP/compile.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    before = checkpoint_path.stat()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    model, vae, config, vae_source = load_models(checkpoint, args.weights, device, args.vae_ckpt)
    step = checkpoint.get('steps', checkpoint_path.stem)
    del checkpoint
    data_dir = Path(args.data_dir or config['data_dir']).expanduser()
    if args.data_dir is None and not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    split_dir = Path(args.split_dir or config['split_dir']).expanduser()
    if args.split_dir is None and not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    split_file = split_dir / (args.split + '.csv')
    split_hash = hashlib.sha256(split_file.read_bytes()).hexdigest()
    frame = pd.read_csv(split_file, low_memory=False)
    if frame.empty:
        raise ValueError('Evaluation split is empty.')
    if args.indices_file:
        selection = json.loads(Path(args.indices_file).read_text())
        if selection['split'] != args.split or selection['split_csv_sha256'] != split_hash:
            raise ValueError('Indices file belongs to a different split CSV or row ordering.')
        indices = selection['indices']
    else:
        order = torch.randperm(len(frame), generator=torch.Generator().manual_seed(args.seed))
        count = min(args.num_samples, len(frame)) if args.num_samples else len(frame)
        indices = order[:count].tolist()
    if not indices or any(type(index) is not int or not 0 <= index < len(frame) for index in indices) or len(set(indices)) != len(indices):
        raise ValueError('Selected split indices must be nonempty, unique, and in range.')
    dataset_config = DatasetConfig(img_height=config['height'], img_width=config['width'], img_channels=1)
    dataset = EvaluationDataset(data_dir, frame, args.split,
                                transform=build_image_transform(dataset_config), parents=list(ATTRIBUTES))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False, pin_memory=device.type == 'cuda')
    metrics = IdentityMetrics(args.metrics, device, args.torch_hub_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / 'evaluation/results' / f'{checkpoint_path.parent.parent.name}_{step}_{args.weights}_{args.split}_s{args.seed}_n{len(indices)}_euler{args.num_steps}'
    # Refuse to silently replace a previous evaluation with a different configuration.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'indices.json').write_text(json.dumps({'split': args.split, 'split_csv_sha256': split_hash, 'indices': indices}, indent=2) + '\n')
    metadata = {
        'checkpoint': str(checkpoint_path), 'checkpoint_step': step, 'weights': args.weights,
        'checkpoint_bytes': before.st_size, 'checkpoint_mtime_ns': before.st_mtime_ns,
        'vae_source': vae_source, 'model': config['model'], 'resolution': [config['height'], config['width']],
        'split': args.split, 'split_csv': str(split_file), 'split_csv_sha256': split_hash,
        'num_images': len(indices), 'seed': args.seed, 'num_steps_per_direction': args.num_steps,
        'torch_version': torch.__version__,
        'status': 'running',
        'inference': 'eager FP32, TF32 off, posterior mode, saved BN eval, conditional velocity without CFG',
        'comparison_meaning': {
            'vae_vs_input': 'VAE reconstruction error without flow integration',
            'cycle_vs_input': 'Same-condition inversion and return; no-intervention baseline',
            'cf_vs_input': 'Requested input-to-counterfactual image distance',
            'cf_vs_cycle': 'Counterfactual distance from the matched no-intervention reconstruction',
        },
        'interpretation': 'Image similarity proxies; a successful intervention may legitimately change the image. Low distance alone does not establish identity preservation or intervention effectiveness.',
        'metric_settings': {'image_range': [0, 1], 'ssim': 'Gaussian11, sigma1.5, population covariance, valid crop',
                            'lpips': 'Frozen repository VGG LPIPS; RGB in[-1,1]; cached weights, no downloads'},
        'arguments': vars(args),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['evaluation/evaluate_identity.py', 'plot_utils.py', 'models/sit.py', 'models/embedder.py', 'models/autoencoder.py', 'data_handle/embed.py', 'utils.py', 'loss/lpips.py']},
    }
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    rows = []
    eligible_counts = {attribute: 0 for attribute in args.attributes}
    example_counts = {attribute: 0 for attribute in args.attributes}
    print(f'Evaluating {len(indices)} images; {args.weights} step {step}; output: {output_dir}')
    for batch in tqdm(loader, desc='Identity evaluation'):
        original = batch['x'].float().to(device)
        parents = {key: value.float().to(device) for key, value in batch['pa'].items()}
        # 1). Encode the input once and abduct its latent noise under factual parents.
        # This is encode_flow_input's FP32/posterior-mode path, retaining raw_z
        # so the VAE-only baseline avoids any BN normalize/denormalize error.
        processed = preprocess_imgs_vae(original.repeat(1, 3, 1, 1) * 255.)
        raw_z = vae.encode(processed.float()).mode().float()
        z0 = model.bn(raw_z)
        zT = flow_euler(model, z0, parents, 0., 1., args.num_steps)
        # 2). Reconstruction baselines quantify VAE and numerical inversion errors.
        vae_reconstruction = ((vae.decode(raw_z).sample.mean(1, keepdim=True) + 1) / 2).clamp(0, 1)
        cycle_z = flow_euler(model, zT, parents, 1., 0., args.num_steps)
        cycle = decode_flow_latent(model, vae, cycle_z)
        baseline_values = {}
        for comparison, output in [('vae_vs_input', vae_reconstruction), ('cycle_vs_input', cycle)]:
            values = metrics(original, output)
            baseline_values[comparison] = values
            for index, sample_index in enumerate(batch['index'].tolist()):
                rows.append({'index': sample_index, 'comparison': comparison, 'attribute': 'none', 'source': '', 'target': '',
                             **{key: value[index] for key, value in values.items()}})
        # 3). Reuse the same zT for each single-attribute intervention.
        for attribute in args.attributes:
            cf_parents, eligible = make_intervention(parents, attribute, args)
            selected = torch.where(eligible)[0]
            eligible_counts[attribute] += len(selected)
            if not len(selected):
                continue
            target_parents = {key: value[selected] for key, value in cf_parents.items()}
            cf_z = flow_euler(model, zT[selected], target_parents, 1., 0., args.num_steps)
            cf = decode_flow_latent(model, vae, cf_z)
            sources = parents[attribute][selected]
            targets = cf_parents[attribute][selected]
            sources = sources[:, 0] * 100 if attribute == 'age' else sources.argmax(1)
            targets = targets[:, 0] * 100 if attribute == 'age' else targets.argmax(1)
            # Attribute-specific baselines use exactly the eligible CF cohort.
            # The attribute='none' rows above also retain the whole-subset baseline.
            for comparison, values in baseline_values.items():
                for index, position in enumerate(selected.cpu().tolist()):
                    rows.append({'index': int(batch['index'][position]), 'comparison': comparison, 'attribute': attribute,
                                 'source': float(sources[index]), 'target': float(targets[index]),
                                 **{key: value[position] for key, value in values.items()}})
            # 4). Record per-image/per-intervention distances; do not mix attributes.
            for comparison, reference in [('cf_vs_input', original[selected]), ('cf_vs_cycle', cycle[selected])]:
                values = metrics(reference, cf)
                for index, position in enumerate(selected.cpu().tolist()):
                    rows.append({'index': int(batch['index'][position]), 'comparison': comparison, 'attribute': attribute,
                                 'source': float(sources[index]), 'target': float(targets[index]),
                                 **{key: value[index] for key, value in values.items()}})
            count = min(args.save_examples - example_counts[attribute], len(selected))
            if count > 0:
                from PIL import Image
                examples = output_dir / 'examples'
                examples.mkdir(exist_ok=True)
                grid = comparison_figure_to_array(original[selected][:count], [cycle[selected][:count], cf[:count]],
                                                 ['No-edit cycle', f'{attribute} CF'])
                Image.fromarray(grid).save(examples / f'{attribute}_{example_counts[attribute]:06d}.png')
                example_counts[attribute] += count
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during evaluation; results are not a fixed-step comparison.')
    metadata['eligible_interventions'] = eligible_counts
    metadata['status'] = 'complete'
    write_summaries(output_dir, rows, metadata)
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(f'Saved per_sample.csv and summary.csv/json to {output_dir}')


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Specific checkpoint, e.g. checkpoints/RUN/checkpoints/0200000.pt')
    parser.add_argument('--weights', choices=['ema', 'model'], default='ema')
    parser.add_argument('--vae-ckpt', default=None, help='Frozen VAE path override, only when checkpoint has no saved VAE.')
    parser.add_argument('--data-dir', default=None, help='Override saved raw-image root.')
    parser.add_argument('--split-dir', default=None, help='Override directory containing valid.csv/test.csv.')
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--num-samples', type=int, default=256, help='Seeded subset size;0 means the whole split. Ignored with --indices-file.')
    parser.add_argument('--indices-file', default=None, help='Reuse indices.json from another evaluation for a paired comparison.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=50, help='Euler steps in each direction, matching training CF plots.')
    parser.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=list(ATTRIBUTES))
    parser.add_argument('--target-view', type=int, choices=range(2), default=None)
    parser.add_argument('--target-density', type=int, choices=range(4), default=None)
    parser.add_argument('--target-scanner', type=int, choices=range(5), default=None)
    parser.add_argument('--target-age', type=float, default=None, help='Age in years; default matches training plots (+10, or-10 above90).')
    parser.add_argument('--metrics', nargs='+', choices=['mae', 'mse', 'ssim', 'lpips'], default=['mae', 'mse', 'ssim', 'lpips'])
    parser.add_argument('--torch-hub-dir', default=str(ROOT / 'cache/torch/hub'), help='Cached VGG16 checkpoint directory for LPIPS.')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--save-examples', type=int, default=0, help='Optional number of example images per attribute, saved beside metrics.')
    parser.add_argument('--output-dir', default=None, help='New result directory; existing directories are never overwritten.')
    args = parser.parse_args(input_args)
    args.attributes = list(dict.fromkeys(args.attributes))
    args.metrics = list(dict.fromkeys(args.metrics))
    return args


if __name__ == '__main__':
    main(parse_args())
