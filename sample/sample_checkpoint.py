#!/usr/bin/env python3
r"""Sample and export a checkpoint once; evaluate the saved PNGs separately.

    python sample/sample_checkpoint.py --checkpoint checkpoints/RUN/checkpoints/0065000.pt

Outputs are fixed-scale 16-bit grayscale PNGs, samples.csv, indices.json and
config.json, plus original / counterfactual / difference panels for inspection.
No metric feature models are loaded by this command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sample.sample_io import IMAGE_STORAGE, reference_images_hash, save_json, sha256_file, write_image, write_records

ATTRIBUTES = ('view', 'density', 'scanner', 'age')


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
    import torch
    from models.autoencoder import vae_models
    from models.sit import SiT_models

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


def build_dataset(data_dir, frame, split, height, width):
    from data_handle.embed import EMBED, DatasetConfig, build_image_transform, get_sample

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

    return EvaluationDataset(data_dir, frame, split, parents=list(ATTRIBUTES),
                             transform=build_image_transform(DatasetConfig(
                                 img_height=height, img_width=width, img_channels=1)))



def make_intervention(parents, attribute, args):
    target = getattr(args, 'target_' + attribute)
    if target is None:
        from plot_utils import make_counterfactual_parents
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


def parent_metadata(parents, position):
    return {'age': float(parents['age'][position, 0]) * 100,
            **{key: int(parents[key][position].argmax()) for key in ('view', 'density', 'scanner')}}


def _sample(args):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from plot_utils import decode_flow_latent, flow_euler
    from samplers import euler_sampler
    from utils import preprocess_imgs_vae

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

    data_dir = Path(args.data_dir or config['data_dir']).expanduser()
    split_dir = Path(args.split_dir or config['split_dir']).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    split_file = split_dir / f'{args.split}.csv'
    split_hash = sha256_file(split_file)
    frame = pd.read_csv(split_file, low_memory=False)
    if args.indices_file:
        selection = json.loads(Path(args.indices_file).expanduser().read_text())
        if selection['split'] != args.split or selection['split_csv_sha256'] != split_hash:
            raise ValueError('Indices file belongs to a different split CSV or row ordering.')
        indices = selection['indices']
    else:
        order = torch.randperm(len(frame), generator=torch.Generator().manual_seed(args.seed))
        indices = order[:min(args.num_samples, len(frame))].tolist() if args.num_samples else order.tolist()
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(frame) for index in indices)):
        raise ValueError('Sampling needs nonempty, unique, valid split indices.')
    selection = {'split': args.split, 'split_csv_sha256': split_hash, 'indices': indices}
    dataset = build_dataset(data_dir, frame, args.split, config['height'], config['width'])
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False, pin_memory=device.type == 'cuda')
    experiment = config.get('exp_name') or (checkpoint_path.parent.parent.name
                 if checkpoint_path.parent.name == 'checkpoints' else checkpoint_path.parent.name)
    if Path(experiment).name != experiment or experiment in ('.', '..'):
        raise ValueError('Saved experiment name must be one folder name.')
    step_label = f'{int(step):07d}' if str(step).isdigit() else checkpoint_path.stem
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        output_root / experiment / f'step_{step_label}' /
        f'{args.split}_n{args.num_samples}_seed{args.seed}_steps{args.num_steps}_{args.weights}')
    # Slurm creates logs before Python starts; existing samples are never reused.
    if output_dir.exists() and (not (output_dir / 'logs').is_dir()
            or any(path.name != 'logs' for path in output_dir.iterdir())):
        raise FileExistsError(f'Sample cache already exists: {output_dir}. Choose another --output-dir.')
    source_files = ['sample/sample_checkpoint.py', 'sample/sample_io.py',
                    'sample/visualize_counterfactuals.py', 'plot_utils.py', 'samplers.py',
                    'models/sit.py', 'models/embedder.py', 'models/autoencoder.py', 'data_handle/embed.py', 'utils.py']
    metadata = {
        'status': 'running', 'checkpoint': str(checkpoint_path), 'step': step, 'experiment': experiment,
        'checkpoint_bytes': before.st_size, 'checkpoint_mtime_ns': before.st_mtime_ns,
        'weights': args.weights, 'vae_source': vae_source, 'model': config['model'],
        'resolution': [config['height'], config['width']], 'split': args.split,
        'split_csv': str(split_file.resolve()), 'split_csv_sha256': split_hash,
        'data_dir': str(data_dir.resolve()), 'num_images': len(indices),
        'indices_sha256': hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        'arguments': vars(args), 'image_storage': IMAGE_STORAGE, 'torch_version': torch.__version__,
        'inference': 'FP32, TF32 off, posterior mode, selected saved BN; CF/cycle CFG1; generation requested CFG/Heun',
        'sampling_source_sha256': {name: sha256_file(ROOT / name) for name in source_files},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Claim a log-only folder exclusively, including when two jobs start together.
    (output_dir / '.sampling_started').touch(exist_ok=False)
    save_json(output_dir / 'indices.json', selection)
    save_json(output_dir / 'config.json', metadata)
    records = []
    eligible_counts = {attribute: 0 for attribute in args.attributes}

    def save_batch(task, images, batch_indices, parents, attribute='none', sources=None, targets=None):
        directory = {'real': 'real', 'generation': 'generation',
                     'reconstruction_vae': 'reconstruction/vae',
                     'reconstruction_cycle': 'reconstruction/cycle',
                     'counterfactual': f'counterfactual/{attribute}'}[task]
        for position, index in enumerate(batch_indices):
            relative_path = f'{directory}/{index:08d}.png'
            digest = write_image(output_dir, relative_path, images[position])
            records.append({
                'index': index, 'task': task, 'attribute': attribute,
                'source': '' if sources is None else float(sources[position]),
                'target': '' if targets is None else float(targets[position]),
                'file_path': relative_path, 'sha256': digest,
                'input_path': f'real/{index:08d}.png', **parent_metadata(parents, position),
            })

    print(f'Sampling {len(indices)} images from {experiment}, step {step}; output: {output_dir}', flush=True)
    try:
        for batch in tqdm(loader, desc='Saving samples'):
            original = batch['x'].float().to(device)
            parents = {key: value.float().to(device) for key, value in batch['pa'].items()}
            batch_indices = batch['index'].tolist()
            save_batch('real', original, batch_indices, parents)
            if 'generation' in args.tasks:
                factor = 8 if config['vae'] == 'f8d4' else 16
                shape = (model.in_channels, config['height'] // factor, config['width'] // factor)
                noise = torch.stack([torch.randn(shape, device=device,
                                    generator=torch.Generator(device=device).manual_seed(args.seed + index))
                                     for index in batch_indices])
                z = euler_sampler(model, noise, parents, num_steps=args.num_steps, heun=args.heun,
                                  cfg_scale=args.cfg_scale, path_type=config.get('path_type', 'linear')).float()
                save_batch('generation', decode_flow_latent(model, vae, z), batch_indices, parents)
            if not set(args.tasks) & {'reconstruction', 'counterfactual'}:
                continue
            # The VAE encode and factual inversion are reused for every output.
            raw_z = vae.encode(preprocess_imgs_vae(original.repeat(1, 3, 1, 1) * 255.).float()).mode().float()
            z0 = model.bn(raw_z)
            zT = flow_euler(model, z0, parents, 0., 1., args.num_steps)
            vae_image = ((vae.decode(raw_z).sample.mean(1, keepdim=True) + 1) / 2).clamp(0, 1)
            save_batch('reconstruction_vae', vae_image, batch_indices, parents)
            cycle_z = flow_euler(model, zT, parents, 1., 0., args.num_steps)
            save_batch('reconstruction_cycle', decode_flow_latent(model, vae, cycle_z), batch_indices, parents)
            if 'counterfactual' not in args.tasks:
                continue
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
                sources, targets = parents[attribute][selected], target_parents[attribute]
                sources = sources[:, 0] * 100 if attribute == 'age' else sources.argmax(1)
                targets = targets[:, 0] * 100 if attribute == 'age' else targets.argmax(1)
                save_batch('counterfactual', cf, [batch_indices[position] for position in positions],
                           target_parents, attribute=attribute, sources=sources, targets=targets)
        after = checkpoint_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint changed while sampling; this cache will not be marked complete.')
        write_records(output_dir, records)
        metadata['num_counterfactual_visuals'] = 0
        if args.save_visuals and 'counterfactual' in args.tasks:
            from sample.visualize_counterfactuals import save_counterfactual_visuals
            metadata['num_counterfactual_visuals'] = save_counterfactual_visuals(output_dir, records)
        metadata['eligible_interventions'] = eligible_counts
        metadata['reference_images_sha256'] = reference_images_hash(records)
        metadata['num_saved_images'] = len(records)
        metadata['status'] = 'complete'
        save_json(output_dir / 'config.json', metadata)
        (output_dir / 'COMPLETE').write_text('Sampling complete.\n')
    except BaseException as error:
        metadata['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        metadata['error'] = str(error)
        save_json(output_dir / 'config.json', metadata)
        raise
    print(f'Saved {len(records)} PNGs and samples.csv: {output_dir}', flush=True)
    return output_dir


def main(args):
    import torch
    with torch.no_grad():
        return _sample(args)


def parse_args(argv=None):
    import math
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=['ema', 'model'], default='ema')
    parser.add_argument('--vae-ckpt', default=None)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--split-dir', default=None)
    parser.add_argument('--split', choices=['test', 'valid'], default='test')
    parser.add_argument('--num-samples', type=int, default=5000, help='Requested cohort size; 0 means the whole split.')
    parser.add_argument('--indices-file', default=None)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--cfg-scale', type=float, default=1., help='Generation only; CF/cycle use CFG1.')
    parser.add_argument('--heun', action='store_true', help='Heun correction for generation only.')
    parser.add_argument('--tasks', nargs='+', choices=['generation', 'reconstruction', 'counterfactual'],
                        default=['generation', 'reconstruction', 'counterfactual'])
    parser.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=list(ATTRIBUTES))
    parser.add_argument('--target-view', type=int, choices=range(2), default=None)
    parser.add_argument('--target-density', type=int, choices=range(4), default=None)
    parser.add_argument('--target-scanner', type=int, choices=range(5), default=None)
    parser.add_argument('--target-age', type=float, default=None)
    parser.add_argument('--save-visuals', action=argparse.BooleanOptionalAction, default=True,
                        help='Save original / counterfactual / difference panels (default: enabled).')
    parser.add_argument('--output-root', default=str(ROOT / 'sample_results'))
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.num_steps, args.cpu_threads) < 1 or min(args.num_samples, args.num_workers) < 0:
        parser.error('Batch size, steps and CPU threads must be positive; sample/worker counts nonnegative.')
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 1:
        parser.error('--cfg-scale must be finite and >=1.')
    if args.target_age is not None and not 0 <= args.target_age <= 100:
        parser.error('--target-age must be in [0,100] years.')
    args.tasks = list(dict.fromkeys(args.tasks))
    args.attributes = list(dict.fromkeys(args.attributes))
    return args


if __name__ == '__main__':
    try:
        main(parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Sampling failed: {error}', file=sys.stderr)
        sys.exit(1)
