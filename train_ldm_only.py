import argparse
import copy
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import torch
import wandb
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm.auto import tqdm

from data_handle.embed import DatasetConfig, get_embed
from models.autoencoder import vae_models
from models.sit import SiT_models
from plot_utils import (
    array2grid,
    generate_ema_samples,
    make_counterfactual_grid,
    make_flow_reconstruction_grid,
    prepare_ema_sampling_inputs,
)
from utils import load_encoders, preprocess_imgs_vae


logger = get_logger(__name__)
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    h, w = x.shape[-2:]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, size=(14 * (h // 16), 14 * (w // 16)), mode='bicubic', align_corners=False)
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
        x = torch.nn.functional.interpolate(x, size=(14 * (h // 16), 14 * (w // 16)), mode='bicubic', align_corners=False)
    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
        x = torch.nn.functional.interpolate(x, size=(14 * (h // 16), 14 * (w // 16)), mode='bicubic', align_corners=False)
    return x


def append_jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('module.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace('module.', '')
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f'{logging_dir}/log.txt')],
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def original_model(accelerator, model, compiled=False):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if compiled and hasattr(model, '_orig_mod') else model


def _snapshot_value(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _snapshot_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_snapshot_value(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_snapshot_value(v) for v in x)
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    return repr(x)


def capture_rng_state(device):
    state = {'cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device).cpu()
    return state


def collect_bad_gradients(model):
    bad_grad_stats = []
    bad_grad_tensors = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        finite_mask = torch.isfinite(grad)
        if finite_mask.all():
            continue
        finite = grad[finite_mask]
        bad_grad_stats.append({
            'name': name,
            'shape': tuple(grad.shape),
            'dtype': str(grad.dtype),
            'finite_absmax': finite.abs().max().item() if finite.numel() else float('nan'),
            'num_nan': torch.isnan(grad).sum().item(),
            'num_posinf': torch.isposinf(grad).sum().item(),
            'num_neginf': torch.isneginf(grad).sum().item(),
            'numel': grad.numel(),
        })
        bad_grad_tensors[name] = grad.cpu()
    return bad_grad_stats, bad_grad_tensors


def save_failure_snapshot(
    accelerator, model, ema, optimizer, lr_scheduler, args, failure_dir, global_step,
    reason, z=None, parents=None, zs=None, raw_image=None, batch=None,
    forward_rng_state=None, losses=None, bad_grad_stats=None, bad_grad_tensors=None,
):
    failing_step = global_step + 1
    rank = accelerator.process_index
    step_dir = os.path.join(failure_dir, f'step_{failing_step:07d}')
    os.makedirs(step_dir, exist_ok=True)

    rank_snapshot = {
        'failing_step': failing_step,
        'rank': rank,
        'reason': reason,
        'mixed_precision': args.mixed_precision,
        'compile': args.compile,
        'fused_attn': args.fused_attn,
        'qk_norm': args.qk_norm,
        'z': _snapshot_value(z),
        'parents': _snapshot_value(parents),
        'zs': _snapshot_value(zs),
        'raw_image': _snapshot_value(raw_image),
        'batch': _snapshot_value(batch),
        'forward_rng_state': _snapshot_value(forward_rng_state),
        'losses': losses or {},
        'bad_grad_stats': bad_grad_stats or [],
        'bad_grad_tensors': bad_grad_tensors or {},
    }
    rank_path = os.path.join(step_dir, f'rank_{rank}.pt')
    torch.save(rank_snapshot, rank_path)
    print(f'[rank{rank}] Saved failure inputs to {rank_path}', flush=True)

    if accelerator.is_main_process:
        base_model = original_model(accelerator, model, args.compile)
        checkpoint = {
            'model': base_model.state_dict(),
            'ema': ema.state_dict(),
            'opt': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'args': args,
            'steps': global_step,
            'failing_step': failing_step,
            'reason': reason,
        }
        checkpoint_path = os.path.join(step_dir, 'model_before_failure.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f'[rank0] Saved pre-update model checkpoint to {checkpoint_path}', flush=True)

    return step_dir


def batch_to_parents(batch, device):
    pa = batch['pa']
    return {
        'age': pa['age'].float().to(device),
        'view': pa['view'].float().to(device),
        'density': pa['density'].float().to(device),
        'scanner': pa['scanner'].float().to(device),
    }


@torch.no_grad()
def extract_encoder_features(raw_image, encoders, encoder_types, architectures, accelerator):
    zs = []
    with accelerator.autocast():
        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
            z = encoder.forward_features(raw_image_)
            if 'mocov3' in encoder_type:
                z = z[:, 1:]
            if 'dinov2' in encoder_type:
                z = z['x_norm_patchtokens']
            zs.append(z)
    return zs


@torch.no_grad()
def encode_vae_latent(vae, raw_image):
    processed_image = preprocess_imgs_vae(raw_image)
    with torch.autocast(device_type=raw_image.device.type, enabled=False):
        _, z, _ = vae(processed_image.float())
    return z.float()


@torch.no_grad()
def validate_sit(model, vae, val_dataloader, encoders, encoder_types, architectures, accelerator, args):
    use_repa = args.proj_coeff > 0
    model.eval()
    vae.eval()
    sums = torch.zeros(4, device=accelerator.device, dtype=torch.float32)

    for i, batch in enumerate(val_dataloader):
        if args.validation_batches > 0 and i >= args.validation_batches:
            break

        raw_image = batch['x'].to(accelerator.device).repeat(1, 3, 1, 1) * 255.0
        parents = batch_to_parents(batch, accelerator.device)
        zs = extract_encoder_features(raw_image, encoders, encoder_types, architectures, accelerator) if use_repa else None
        z = encode_vae_latent(vae, raw_image)

        with accelerator.autocast():
            outputs = model(
                x=z,
                parents=parents,
                zs=zs,
                loss_kwargs=dict(
                    weighting=args.weighting,
                    path_type=args.path_type,
                    prediction=args.prediction,
                    align_only=False,
                ),
            )

            denoise = outputs['denoising_loss'].mean()
            proj = outputs['proj_loss'].mean()
            total = denoise + args.proj_coeff * proj if use_repa else denoise

        sums += torch.stack([total.float(), denoise.float(), proj.float(), torch.ones((), device=sums.device)])

    gathered = accelerator.gather(sums.unsqueeze(0)).sum(dim=0)
    count = max(gathered[3].item(), 1.0)

    model.train()
    original_model(accelerator, model, args.compile).bn.eval()

    return {
        'val/sit_loss': gathered[0].item() / count,
        'val/denoising_loss': gathered[1].item() / count,
        'val/proj_loss': gathered[2].item() / count,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    sample_dir = os.path.join(save_dir, 'samples')
    recon_dir = os.path.join(save_dir, 'reconstructions')
    counterfactual_dir = os.path.join(save_dir, 'counterfactuals')
    failure_dir = os.path.join(save_dir, 'failures')
    validation_path = os.path.join(save_dir, 'validation_metrics.jsonl')
    os.makedirs(failure_dir, exist_ok=True)

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(counterfactual_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        run_logger = create_logger(save_dir)
        run_logger.info(f'Experiment directory created at {save_dir}')
    else:
        run_logger = logger

    device = accelerator.device

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    image_size = (args.height, args.width)

    if args.vae == 'f8d4':
        assert args.height % 8 == 0 and args.width % 8 == 0
        latent_size = (args.height // 8, args.width // 8)
        in_channels = 4
    elif args.vae == 'f16d32':
        assert args.height % 16 == 0 and args.width % 16 == 0
        latent_size = (args.height // 16, args.width // 16)
        in_channels = 32
    else:
        raise NotImplementedError()

    use_repa = args.proj_coeff > 0

    if use_repa:
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, image_size)
        z_dims = [encoder.embed_dim for encoder in encoders]
    else:
        encoders, encoder_types, architectures = [], [], []
        z_dims = []

    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        fused_attn=args.fused_attn,
        qk_norm=args.qk_norm,
    ).to(device)

    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)

    stats_path = Path(args.vae_ckpt).parent / f'embed_{args.vae}_{args.height}x{args.width}_latent_stats.pt'
    if not stats_path.exists():
        raise FileNotFoundError(
            f'Latent statistics not found: {stats_path}\n'
            f'Run compute_embed_latent_stats.py for {args.height}x{args.width} first.'
        )

    latents_stats = torch.load(stats_path, map_location='cpu')
    if 'mean' not in latents_stats or 'std' not in latents_stats:
        raise KeyError(f'Latent statistics file must contain mean and std: {stats_path}')

    latents_bias = latents_stats['mean'].squeeze().float()
    latents_std = latents_stats['std'].squeeze().float()

    if latents_bias.numel() != in_channels or latents_std.numel() != in_channels:
        raise ValueError(
            f'Latent statistics channel mismatch: model expects {in_channels} channels, '
            f'but stats contain mean={latents_bias.numel()}, std={latents_std.numel()}.'
        )

    if not torch.isfinite(latents_bias).all() or not torch.isfinite(latents_std).all() or (latents_std <= 0).any():
        raise ValueError(f'Invalid latent statistics in {stats_path}')

    latents_bias = latents_bias.to(device)
    latents_scale = latents_std.reciprocal().to(device)
    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

    if accelerator.is_main_process:
        run_logger.info(f'Loaded latent statistics from {stats_path}')
        run_logger.info(f'Latent bias: {latents_bias.detach().cpu()}')
        run_logger.info(f'Latent scale: {latents_scale.detach().cpu()}')

    vae = vae_models[args.vae]().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_ckpt, strict=False)
    del vae_ckpt

    requires_grad(vae, False)
    vae.eval()
    vae.float()

    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if accelerator.is_main_process:
        run_logger.info(f'SiT Parameters: {sum(p.numel() for p in model.parameters()):,}')

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.lr_warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / args.lr_warmup_steps,
            end_factor=1.0,
            total_iters=args.lr_warmup_steps,
        )
    else:
        lr_scheduler = None

    dataset_cfg = DatasetConfig(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        cache_dir=None,
        parents=['age', 'view', 'density', 'scanner'],
        img_height=args.height,
        img_width=args.width,
        img_channels=1,
    )

    datasets = get_embed(dataset_cfg)
    local_batch_size = int(args.batch_size // accelerator.num_processes)

    if local_batch_size < 1 or args.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f'--batch-size ({args.batch_size}) must be divisible by number of processes '
            f'({accelerator.num_processes}).'
        )

    train_dataloader = DataLoader(
        datasets['train'],
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        datasets['valid'],
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if accelerator.is_main_process:
        run_logger.info(f"Train: {len(datasets['train']):,} | Valid: {len(datasets['valid']):,}")
        run_logger.info(f'Global batch: {args.batch_size} | Local batch/GPU: {local_batch_size}')

    update_ema(ema, model, decay=0)
    model.eval()
    ema.eval()

    global_step = 0

    if args.resume_step > 0:
        if args.continue_train_exp_dir is None:
            raise ValueError('--continue-train-exp-dir is required when --resume-step > 0.')

        ckpt_path = f'{args.continue_train_exp_dir}/checkpoints/{args.resume_step:07d}.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu')

        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        if lr_scheduler is not None and ckpt.get('scheduler') is not None:
            lr_scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt['steps']

        # Keep latent normalization tied to the current dataset/resolution stats.
        model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)
        ema.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512

    if args.compile:
        model = torch.compile(model, backend='inductor', mode='default')

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name='gradient-pass-through',
            config=vars(copy.deepcopy(args)),
            init_kwargs={'wandb': {'name': args.exp_name}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc='Steps',
        disable=not accelerator.is_local_main_process,
    )

    xT, sample_parents = prepare_ema_sampling_inputs(
        val_dataloader, accelerator, args, in_channels, latent_size
    )

    for epoch in range(args.epochs):
        for batch in train_dataloader:
            raw_image = batch['x'].to(device).repeat(1, 3, 1, 1) * 255.0
            parents = batch_to_parents(batch, device)

            zs = (
                extract_encoder_features(raw_image, encoders, encoder_types, architectures, accelerator)
                if use_repa else None
            )

            z = encode_vae_latent(vae, raw_image)

            if not torch.isfinite(z).all():
                reason = f'Non-finite VAE latent at step {global_step + 1}.'
                save_failure_snapshot(
                    accelerator, model, ema, optimizer, lr_scheduler, args, failure_dir, global_step,
                    reason=reason, z=z, parents=parents, zs=zs, raw_image=raw_image, batch=batch,
                )
                raise FloatingPointError(reason)

            with accelerator.accumulate(model), accelerator.autocast():
                model.train()
                base_model = original_model(accelerator, model, args.compile)
                base_model.bn.eval()

                # Save the exact RNG state immediately before SiT forward. The flow model samples
                # t/noise internally, so restoring this state with the saved z/parents reproduces
                # the same stochastic training example without modifying models/sit.py.
                forward_rng_state = capture_rng_state(device)

                sit_outputs = model(
                    x=z,
                    parents=parents,
                    zs=zs,
                    loss_kwargs=dict(
                        weighting=args.weighting,
                        path_type=args.path_type,
                        prediction=args.prediction,
                        align_only=False,
                    ),
                )

                denoising_loss = sit_outputs['denoising_loss'].mean()
                proj_loss = sit_outputs['proj_loss'].mean()
                sit_loss = denoising_loss + args.proj_coeff * proj_loss if use_repa else denoising_loss
                losses = {
                    'sit_loss': sit_loss.detach().float().item(),
                    'denoising_loss': denoising_loss.detach().float().item(),
                    'proj_loss': proj_loss.detach().float().item(),
                }

                if not torch.isfinite(sit_loss):
                    reason = (
                        f'Non-finite loss at step {global_step + 1}: '
                        f'denoise={denoising_loss.item()}, proj={proj_loss.item()}'
                    )
                    save_failure_snapshot(
                        accelerator, model, ema, optimizer, lr_scheduler, args, failure_dir, global_step,
                        reason=reason, z=z, parents=parents, zs=zs, raw_image=raw_image, batch=batch,
                        forward_rng_state=forward_rng_state, losses=losses,
                    )
                    raise FloatingPointError(reason)

                accelerator.backward(sit_loss)

                base_model = original_model(accelerator, model, args.compile)
                bad_grad_stats, bad_grad_tensors = collect_bad_gradients(base_model)
                if bad_grad_stats:
                    first = bad_grad_stats[0]
                    reason = (
                        f'Non-finite gradients at step {global_step + 1}: {len(bad_grad_stats)} parameter(s); '
                        f'first={first["name"]}, shape={first["shape"]}, finite_absmax={first["finite_absmax"]}, '
                        f'nan={first["num_nan"]}, +inf={first["num_posinf"]}, -inf={first["num_neginf"]}'
                    )
                    for item in bad_grad_stats:
                        print(f'[rank{accelerator.process_index}] bad_grad: {item}', flush=True)
                    save_failure_snapshot(
                        accelerator, model, ema, optimizer, lr_scheduler, args, failure_dir, global_step,
                        reason=reason, z=z, parents=parents, zs=zs, raw_image=raw_image, batch=batch,
                        forward_rng_state=forward_rng_state, losses=losses,
                        bad_grad_stats=bad_grad_stats, bad_grad_tensors=bad_grad_tensors,
                    )
                    raise FloatingPointError(reason)

                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if not torch.isfinite(torch.as_tensor(grad_norm_sit, device=device)):
                        reason = f'Non-finite gradient norm at step {global_step + 1}: {grad_norm_sit}'
                        save_failure_snapshot(
                            accelerator, model, ema, optimizer, lr_scheduler, args, failure_dir, global_step,
                            reason=reason, z=z, parents=parents, zs=zs, raw_image=raw_image, batch=batch,
                            forward_rng_state=forward_rng_state, losses=losses,
                        )
                        raise FloatingPointError(reason)

                optimizer.step()

                if accelerator.sync_gradients and lr_scheduler is not None:
                    lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, original_model(accelerator, model, args.compile))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    'train/sit_loss': accelerator.gather(sit_loss.detach().reshape(1)).mean().item(),
                    'train/denoising_loss': accelerator.gather(sit_outputs['denoising_loss'].detach()).mean().item(),
                    'train/proj_loss': accelerator.gather(proj_loss.detach().reshape(1)).mean().item(),
                    'train/grad_norm_sit': accelerator.gather(
                        torch.as_tensor(grad_norm_sit, device=device).detach().reshape(1)
                    ).mean().item(),
                    'epoch': epoch,
                }

                progress_bar.set_postfix(
                    sit_loss=logs['train/sit_loss'],
                    denoise=logs['train/denoising_loss'],
                    proj=logs['train/proj_loss'],
                )

                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    base_model = original_model(accelerator, model, args.compile)

                    checkpoint = {
                        'model': base_model.state_dict(),
                        'ema': ema.state_dict(),
                        'opt': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'args': args,
                        'steps': global_step,
                    }

                    checkpoint_path = f'{checkpoint_dir}/{global_step:07d}.pt'
                    torch.save(checkpoint, checkpoint_path)
                    run_logger.info(f'Saved checkpoint to {checkpoint_path}')

                if global_step == 1 or global_step % args.validation_steps == 0:
                    val_logs = validate_sit(
                        model,
                        vae,
                        val_dataloader,
                        encoders,
                        encoder_types,
                        architectures,
                        accelerator,
                        args,
                    )

                    accelerator.log(val_logs, step=global_step)

                    flow_recon_grid = make_flow_reconstruction_grid(
                        ema,
                        vae,
                        val_dataloader,
                        accelerator,
                        args,
                    )

                    counterfactual_grid = make_counterfactual_grid(
                        ema,
                        vae,
                        val_dataloader,
                        accelerator,
                        args,
                    )

                    if accelerator.is_main_process:
                        append_jsonl(validation_path, {'step': global_step, **val_logs})

                        recon_path = os.path.join(
                            recon_dir,
                            f'flow_recon_step_{global_step:07d}.png',
                        )
                        Image.fromarray(flow_recon_grid).save(recon_path)

                        counterfactual_path = os.path.join(
                            counterfactual_dir,
                            f'counterfactual_step_{global_step:07d}.png',
                        )
                        Image.fromarray(counterfactual_grid).save(counterfactual_path)

                        accelerator.log(
                            {
                                'reconstructions/flow_inversion': wandb.Image(flow_recon_grid),
                                'counterfactuals/flow_inversion': wandb.Image(counterfactual_grid),
                            },
                            step=global_step,
                        )

                        run_logger.info(f'Validation step {global_step}: {val_logs}')
                        run_logger.info(f'Saved flow inversion reconstructions to {recon_path}')
                        run_logger.info(f'Saved counterfactuals to {counterfactual_path}')

                if global_step == 1 or global_step % args.sampling_steps == 0:
                    out_samples = generate_ema_samples(
                        ema,
                        vae,
                        xT,
                        sample_parents,
                        in_channels,
                        accelerator,
                        args,
                    )

                    if accelerator.is_main_process:
                        out_samples = out_samples[:args.num_sample_images]
                        grid = array2grid(out_samples)
                        sample_path = os.path.join(sample_dir, f'ema_step_{global_step:07d}.png')
                        Image.fromarray(grid).save(sample_path)
                        accelerator.log({'samples/ema': wandb.Image(grid)}, step=global_step)
                        run_logger.info(f'Saved EMA samples to {sample_path}')

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        run_logger.info('Done!')

    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='Latent DiT / REPA training on EMBED')

    parser.add_argument('--output-dir', type=str, default='exps')
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--logging-dir', type=str, default='logs')
    parser.add_argument('--report-to', type=str, default='wandb')
    parser.add_argument('--sampling-steps', type=int, default=5000)
    parser.add_argument('--validation-steps', type=int, default=5000)
    parser.add_argument('--validation-batches', type=int, default=16)
    parser.add_argument('--num-sample-images', type=int, default=16)
    parser.add_argument('--num-recon-images', type=int, default=8)
    parser.add_argument('--reconstruction-num-steps', type=int, default=50)
    parser.add_argument('--num-counterfactual-images', type=int, default=4)
    parser.add_argument('--counterfactual-num-steps', type=int, default=50)
    parser.add_argument('--sample-seed', type=int, default=1234)
    parser.add_argument('--sample-num-steps', type=int, default=50)
    parser.add_argument('--sample-cfg-scale', type=float, default=1.0)
    parser.add_argument('--resume-step', type=int, default=0)
    parser.add_argument('--continue-train-exp-dir', type=str, default=None)

    parser.add_argument('--model', type=str, default='SiT-B/2', choices=SiT_models.keys())
    parser.add_argument('--encoder-depth', type=int, default=8)
    parser.add_argument('--qk-norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fused-attn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bn-momentum', type=float, default=0.1)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--split-dir', type=str, required=True)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--allow-tf32', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mixed-precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--epochs', type=int, default=1400)
    parser.add_argument('--max-train-steps', type=int, default=400000)
    parser.add_argument('--checkpointing-steps', type=int, default=5000)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lr-warmup-steps', type=int, default=2000)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--adam-weight-decay', type=float, default=0.)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)

    parser.add_argument('--path-type', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--prediction', type=str, default='v', choices=['v'])
    parser.add_argument('--cfg-prob', type=float, default=0.1)
    parser.add_argument('--enc-type', type=str, default='dinov2-vit-b')
    parser.add_argument('--proj-coeff', type=float, default=0.5)
    parser.add_argument('--weighting', type=str, default='uniform', choices=['uniform', 'lognormal'])

    parser.add_argument('--vae', type=str, default='f8d4', choices=['f8d4', 'f16d32'])
    parser.add_argument('--vae-ckpt', type=str, default='pretrained/sdvae-f8d4/sdvae-f8d4.pt')

    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


if __name__ == '__main__':
    main(parse_args())