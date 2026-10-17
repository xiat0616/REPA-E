import argparse
import copy
import json
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import wandb
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from data_handle.embed import DatasetConfig, get_embed
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from models.sit import SiT_models
from samplers import euler_sampler
from utils import load_encoders, denormalize_latents, preprocess_imgs_vae, count_trainable_params


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


def array2grid(x, nrow=None):
    if nrow is None:
        nrow = max(1, round(math.sqrt(x.size(0))))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


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


def batch_to_parents(batch, device):
    pa = batch['pa']
    return {
        'age': pa['age'].to(device),
        'view': pa['view'].argmax(dim=1).long().to(device),
        'density': pa['density'].argmax(dim=1).long().to(device),
        'scanner': pa['scanner'].argmax(dim=1).long().to(device),
    }


def slice_parents(parents, n):
    return {k: v[:n] for k, v in parents.items()}


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
def validate_repae(model, vae, val_dataloader, encoders, encoder_types, architectures, accelerator, args):
    model.eval()
    vae.eval()

    sums = torch.zeros(7, device=accelerator.device, dtype=torch.float32)

    for i, batch in enumerate(val_dataloader):
        if args.validation_batches > 0 and i >= args.validation_batches:
            break

        raw_image = batch['x'].to(accelerator.device).repeat(1, 3, 1, 1) * 255.0
        parents = batch_to_parents(batch, accelerator.device)
        zs = extract_encoder_features(raw_image, encoders, encoder_types, architectures, accelerator)

        with accelerator.autocast():
            processed_image = preprocess_imgs_vae(raw_image)
            posterior, z, recon_image = vae(processed_image)

            recon_l1 = torch.nn.functional.l1_loss(recon_image, processed_image)

            loss_kwargs = dict(
                path_type=args.path_type,
                prediction=args.prediction,
                weighting=args.weighting,
                align_only=True,
            )

            vae_align_outputs = model(
                x=z,
                parents=parents,
                zs=zs,
                loss_kwargs=loss_kwargs,
                time_input=None,
                noises=None,
            )

            time_input = vae_align_outputs['time_input']
            noises = vae_align_outputs['noises']

            loss_kwargs['align_only'] = False

            sit_outputs = model(
                x=z.detach(),
                parents=parents,
                zs=zs,
                loss_kwargs=loss_kwargs,
                time_input=time_input,
                noises=noises,
            )

            denoise = sit_outputs['denoising_loss'].mean()
            proj = sit_outputs['proj_loss'].mean()
            sit_loss = denoise + args.proj_coeff * proj
            vae_align = vae_align_outputs['proj_loss'].mean()

        sums += torch.stack([
            sit_loss.float(),
            denoise.float(),
            proj.float(),
            recon_l1.float(),
            vae_align.float(),
            (recon_l1 + args.vae_align_proj_coeff * vae_align).float(),
            torch.ones((), device=sums.device),
        ])

    gathered = accelerator.gather(sums.unsqueeze(0)).sum(dim=0)
    count = max(gathered[6].item(), 1.0)

    model.train()
    vae.train()

    return {
        'val/sit_loss': gathered[0].item() / count,
        'val/denoising_loss': gathered[1].item() / count,
        'val/proj_loss': gathered[2].item() / count,
        'val/vae_reconstruction_l1': gathered[3].item() / count,
        'val/vae_align_loss': gathered[4].item() / count,
        'val/vae_monitor_loss': gathered[5].item() / count,
    }


@torch.no_grad()
def make_reconstruction_grid(vae, val_dataloader, accelerator, args):
    vae.eval()

    batch = next(iter(val_dataloader))
    raw_image = batch['x'].to(accelerator.device).repeat(1, 3, 1, 1) * 255.0

    with accelerator.autocast():
        processed_image = preprocess_imgs_vae(raw_image)
        _, _, recon_image = vae(processed_image)

    originals = accelerator.gather(((processed_image + 1) / 2.).to(torch.float32))
    recons = accelerator.gather(((recon_image + 1) / 2.).to(torch.float32))

    if not accelerator.is_main_process:
        return None

    n = min(args.num_recon_images, originals.shape[0], recons.shape[0])
    paired = torch.cat([originals[:n], recons[:n]], dim=0)
    return array2grid(paired, nrow=n)


@torch.no_grad()
def generate_ema_samples(ema, vae, xT, parents, in_channels, accelerator, args):
    ema.eval()
    vae.eval()

    samples = euler_sampler(
        ema,
        xT,
        parents,
        num_steps=args.sample_num_steps,
        cfg_scale=args.sample_cfg_scale,
        guidance_low=0.,
        guidance_high=1.,
        path_type=args.path_type,
        heun=False,
    ).to(torch.float32)

    latents_stats = ema.extract_latents_stats()
    latents_scale = latents_stats['latents_scale'].view(1, in_channels, 1, 1)
    latents_bias = latents_stats['latents_bias'].view(1, in_channels, 1, 1)

    samples = vae.decode(denormalize_latents(samples, latents_scale, latents_bias)).sample
    samples = (samples + 1) / 2.

    return accelerator.gather(samples.to(torch.float32))


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
    validation_path = os.path.join(save_dir, 'validation_metrics.jsonl')

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

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

    encoders, encoder_types, architectures = load_encoders(args.enc_type, device, image_size)
    z_dims = [encoder.embed_dim for encoder in encoders]

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

    vae = vae_models[args.vae]().to(device)

    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(vae_ckpt, strict=False)
    del vae_ckpt

    if accelerator.is_main_process:
        run_logger.info(f'Total trainable params in VAE: {count_trainable_params(vae):,}')

    latents_stats = torch.load(args.vae_ckpt.replace('.pt', '-latents-stats.pt'), map_location='cpu')
    latents_scale = latents_stats['latents_scale'].squeeze().to(device)
    latents_bias = latents_stats['latents_bias'].squeeze().to(device)

    model.init_bn(
        latents_bias=latents_bias,
        latents_scale=latents_scale,
    )

    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    if args.disc_pretrained_ckpt is not None:
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vae_loss_fn.discriminator.load_state_dict(disc_ckpt)

        if accelerator.is_main_process:
            run_logger.info(f'Loaded discriminator from {args.disc_pretrained_ckpt}')

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

    optimizer_vae = torch.optim.AdamW(
        vae.parameters(),
        lr=args.vae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_loss_fn = torch.optim.AdamW(
        vae_loss_fn.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

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

    update_ema(ema, model, decay=0)

    model.eval()
    ema.eval()
    vae.eval()

    global_step = 0

    if args.resume_step > 0:
        if args.continue_train_exp_dir is None:
            raise ValueError('--continue-train-exp-dir is required when --resume-step > 0.')

        ckpt_path = f"{args.continue_train_exp_dir}/checkpoints/{args.resume_step:07d}.pt"
        ckpt = torch.load(ckpt_path, map_location='cpu')

        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        vae.load_state_dict(ckpt['vae'])
        vae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])

        optimizer.load_state_dict(ckpt['opt'])
        optimizer_vae.load_state_dict(ckpt['opt_vae'])
        optimizer_loss_fn.load_state_dict(ckpt['opt_disc'])

        global_step = ckpt['steps']

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512

    if args.compile:
        model = torch.compile(model, backend='inductor', mode='default')
        vae = torch.compile(vae, backend='inductor', mode='default')
        vae_loss_fn = torch.compile(vae_loss_fn, backend='inductor', mode='default')

    model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_loss_fn, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        vae,
        vae_loss_fn,
        optimizer,
        optimizer_vae,
        optimizer_loss_fn,
        train_dataloader,
        val_dataloader,
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

    local_sample_count = max(1, math.ceil(args.num_sample_images / accelerator.num_processes))

    sample_batch = next(iter(val_dataloader))
    sample_parents = batch_to_parents(sample_batch, device)

    local_sample_count = min(local_sample_count, sample_parents['age'].shape[0])
    sample_parents = slice_parents(sample_parents, local_sample_count)

    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(args.sample_seed + accelerator.process_index)

    xT = torch.randn(
        (local_sample_count, in_channels, latent_size[0], latent_size[1]),
        generator=sample_gen,
        device=device,
    )

    for epoch in range(args.epochs):
        for batch in train_dataloader:
            raw_image = batch['x'].to(device).repeat(1, 3, 1, 1) * 255.0
            parents = batch_to_parents(batch, device)
            zs = extract_encoder_features(raw_image, encoders, encoder_types, architectures, accelerator)

            vae.train()
            model.train()
            vae_loss_fn.train()

            with accelerator.accumulate([model, vae, vae_loss_fn]), accelerator.autocast():
                processed_image = preprocess_imgs_vae(raw_image)
                posterior, z, recon_image = vae(processed_image)

                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )

                time_input = None
                noises = None

                # VAE update
                requires_grad(model, False)
                model.eval()

                vae_loss, vae_loss_dict = vae_loss_fn(
                    processed_image,
                    recon_image,
                    posterior,
                    global_step,
                    'generator',
                )

                vae_loss = vae_loss.mean()
                loss_kwargs['align_only'] = True

                vae_align_outputs = model(
                    x=z,
                    parents=parents,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_input,
                    noises=noises,
                )

                vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_outputs['proj_loss'].mean()

                time_input = vae_align_outputs['time_input']
                noises = vae_align_outputs['noises']

                accelerator.backward(vae_loss)

                if accelerator.sync_gradients:
                    grad_norm_vae = accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)

                optimizer_vae.step()
                optimizer_vae.zero_grad(set_to_none=True)

                # Discriminator update
                d_loss, d_loss_dict = vae_loss_fn(
                    processed_image,
                    recon_image,
                    posterior,
                    global_step,
                    'discriminator',
                )

                d_loss = d_loss.mean()
                accelerator.backward(d_loss)

                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)

                optimizer_loss_fn.step()
                optimizer_loss_fn.zero_grad(set_to_none=True)

                # SiT update
                requires_grad(model, True)
                model.train()

                loss_kwargs['align_only'] = False

                sit_outputs = model(
                    x=z.detach(),
                    parents=parents,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_input,
                    noises=noises,
                )

                sit_loss = sit_outputs['denoising_loss'].mean() + args.proj_coeff * sit_outputs['proj_loss'].mean()

                accelerator.backward(sit_loss)

                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, original_model(accelerator, model, args.compile))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    'train/sit_loss': accelerator.gather(sit_loss).mean().detach().item(),
                    'train/denoising_loss': accelerator.gather(sit_outputs['denoising_loss']).mean().detach().item(),
                    'train/proj_loss': accelerator.gather(sit_outputs['proj_loss']).mean().detach().item(),
                    'train/grad_norm_sit': accelerator.gather(grad_norm_sit).mean().detach().item(),
                    'train/vae_loss': accelerator.gather(vae_loss).mean().detach().item(),
                    'train/reconstruction_loss': accelerator.gather(vae_loss_dict['reconstruction_loss'].mean()).mean().detach().item(),
                    'train/perceptual_loss': accelerator.gather(vae_loss_dict['perceptual_loss'].mean()).mean().detach().item(),
                    'train/kl_loss': accelerator.gather(vae_loss_dict['kl_loss'].mean()).mean().detach().item(),
                    'train/weighted_gan_loss': accelerator.gather(vae_loss_dict['weighted_gan_loss'].mean()).mean().detach().item(),
                    'train/gan_loss': accelerator.gather(vae_loss_dict['gan_loss'].mean()).mean().detach().item(),
                    'train/vae_align_loss': accelerator.gather(vae_align_outputs['proj_loss'].mean()).mean().detach().item(),
                    'train/d_loss': accelerator.gather(d_loss).mean().detach().item(),
                    'train/grad_norm_vae': accelerator.gather(grad_norm_vae).mean().detach().item(),
                    'train/grad_norm_disc': accelerator.gather(grad_norm_disc).mean().detach().item(),
                    'epoch': epoch,
                }

                progress_bar.set_postfix(
                    sit_loss=logs['train/sit_loss'],
                    vae_loss=logs['train/vae_loss'],
                )

                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    base_model = original_model(accelerator, model, args.compile)
                    base_vae = original_model(accelerator, vae, args.compile)
                    base_loss_fn = original_model(accelerator, vae_loss_fn, args.compile)

                    checkpoint = {
                        'model': base_model.state_dict(),
                        'ema': ema.state_dict(),
                        'vae': base_vae.state_dict(),
                        'discriminator': base_loss_fn.discriminator.state_dict(),
                        'opt': optimizer.state_dict(),
                        'opt_vae': optimizer_vae.state_dict(),
                        'opt_disc': optimizer_loss_fn.state_dict(),
                        'args': args,
                        'steps': global_step,
                    }

                    checkpoint_path = f'{checkpoint_dir}/{global_step:07d}.pt'
                    torch.save(checkpoint, checkpoint_path)

                    run_logger.info(f'Saved checkpoint to {checkpoint_path}')

                if global_step == 1 or global_step % args.validation_steps == 0:
                    val_logs = validate_repae(
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

                    recon_grid = make_reconstruction_grid(
                        vae,
                        val_dataloader,
                        accelerator,
                        args,
                    )

                    if accelerator.is_main_process:
                        append_jsonl(validation_path, {'step': global_step, **val_logs})

                        recon_path = os.path.join(recon_dir, f'recon_step_{global_step:07d}.png')
                        Image.fromarray(recon_grid).save(recon_path)

                        accelerator.log(
                            {'reconstructions/vae': wandb.Image(recon_grid)},
                            step=global_step,
                        )

                        run_logger.info(f'Validation step {global_step}: {val_logs}')
                        run_logger.info(f'Saved VAE reconstructions to {recon_path}')

                if global_step == 1 or global_step % args.sampling_steps == 0:
                    out_samples = generate_ema_samples(
                        ema,
                        original_model(accelerator, vae, args.compile),
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

                        accelerator.log(
                            {'samples/ema_sit_current_vae': wandb.Image(grid)},
                            step=global_step,
                        )

                        run_logger.info(f'Saved EMA SiT samples to {sample_path}')

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        run_logger.info('Done!')

    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='REPA-E training on EMBED')

    parser.add_argument('--output-dir', type=str, default='exps')
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--logging-dir', type=str, default='logs')
    parser.add_argument('--report-to', type=str, default='wandb')
    parser.add_argument('--sampling-steps', type=int, default=5000)
    parser.add_argument('--validation-steps', type=int, default=5000)
    parser.add_argument('--validation-batches', type=int, default=16)
    parser.add_argument('--num-sample-images', type=int, default=16)
    parser.add_argument('--num-recon-images', type=int, default=8)
    parser.add_argument('--sample-seed', type=int, default=1234)
    parser.add_argument('--sample-num-steps', type=int, default=50)
    parser.add_argument('--sample-cfg-scale', type=float, default=4.0)
    parser.add_argument('--resume-step', type=int, default=0)
    parser.add_argument('--continue-train-exp-dir', type=str, default=None)

    parser.add_argument('--model', type=str, default='SiT-B/2', choices=SiT_models.keys())
    parser.add_argument('--encoder-depth', type=int, default=8)
    parser.add_argument('--qk-norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fused-attn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bn-momentum', type=float, default=0.1)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--split-dir', type=str, required=True)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--allow-tf32', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mixed-precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--epochs', type=int, default=1400)
    parser.add_argument('--max-train-steps', type=int, default=400000)
    parser.add_argument('--checkpointing-steps', type=int, default=50000)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--adam-weight-decay', type=float, default=0.)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--path-type', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--prediction', type=str, default='v', choices=['v'])
    parser.add_argument('--cfg-prob', type=float, default=0.1)
    parser.add_argument('--enc-type', type=str, default='dinov2-vit-b')
    parser.add_argument('--proj-coeff', type=float, default=0.5)
    parser.add_argument('--weighting', type=str, default='uniform', choices=['uniform', 'lognormal'])

    parser.add_argument('--vae', type=str, default='f8d4', choices=['f8d4', 'f16d32'])
    parser.add_argument('--vae-ckpt', type=str, default='pretrained/sdvae-f8d4/sdvae-f8d4.pt')
    parser.add_argument('--disc-pretrained-ckpt', type=str, default=None)
    parser.add_argument('--loss-cfg-path', type=str, default='configs/l1_lpips_kl_gan.yaml')
    parser.add_argument('--vae-learning-rate', type=float, default=1e-4)
    parser.add_argument('--disc-learning-rate', type=float, default=1e-4)
    parser.add_argument('--vae-align-proj-coeff', type=float, default=1.5)

    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


if __name__ == '__main__':
    main(parse_args())