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
from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm.auto import tqdm

from data_handle.embed import DatasetConfig, get_embed
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from models.sit import SiT_models
from utils import load_encoders, preprocess_imgs_vae, count_trainable_params
from plot_utils import (
    array2grid, comparison_figure_to_array, generate_ema_samples,
    make_counterfactual_grid, make_flow_reconstruction_grid, prepare_ema_sampling_inputs,
)


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
    return comparison_figure_to_array(originals[:n], [recons[:n]], ['VAE reconstruction'])

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # Set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == 'none' else args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    sample_dir = os.path.join(save_dir, 'samples')
    recon_dir = os.path.join(save_dir, 'reconstructions')
    counterfactual_dir = os.path.join(save_dir, 'counterfactuals')
    validation_path = os.path.join(save_dir, 'validation_metrics.jsonl')

    # Set up the logger and checkpoint/output directories.
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

    # Create model
    if args.vae == 'f8d4':
        assert args.height % 8 == 0 and args.width % 8 == 0, "Image dimensions must be divisible by 8 for f8d4 VAE."
        latent_size = (args.height // 8, args.width // 8)
        in_channels = 4
    elif args.vae == 'f16d32':
        assert args.height % 16 == 0 and args.width % 16 == 0, "Image dimensions must be divisible by 16 for f16d32 VAE."
        latent_size = (args.height // 16, args.width // 16)
        in_channels = 32
    else:
        raise NotImplementedError()

    # Load fixed pretrained encoders (e.g. DINOv2) to provide REPA target features.
    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device, image_size)
    else:
        raise NotImplementedError("Please specify at least one encoder type using --enc-type.")
    
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

    # Make a copy of SiT for EMA; its parameters are updated by averaging, not backward.
    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)

    # Load the trainable VAE; the original REPA-E code maintains EMA for SiT only.
    vae = vae_models[args.vae]().to(device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
    vae.load_state_dict(vae_ckpt, strict=False) # We may not have the projection layer in the VAE

    del vae_ckpt

    if accelerator.is_main_process:
        run_logger.info(f'Total trainable params in VAE: {count_trainable_params(vae):,}')

    # Initialize latent BN from pretrained VAE statistics (paper Sec. 3.2).
    # During joint training, BN tracks the changing VAE latent mean and variance.
    stats_path = Path(args.vae_ckpt).parent / f'embed_{args.vae}_{args.height}x{args.width}_latent_stats.pt'
    latents_stats = torch.load(stats_path, map_location='cpu')
    latents_bias = latents_stats['mean'].squeeze().float().to(device)
    latents_scale = latents_stats['std'].squeeze().float().reciprocal().to(device)

    model.init_bn(
        latents_bias=latents_bias,
        latents_scale=latents_scale,
    )

    # Apply SyncBatchNorm if using distributed training
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # VAE regularization: reconstruction + perceptual + KL + GAN losses.
    # This module also contains the discriminator and computes its separate loss.
    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    if accelerator.is_main_process:
        run_logger.info(f'SiT Parameters: {sum(p.numel() for p in model.parameters()):,}')

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set up separate optimizers for SiT, VAE and the discriminator.
    # The paper uses AdamW with betas=(0.9, 0.999) and a constant learning rate of 1e-4.
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

    # Apply the same linear warmup length to SiT, VAE and discriminator.
    # Each optimizer ramps to its own configured learning rate; 0 disables all warmup.
    if args.lr_warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / args.lr_warmup_steps,
            end_factor=1.0,
            total_iters=args.lr_warmup_steps,
        )
        lr_scheduler_vae = torch.optim.lr_scheduler.LinearLR(
            optimizer_vae,
            start_factor=1.0 / args.lr_warmup_steps,
            end_factor=1.0,
            total_iters=args.lr_warmup_steps,
        )
        lr_scheduler_disc = torch.optim.lr_scheduler.LinearLR(
            optimizer_loss_fn,
            start_factor=1.0 / args.lr_warmup_steps,
            end_factor=1.0,
            total_iters=args.lr_warmup_steps,
        )
    else:
        lr_scheduler = None
        lr_scheduler_vae = None
        lr_scheduler_disc = None

    # Set up dataset
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

    # Prepare models for training
    update_ema(ema, model, decay=0)  # Initialize EMA with exactly the current SiT weights.

    # Start with eval mode for all models.
    model.eval()
    ema.eval()
    vae.eval()

    # Load discriminator checkpoint if provided
    if args.disc_pretrained_ckpt is not None:
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location='cpu')
        vae_loss_fn.discriminator.load_state_dict(disc_ckpt)

        if accelerator.is_main_process:
            run_logger.info(f'Loaded discriminator from {args.disc_pretrained_ckpt}')

    global_step = 0

    # Resume all model and optimizer states; this overrides pretrained initialization.
    if args.resume_current or args.resume_step > 0:
        if args.continue_train_exp_dir is None:
            raise ValueError('--continue-train-exp-dir is required when resuming training.')

        ckpt_name = 'current.pt' if args.resume_current else f'{args.resume_step:07d}.pt'
        ckpt_path = f'{args.continue_train_exp_dir}/checkpoints/{ckpt_name}'
        ckpt = torch.load(ckpt_path, map_location='cpu')

        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        vae.load_state_dict(ckpt['vae'])
        vae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])

        optimizer.load_state_dict(ckpt['opt'])
        optimizer_vae.load_state_dict(ckpt['opt_vae'])
        optimizer_loss_fn.load_state_dict(ckpt['opt_disc'])

        if lr_scheduler is not None:
            if ckpt.get('scheduler') is not None:
                lr_scheduler.load_state_dict(ckpt['scheduler'])
            else:
                # Missing scheduler state: preserve the restored optimizer LR without restarting warmup.
                lr_scheduler = None
                run_logger.info('SiT checkpoint has no scheduler state; keeping the restored learning rate without warmup.')
        if lr_scheduler_vae is not None:
            if ckpt.get('scheduler_vae') is not None:
                lr_scheduler_vae.load_state_dict(ckpt['scheduler_vae'])
            else:
                # Missing scheduler state: preserve the restored optimizer LR without restarting warmup.
                lr_scheduler_vae = None
                run_logger.info('VAE checkpoint has no scheduler state; keeping the restored learning rate without warmup.')
        if lr_scheduler_disc is not None:
            if ckpt.get('scheduler_disc') is not None:
                lr_scheduler_disc.load_state_dict(ckpt['scheduler_disc'])
            else:
                # Missing scheduler state: preserve the restored optimizer LR without restarting warmup.
                lr_scheduler_disc = None
                run_logger.info('discriminator checkpoint has no scheduler state; keeping the restored learning rate without warmup.')
        global_step = ckpt['steps']

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512

    # Model compile for better performance.
    if args.compile:
        model = torch.compile(model, backend='inductor', mode='default')
        vae = torch.compile(vae, backend='inductor', mode='default')
        vae_loss_fn = torch.compile(vae_loss_fn, backend='inductor', mode='default')

    # Prepare models, optimizers and dataloaders for distributed/mixed-precision training.
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
        # Only show the progress bar on the main process.
        disable=not accelerator.is_local_main_process,
    )

    # Prepare EMA sampling inputs for validation and sample generation
    xT, sample_parents = prepare_ema_sampling_inputs(
        val_dataloader, accelerator, args, in_channels, latent_size,
    )

    # Main training loop: REPA-E jointly tunes the VAE and SiT (paper Sec. 3.2).
    # REPA alignment updates both models in separate phases; diffusion loss updates SiT only.
    for epoch in range(args.epochs):
        for batch in train_dataloader:
            raw_image = batch['x'].to(device).repeat(1, 3, 1, 1) * 255.0
            parents = batch_to_parents(batch, device)
            # Extract fixed pretrained features as targets for representation alignment.
            zs = extract_encoder_features(raw_image, encoders, encoder_types, architectures, accelerator)

            vae.train()
            model.train()
            vae_loss_fn.train()

            # Accumulate gradients across microbatches; keep backward outside autocast.
            with accelerator.accumulate(model, vae, vae_loss_fn):
                # 1). Forward pass: VAE
                with accelerator.autocast():
                    # Normalize images to [-1, 1], encode/sample z, then decode the reconstruction.
                    processed_image = preprocess_imgs_vae(raw_image)
                    posterior, z, recon_image = vae(processed_image)

                # 2). Backward pass: VAE, compute the VAE loss, backpropagate, and update the VAE;
                #     then compute the discriminator loss and update the discriminator.
                # loss_kwargs is used for the SiT forward function and reused for both VAE and SiT.
                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )

                # Record time_input and noises from VAE alignment to avoid sampling again for SiT.
                time_input = None
                noises = None

                # Turn off grads for the SiT model (avoid REPA gradient on the SiT model).
                requires_grad(model, False)
                # Avoid BN stats being updated by the VAE.
                model.eval()

                base_model = original_model(accelerator, model, args.compile)
                base_loss_fn = original_model(accelerator, vae_loss_fn, args.compile)

                with accelerator.autocast():
                    # Frozen modules bypass DDP during the VAE-only backward.
                    # Compute VAE regularization losses to preserve reconstruction quality.
                    vae_loss, vae_loss_dict = base_loss_fn(
                        processed_image,
                        recon_image,
                        posterior,
                        global_step,
                        'generator',
                    )

                    vae_loss = vae_loss.mean()

                    # Compute the REPA alignment loss for VAE updates.
                    # Gradients pass through frozen SiT into z and the VAE encoder.
                    loss_kwargs['align_only'] = True

                    vae_align_outputs = base_model(
                        x=z,
                        parents=parents,
                        zs=zs,
                        loss_kwargs=loss_kwargs,
                        time_input=time_input,
                        noises=noises,
                    )

                    # VAE objective: L_REG + vae_align_proj_coeff * L_REPA.
                    vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_outputs['proj_loss'].mean()

                # Save time_input and noises and reuse them for the SiT model forward pass.
                time_input = vae_align_outputs['time_input']
                noises = vae_align_outputs['noises']

                accelerator.backward(vae_loss)

                if accelerator.sync_gradients:
                    # Unscale this optimizer only; REPA-E updates three optimizers.
                    accelerator.unscale_gradients(optimizer=optimizer_vae)
                    grad_norm_vae = torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)

                optimizer_vae.step()
                # Advance warmup once per successful VAE update, not per microbatch.
                if accelerator.sync_gradients and not optimizer_vae.step_was_skipped and lr_scheduler_vae is not None:
                    lr_scheduler_vae.step()
                optimizer_vae.zero_grad(set_to_none=True)

                # Discriminator loss and update (continuation of step 2).
                # The loss module detaches reconstructed images, so this does not update the VAE.
                requires_grad(base_loss_fn.discriminator, True)
                with accelerator.autocast():
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
                    accelerator.unscale_gradients(optimizer=optimizer_loss_fn)
                    grad_norm_disc = torch.nn.utils.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)

                optimizer_loss_fn.step()
                # Advance warmup once per successful discriminator update, not per microbatch.
                if accelerator.sync_gradients and not optimizer_loss_fn.step_was_skipped and lr_scheduler_disc is not None:
                    lr_scheduler_disc.step()
                optimizer_loss_fn.zero_grad(set_to_none=True)

                # Turn the grads back on for the SiT model, and put the model into training mode.
                requires_grad(model, True)
                model.train()

                # 3). Forward pass: SiT
                # Avoid diffusion loss backpropagating to the VAE, so detach z below.
                # Reuse the same time_input and noises sampled during VAE alignment.
                loss_kwargs['align_only'] = False

                with accelerator.autocast():
                    sit_outputs = model(
                        x=z.detach(),
                        parents=parents,
                        zs=zs,
                        loss_kwargs=loss_kwargs,
                        time_input=time_input,
                        noises=noises,
                    )

                    # 4). Compute diffusion loss and REPA alignment loss, backpropagate the SiT loss,
                    #     and update the model. Backward and optimizer steps stay outside autocast.
                    # SiT objective: L_DIFF + proj_coeff * L_REPA.
                    sit_loss = sit_outputs['denoising_loss'].mean() + args.proj_coeff * sit_outputs['proj_loss'].mean()

                accelerator.backward(sit_loss)

                if accelerator.sync_gradients:
                    accelerator.unscale_gradients(optimizer=optimizer)
                    grad_norm_sit = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                # Advance warmup once per successful SiT update, not per microbatch.
                if accelerator.sync_gradients and not optimizer.step_was_skipped and lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 5). Update SiT EMA
                # Update the moving average once per synchronized training update.
                if accelerator.sync_gradients:
                    update_ema(ema, original_model(accelerator, model, args.compile))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Gather training metrics across processes for logging.
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
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/vae_learning_rate': optimizer_vae.param_groups[0]['lr'],
                    'train/disc_learning_rate': optimizer_loss_fn.param_groups[0]['lr'],
                    'epoch': epoch,
                }

                progress_bar.set_postfix(
                    sit_loss=logs['train/sit_loss'],
                    vae_loss=logs['train/vae_loss'],
                )

                accelerator.log(logs, step=global_step)

                # Keep a rolling recovery checkpoint and less frequent numbered snapshots.
                save_current = global_step % args.current_checkpointing_steps == 0
                save_numbered = global_step % args.checkpointing_steps == 0
                if (save_current or save_numbered) and accelerator.is_main_process:
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
                        'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'scheduler_vae': lr_scheduler_vae.state_dict() if lr_scheduler_vae is not None else None,
                        'scheduler_disc': lr_scheduler_disc.state_dict() if lr_scheduler_disc is not None else None,
                        'args': args,
                        'steps': global_step,
                    }

                    checkpoint_paths = []
                    if save_current:
                        checkpoint_paths.append(f'{checkpoint_dir}/current.pt')
                    if save_numbered:
                        checkpoint_paths.append(f'{checkpoint_dir}/{global_step:07d}.pt')

                    for checkpoint_path in checkpoint_paths:
                        temporary_path = f'{checkpoint_path}.tmp'
                        try:
                            torch.save(checkpoint, temporary_path)
                            os.replace(temporary_path, checkpoint_path)
                        finally:
                            if os.path.exists(temporary_path):
                                os.remove(temporary_path)
                        run_logger.info(f'Saved checkpoint to {checkpoint_path}')

                # Evaluate losses and save VAE/flow reconstructions and counterfactuals.
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

                    base_vae = original_model(accelerator, vae, args.compile)
                    vae_recon_grid = make_reconstruction_grid(base_vae, val_dataloader, accelerator, args)
                    flow_recon_grid = make_flow_reconstruction_grid(ema, base_vae, val_dataloader, accelerator, args)
                    counterfactual_grid = make_counterfactual_grid(ema, base_vae, val_dataloader, accelerator, args)

                    if accelerator.is_main_process:
                        append_jsonl(validation_path, {'step': global_step, **val_logs})
                        vae_recon_path = os.path.join(recon_dir, f'vae_recon_step_{global_step:07d}.png')
                        recon_path = os.path.join(recon_dir, f'flow_recon_step_{global_step:07d}.png')
                        counterfactual_path = os.path.join(counterfactual_dir, f'counterfactual_step_{global_step:07d}.png')
                        Image.fromarray(vae_recon_grid).save(vae_recon_path)
                        Image.fromarray(flow_recon_grid).save(recon_path)
                        Image.fromarray(counterfactual_grid).save(counterfactual_path)
                        accelerator.log({
                            'reconstructions/vae': wandb.Image(vae_recon_grid),
                            'reconstructions/flow_inversion': wandb.Image(flow_recon_grid),
                            'counterfactuals/flow_inversion': wandb.Image(counterfactual_grid),
                        }, step=global_step)
                        run_logger.info(f'Validation step {global_step}: {val_logs}')
                        run_logger.info(f'Saved VAE reconstructions to {vae_recon_path}')
                        run_logger.info(f'Saved flow inversion reconstructions to {recon_path}')
                        run_logger.info(f'Saved counterfactuals to {counterfactual_path}')

                # Generate samples with EMA SiT and decode them using the current trained VAE.
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
    parser.add_argument('--reconstruction-num-steps', type=int, default=50)
    parser.add_argument('--num-counterfactual-images', type=int, default=4)
    parser.add_argument('--counterfactual-num-steps', type=int, default=50)
    parser.add_argument('--sample-seed', type=int, default=1234)
    parser.add_argument('--sample-num-steps', type=int, default=50)
    parser.add_argument('--sample-cfg-scale', type=float, default=4.0)
    parser.add_argument('--resume-step', type=int, default=0)
    parser.add_argument('--resume-current', action='store_true', help='Resume checkpoints/current.pt.')
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
    parser.add_argument('--current-checkpointing-steps', type=int, default=5000,
                        help='Overwrite checkpoints/current.pt every N optimizer updates.')
    parser.add_argument('--checkpointing-steps', type=int, default=50000,
                        help='Keep a numbered checkpoint every N optimizer updates.')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lr-warmup-steps', type=int, default=2000,
                        help='Linear warmup updates for SiT, VAE and discriminator; 0 disables all warmup.')
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

    args = parser.parse_args(input_args)
    if min(args.current_checkpointing_steps, args.checkpointing_steps) < 1:
        parser.error('Checkpoint intervals must be positive.')
    if args.resume_current and args.resume_step > 0:
        parser.error('Choose --resume-current or --resume-step, not both.')
    return args


if __name__ == '__main__':
    main(parse_args())
