import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.utils import make_grid

from samplers import euler_sampler
from utils import denormalize_latents, preprocess_imgs_vae


def array2grid(x, nrow=None):
    if x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)
    if nrow is None:
        nrow = max(1, round(math.sqrt(x.size(0))))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x[0]
    return x.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()


def _show_image(ax, img):
    img = img.detach().cpu()
    if img.ndim != 3:
        raise ValueError(f'Expected CHW image, got {tuple(img.shape)}.')
    if img.shape[0] == 1:
        ax.imshow(img[0], cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(img.mean(dim=0), cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _show_difference(ax, original, output):
    diff = (output - original).detach().cpu()
    if diff.ndim != 3:
        raise ValueError(f'Expected CHW difference image, got {tuple(diff.shape)}.')
    if diff.shape[0] == 1:
        diff = diff[0]
    else:
        diff = diff.mean(dim=0)
    diff = diff * 255.0
    amax = float(diff.abs().max())
    if amax == 0:
        amax = 1.0
    effect = ax.imshow(diff, cmap='RdBu_r', vmin=-amax, vmax=amax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='4%', pad=0.06)
    cbar = ax.figure.colorbar(effect, cax=cax, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, pad=1)
    ticks = np.linspace(np.ceil(-amax), np.floor(amax), 5)
    cbar.set_ticks(np.round(ticks))


def comparison_figure_to_array(originals, outputs, output_labels):
    originals = originals.detach().cpu()
    outputs = [x.detach().cpu() for x in outputs]
    if len(outputs) != len(output_labels):
        raise ValueError('outputs and output_labels must have the same length.')
    n = originals.shape[0]
    nrows = 1 + 2 * len(outputs)
    fig, axes = plt.subplots(nrows, n, figsize=(2.8 * n, 2.8 * nrows), squeeze=False)
    for j in range(n):
        _show_image(axes[0, j], originals[j])
        if j == 0:
            axes[0, j].set_ylabel('Original', fontsize=10, labelpad=10)
    for i, (output, label) in enumerate(zip(outputs, output_labels)):
        image_row = 1 + 2 * i
        diff_row = image_row + 1
        for j in range(n):
            _show_image(axes[image_row, j], output[j])
            _show_difference(axes[diff_row, j], originals[j], output[j])
            if j == 0:
                axes[image_row, j].set_ylabel(label, fontsize=10, labelpad=10)
                axes[diff_row, j].set_ylabel(f'{label}\n- Original', fontsize=10, labelpad=10)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.995, bottom=0.01, wspace=0.05, hspace=0.18)
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return array


def slice_parents(parents, n):
    return {k: v[:n] for k, v in parents.items()}


def _batch_to_parents(batch, device):
    pa = batch['pa']
    return {
        'age': pa['age'].float().to(device),
        'view': pa['view'].float().to(device),
        'density': pa['density'].float().to(device),
        'scanner': pa['scanner'].float().to(device),
    }


def make_counterfactual_parents(parents, attr):
    cf = {k: v.clone() for k, v in parents.items()}
    if attr == 'view':
        cf['view'] = torch.roll(cf['view'], shifts=1, dims=1)
    elif attr == 'density':
        cf['density'] = torch.roll(cf['density'], shifts=1, dims=1)
    elif attr == 'scanner':
        cf['scanner'] = torch.roll(cf['scanner'], shifts=1, dims=1)
    elif attr == 'age':
        age = cf['age']
        cf['age'] = torch.where(age <= 0.9, age + 0.1, age - 0.1).clamp(0, 1)
    else:
        raise ValueError(f'Unknown counterfactual attribute: {attr}')
    return cf


@torch.no_grad()
def prepare_ema_sampling_inputs(val_dataloader, accelerator, args, in_channels, latent_size):
    local_sample_count = max(1, math.ceil(args.num_sample_images / accelerator.num_processes))
    sample_batch = next(iter(val_dataloader))
    sample_parents = _batch_to_parents(sample_batch, accelerator.device)
    local_sample_count = min(local_sample_count, sample_parents['age'].shape[0])
    sample_parents = slice_parents(sample_parents, local_sample_count)
    sample_gen = torch.Generator(device=accelerator.device)
    sample_gen.manual_seed(args.sample_seed + accelerator.process_index)
    xT = torch.randn(
        (local_sample_count, in_channels, latent_size[0], latent_size[1]),
        generator=sample_gen,
        device=accelerator.device,
    )
    return xT, sample_parents


@torch.no_grad()
def flow_euler(model, x, parents, t_start, t_end, num_steps):
    timesteps = torch.linspace(t_start, t_end, num_steps + 1, device=x.device, dtype=x.dtype)
    for i in range(num_steps):
        t = torch.full((x.shape[0],), timesteps[i], device=x.device, dtype=x.dtype)
        dt = timesteps[i + 1] - timesteps[i]
        velocity = model.inference(x, t, parents)
        x = x + dt * velocity
    return x


@torch.no_grad()
def encode_flow_input(ema, vae, images):
    raw_image = images.repeat(1, 3, 1, 1) * 255.0
    processed_image = preprocess_imgs_vae(raw_image)
    with torch.autocast(device_type=raw_image.device.type, enabled=False):
        posterior, _, _ = vae(processed_image.float())
        z = posterior.mode().float()
    return ema.bn(z)


@torch.no_grad()
def decode_flow_latent(ema, vae, z):
    latents_stats = ema.extract_latents_stats()
    in_channels = z.shape[1]
    latents_scale = latents_stats['latents_scale'].view(1, in_channels, 1, 1)
    latents_bias = latents_stats['latents_bias'].view(1, in_channels, 1, 1)
    z = denormalize_latents(z.float(), latents_scale, latents_bias)
    with torch.autocast(device_type=z.device.type, enabled=False):
        images = vae.decode(z.float()).sample
    images = images.mean(dim=1, keepdim=True)
    return ((images + 1.) / 2.).clamp(0, 1)


@torch.no_grad()
def make_flow_reconstruction_grid(ema, vae, val_dataloader, accelerator, args):
    ema.eval()
    vae.eval()
    batch = next(iter(val_dataloader))
    original = batch['x'].to(accelerator.device)
    parents = _batch_to_parents(batch, accelerator.device)
    n = min(original.shape[0], args.num_recon_images)
    original = original[:n]
    parents = slice_parents(parents, n)
    z0 = encode_flow_input(ema, vae, original)
    zT = flow_euler(ema, z0, parents, t_start=0.0, t_end=1.0, num_steps=args.reconstruction_num_steps)
    z_recon = flow_euler(ema, zT, parents, t_start=1.0, t_end=0.0, num_steps=args.reconstruction_num_steps)
    recon = decode_flow_latent(ema, vae, z_recon)
    originals = accelerator.gather(original.float())
    recons = accelerator.gather(recon.float())
    if not accelerator.is_main_process:
        return None
    n = min(args.num_recon_images, originals.shape[0], recons.shape[0])
    return comparison_figure_to_array(
        originals=originals[:n],
        outputs=[recons[:n]],
        output_labels=['Reconstruction'],
    )


@torch.no_grad()
def make_counterfactual_grid(ema, vae, val_dataloader, accelerator, args):
    ema.eval()
    vae.eval()
    batch = next(iter(val_dataloader))
    original = batch['x'].to(accelerator.device)
    parents = _batch_to_parents(batch, accelerator.device)
    n = min(original.shape[0], args.num_counterfactual_images)
    original = original[:n]
    parents = slice_parents(parents, n)
    z0 = encode_flow_input(ema, vae, original)
    zT = flow_euler(ema, z0, parents, t_start=0.0, t_end=1.0, num_steps=args.counterfactual_num_steps)
    counterfactuals = []
    for attr in ['view', 'density', 'scanner', 'age']:
        cf_parents = make_counterfactual_parents(parents, attr)
        z_cf = flow_euler(ema, zT, cf_parents, t_start=1.0, t_end=0.0, num_steps=args.counterfactual_num_steps)
        counterfactuals.append(decode_flow_latent(ema, vae, z_cf))
    originals = accelerator.gather(original.float())
    counterfactuals = [accelerator.gather(x.float()) for x in counterfactuals]
    if not accelerator.is_main_process:
        return None
    n = min(args.num_counterfactual_images, originals.shape[0], *[x.shape[0] for x in counterfactuals])
    return comparison_figure_to_array(
        originals=originals[:n],
        outputs=[x[:n] for x in counterfactuals],
        output_labels=['View CF', 'Density CF', 'Scanner CF', 'Age CF'],
    )


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
    samples = samples.mean(dim=1, keepdim=True)
    samples = (samples + 1) / 2.
    return accelerator.gather(samples.to(torch.float32))
