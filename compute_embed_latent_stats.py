import argparse
import os

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data_handle.embed import DatasetConfig, get_embed
from models.autoencoder import vae_models
from utils import preprocess_imgs_vae


@torch.no_grad()
def encode_vae_latent(vae, raw_image, accelerator, vae_fp32=True, posterior_mode="sample"):
    processed_image = preprocess_imgs_vae(raw_image)

    if vae_fp32:
        with torch.autocast(device_type=raw_image.device.type, enabled=False):
            posterior, sampled_z, _ = vae(processed_image.float())
    else:
        with accelerator.autocast():
            posterior, sampled_z, _ = vae(processed_image)

    if posterior_mode == "sample":
        z = sampled_z
    elif posterior_mode == "mean":
        z = posterior.mode()
    else:
        raise ValueError(f"Unknown posterior mode: {posterior_mode}")

    return z.float()


@torch.no_grad()
def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    set_seed(args.seed + rank)

    dataset_cfg = DatasetConfig(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        cache_dir=None,
        parents=["age", "view", "density", "scanner"],
        img_height=args.height,
        img_width=args.width,
        img_channels=1,
    )

    datasets = get_embed(dataset_cfg)
    train_dataset = datasets["train"]

    # Exact non-overlapping split across GPUs.
    indices = range(rank, len(train_dataset), world_size)
    local_dataset = Subset(train_dataset, indices)

    loader = DataLoader(
        local_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    vae = vae_models[args.vae]().to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt, strict=False)
    del ckpt

    vae.requires_grad_(False)
    vae.eval()

    if args.vae_fp32:
        vae.float()

    if args.vae == "f8d4":
        channels = 4
    elif args.vae == "f16d32":
        channels = 32
    else:
        raise NotImplementedError(args.vae)

    channel_sum = torch.zeros(channels, device=device, dtype=torch.float64)
    channel_sq_sum = torch.zeros(channels, device=device, dtype=torch.float64)
    channel_count = torch.zeros((), device=device, dtype=torch.float64)
    image_count = torch.zeros((), device=device, dtype=torch.float64)

    iterator = tqdm(
        loader,
        desc=f"Computing latent stats rank {rank}",
        disable=not accelerator.is_local_main_process,
    )

    for batch in iterator:
        raw_image = batch["x"].to(device, non_blocking=True).repeat(1, 3, 1, 1) * 255.0

        z = encode_vae_latent(
            vae,
            raw_image,
            accelerator,
            vae_fp32=args.vae_fp32,
            posterior_mode=args.vae_posterior_mode,
        )

        if not torch.isfinite(z).all():
            raise FloatingPointError("VAE produced NaN/Inf while computing latent statistics.")

        z = z.double()

        # z: [B, C, H, W]
        channel_sum += z.sum(dim=(0, 2, 3))
        channel_sq_sum += z.square().sum(dim=(0, 2, 3))
        channel_count += z.shape[0] * z.shape[2] * z.shape[3]
        image_count += z.shape[0]

    if world_size > 1:
        dist.all_reduce(channel_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(channel_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(channel_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(image_count, op=dist.ReduceOp.SUM)

    mean = channel_sum / channel_count
    variance = channel_sq_sum / channel_count - mean.square()
    variance = variance.clamp_min(1e-12)
    std = variance.sqrt()

    mean = mean.float().cpu()
    std = std.float().cpu()

    if accelerator.is_main_process:
        print("\n============================================================")
        print("EMBED PER-CHANNEL LATENT STATISTICS")
        print("============================================================")
        print(f"images     : {int(image_count.item())}")
        print(f"resolution : {args.height}x{args.width}")
        print(f"VAE        : {args.vae}")
        print(f"posterior  : {args.vae_posterior_mode}")
        print(f"VAE FP32   : {args.vae_fp32}")
        print()
        print("mean =", mean)
        print("std  =", std)

        output = args.output
        if output is None:
            output = os.path.join(
                os.path.dirname(args.vae_ckpt),
                f"embed_{args.vae}_{args.height}x{args.width}_latent_stats.pt",
            )

        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        torch.save({"mean": mean, "std": std}, output)
        print(f"\nSaved statistics to: {output}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split-dir", type=str, required=True)

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)

    parser.add_argument(
        "--vae",
        type=str,
        default="f8d4",
        choices=["f8d4", "f16d32"],
    )

    parser.add_argument(
        "--vae-ckpt",
        type=str,
        required=True,
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument(
        "--vae-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--vae-posterior-mode",
        type=str,
        default="sample",
        choices=["sample", "mean"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())