"""Replay one saved SiT failure without updating parameters or loading the dataset.

Run from the remote REPA-E repository on an allocated GPU:
  .venv/bin/python debug_failure_precision.py --failure-dir <failure directory>
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys

import torch


def stats(t):
    t = t.detach()
    good = torch.isfinite(t)
    v = t[good].double()
    return {
        "dtype": str(t.dtype), "shape": list(t.shape),
        "nonfinite": int((~good).sum()),
        "absmax": float(v.abs().max()) if v.numel() else None,
        "rms": float(v.square().mean().sqrt()) if v.numel() else None,
    }


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def replay(checkpoint, snapshot, mode, device, rank):
    from models.sit import SiT_models

    args = checkpoint["args"]
    get = lambda name, default=None: args.get(name, default) if isinstance(args, dict) else getattr(args, name, default)
    state = checkpoint["model"]
    z_dims = [state[f"projectors.{i}.4.weight"].shape[0]
              for i in range(100) if f"projectors.{i}.4.weight" in state]
    model = SiT_models[get("model")](
        input_size=tuple(snapshot["z"].shape[-2:]),
        in_channels=snapshot["z"].shape[1], z_dims=z_dims,
        encoder_depth=get("encoder_depth", 8),
        num_classes=get("num_classes", 1000),
        class_dropout_prob=get("cfg_prob", .1),
        bn_momentum=get("bn_momentum", .1),
        fused_attn=snapshot.get("fused_attn", get("fused_attn", True)),
        qk_norm=snapshot.get("qk_norm", get("qk_norm", False)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    model.bn.eval()
    report = {"rank": rank, "mode": mode, "trace": {}, "backward_nonfinite_order": []}
    handles = []
    linear_inputs = {}
    suspect = {x["name"].removesuffix(".weight") for x in snapshot.get("bad_grad_stats", [])
               if x["name"].endswith(".weight")}

    def hook(name):
        def record(module, inputs, output):
            if not torch.is_tensor(output):
                return
            row = report["trace"].setdefault(name, {})
            row["output"] = stats(output)
            if inputs and torch.is_tensor(inputs[0]):
                row["input"] = stats(inputs[0])
                if name in suspect and isinstance(module, torch.nn.Linear):
                    linear_inputs[name] = inputs[0].detach()
            if output.requires_grad:
                def grad_record(gradient):
                    row["grad_output"] = stats(gradient)
                    if row["grad_output"]["nonfinite"]:
                        report["backward_nonfinite_order"].append(name)
                    if name in linear_inputs:
                        # Recompute only dW from the exact activation and upstream gradient.
                        # This distinguishes a Linear weight-gradient GEMM failure from
                        # already non-finite operands arriving from another operation.
                        x = linear_inputs[name].reshape(-1, linear_inputs[name].shape[-1])
                        dy = gradient.detach().reshape(-1, gradient.shape[-1])
                        with torch.autocast("cuda", enabled=False):
                            ref = dy.double().T @ x.double()
                        row["weight_grad_reference_fp64"] = stats(ref)
                        row["weight_grad_operand_x"] = stats(x)
                        row["weight_grad_operand_dy"] = stats(dy)
                output.register_hook(grad_record)
        return record

    for name, module in model.named_modules():
        if (isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm))
                or name in {"x_embedder", "t_embedder", "parent_embedder"}
                or (name.startswith("blocks.") and name.count(".") == 1)):
            handles.append(module.register_forward_hook(hook(name)))

    saved_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False if mode == "bf16-safe-reduction" else saved_reduction)
    torch.backends.cuda.matmul.allow_tf32 = False if mode == "fp32" else get("allow_tf32", False)
    torch.backends.cudnn.allow_tf32 = False if mode == "fp32" else get("allow_tf32", False)
    # Materialize tensors before restoring RNG. Restore for EVERY mode, even if
    # explicit t/noise are available: classifier-free dropout also consumes RNG.
    z = move(snapshot["z"], device)
    parents = move(snapshot["parents"], device)
    zs = move(snapshot.get("zs"), device)
    time_input = move(snapshot.get("time_input"), device)
    noises = move(snapshot.get("noises"), device)
    rng = snapshot.get("forward_rng_state")
    if not rng or rng.get("cpu") is None or rng.get("cuda") is None:
        raise RuntimeError("Missing saved RNG state; exact replay is unavailable")
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"], device)
    precision = snapshot.get("mixed_precision", get("mixed_precision", "bf16"))
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    enabled = mode != "fp32" and precision != "no"
    attention_context = (torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                         enable_mem_efficient=False) if mode == "bf16-math-attention"
                         else contextlib.nullcontext())
    try:
        with attention_context, torch.autocast("cuda", dtype=dtype, enabled=enabled):
            out = model(x=z, parents=parents, zs=zs, time_input=time_input, noises=noises,
                        loss_kwargs={"weighting": get("weighting", "uniform"),
                                     "path_type": get("path_type", "linear"),
                                     "prediction": get("prediction", "v"), "align_only": False})
            loss = out["denoising_loss"].mean()
            if get("proj_coeff", 0) > 0:
                loss = loss + get("proj_coeff") * out["proj_loss"].mean()
        loss_value = float(loss.detach())
        report["loss"] = loss_value if math.isfinite(loss_value) else str(loss_value)
        report["saved_loss"] = snapshot.get("losses", {}).get("sit_loss")
        report["time"] = out["time_input"].detach().flatten().cpu().tolist()
        report["noise"] = stats(out["noises"])
        loss.backward()
        report["bad_parameters"] = {name: stats(p.grad) for name, p in model.named_parameters()
                                    if p.grad is not None and not torch.isfinite(p.grad).all()}
        for name in suspect:
            module = model.get_submodule(name)
            if module.weight.grad is not None:
                report["trace"][name]["actual_weight_grad"] = stats(module.weight.grad)
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        for handle in handles:
            handle.remove()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = saved_reduction
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        torch.backends.cudnn.allow_tf32 = saved_cudnn_tf32
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("failure_precision_report.json"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--modes", nargs="+", default=["baseline", "fp32", "bf16-safe-reduction", "bf16-math-attention"],
                        choices=["baseline", "fp32", "bf16-safe-reduction", "bf16-math-attention"])
    opts = parser.parse_args()
    sys.path.insert(0, str(opts.repo_root.resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic on an allocated GPU, not the login server")
    torch.set_num_threads(2)
    checkpoint = torch.load(opts.failure_dir / "model_before_failure.pt", map_location="cpu", weights_only=False)
    reports = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "failure_dir": str(opts.failure_dir), "results": []}
    for rank in opts.ranks:
        snapshot = torch.load(opts.failure_dir / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        for mode in opts.modes:
            result = replay(checkpoint, snapshot, mode, torch.device("cuda:0"), rank)
            reports["results"].append(result)
            opts.output.write_text(json.dumps(reports, indent=2, allow_nan=False))
            print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
