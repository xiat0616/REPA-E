import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Reshape time t to broadcast over x."""
    dims = [1] * (len(x_cur.size()) - 1)
    return t.view(t.size(0), *dims)


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Transform velocity prediction into score."""
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, -torch.ones_like(xt)
        sigma_t, d_sigma_t = t, torch.ones_like(xt)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    return (reverse_alpha_ratio * vt - xt) / var


def compute_diffusion(t_cur):
    return 2 * t_cur


def _prepare_cfg_inputs(x, parents):
    """Duplicate inputs for conditional/unconditional classifier-free guidance."""
    batch = x.size(0)
    model_input = torch.cat([x, x], dim=0)
    parents_input = {k: torch.cat([v, v], dim=0) for k, v in parents.items()}
    force_drop_ids = torch.cat([
        torch.zeros(batch, device=x.device, dtype=torch.bool),
        torch.ones(batch, device=x.device, dtype=torch.bool),
    ])
    return model_input, parents_input, force_drop_ids


def _apply_cfg(pred, cfg_scale):
    pred_cond, pred_uncond = pred.chunk(2)
    return pred_uncond + cfg_scale * (pred_cond - pred_uncond)


def euler_sampler(
    model,
    latents,
    parents,
    num_steps=20,
    heun=False,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    path_type="linear",  # kept for compatibility
):
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            use_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low

            if use_cfg:
                model_input, parents_input, force_drop_ids = _prepare_cfg_inputs(x_cur, parents)
            else:
                model_input, parents_input, force_drop_ids = x_cur, parents, None

            time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
            d_cur = model.inference(
                model_input.to(dtype=_dtype),
                time_input.to(dtype=_dtype),
                parents_input,
                force_drop_ids=force_drop_ids,
            ).to(torch.float64)

            if use_cfg:
                d_cur = _apply_cfg(d_cur, cfg_scale)

            x_next = x_cur + (t_next - t_cur) * d_cur

            if heun and i < num_steps - 1:
                if use_cfg:
                    model_input, parents_input, force_drop_ids = _prepare_cfg_inputs(x_next, parents)
                else:
                    model_input, parents_input, force_drop_ids = x_next, parents, None

                time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_next
                d_prime = model.inference(
                    model_input.to(dtype=_dtype),
                    time_input.to(dtype=_dtype),
                    parents_input,
                    force_drop_ids=force_drop_ids,
                ).to(torch.float64)

                if use_cfg:
                    d_prime = _apply_cfg(d_prime, cfg_scale)

                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
    model,
    latents,
    parents,
    num_steps=20,
    heun=False,  # kept for compatibility
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    path_type="linear",
):
    _dtype = latents.dtype
    t_steps = torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.0], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
            dt = t_next - t_cur
            x_cur = x_next
            use_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low

            if use_cfg:
                model_input, parents_input, force_drop_ids = _prepare_cfg_inputs(x_cur, parents)
            else:
                model_input, parents_input, force_drop_ids = x_cur, parents, None

            time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            v_cur = model.inference(
                model_input.to(dtype=_dtype),
                time_input.to(dtype=_dtype),
                parents_input,
                force_drop_ids=force_drop_ids,
            ).to(torch.float64)

            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur

            if use_cfg:
                d_cur = _apply_cfg(d_cur, cfg_scale)

            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

        # Final deterministic step to t=0
        t_cur, t_next = t_steps[-2], t_steps[-1]
        dt = t_next - t_cur
        x_cur = x_next
        use_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low

        if use_cfg:
            model_input, parents_input, force_drop_ids = _prepare_cfg_inputs(x_cur, parents)
        else:
            model_input, parents_input, force_drop_ids = x_cur, parents, None

        time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur

        v_cur = model.inference(
            model_input.to(dtype=_dtype),
            time_input.to(dtype=_dtype),
            parents_input,
            force_drop_ids=force_drop_ids,
        ).to(torch.float64)

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        diffusion = compute_diffusion(t_cur)
        d_cur = v_cur - 0.5 * diffusion * s_cur

        if use_cfg:
            d_cur = _apply_cfg(d_cur, cfg_scale)

        return x_cur + dt * d_cur