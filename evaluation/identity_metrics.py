"""Image-similarity metrics for saved grayscale images; no generative models."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


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
                    'Use --identity-metrics mae mse ssim to evaluate without LPIPS.')
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

