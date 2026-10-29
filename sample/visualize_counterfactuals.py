#!/usr/bin/env python3
"""Save original / counterfactual / difference panels from exported images.

    python sample/visualize_counterfactuals.py --samples-dir sample_results/RUN/step_0065000/CACHE
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sample.sample_io import load_samples, read_image


def counterfactual_figure(original, counterfactual, attribute, source, target):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    original = original.detach().float().cpu()
    counterfactual = counterfactual.detach().float().cpu()
    if original.shape != counterfactual.shape or original.ndim != 3 or original.shape[0] != 1:
        raise ValueError('Counterfactual panels need matching grayscale 1xHxW images.')
    figure = Figure(figsize=(9, 3.8), dpi=120)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis, image in zip(axes[:2], (original, counterfactual)):
        axis.imshow(image[0].numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=11)
    axes[1].set_title(f'Counterfactual\n{attribute}: {float(source):g} → {float(target):g}', fontsize=11)
    axes[2].set_title('Difference\n(Counterfactual − Original) × 255', fontsize=10)

    # Match plot_utils._show_difference: signed 8-bit display units, RdBu_r,
    # symmetric limits around zero and a horizontal colorbar below the map.
    difference = (counterfactual - original)[0] * 255.0
    limit = float(difference.abs().max()) or 1.0
    effect = axes[2].imshow(difference.numpy(), cmap='RdBu_r', vmin=-limit, vmax=limit)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    divider = make_axes_locatable(axes[2])
    color_axis = divider.append_axes('bottom', size='4%', pad=0.06)
    colorbar = figure.colorbar(effect, cax=color_axis, orientation='horizontal')
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7, pad=1)
    colorbar.set_ticks(np.round(np.linspace(np.ceil(-limit), np.floor(limit), 5)))
    figure.subplots_adjust(left=.015, right=.985, bottom=.12, top=.83, wspace=.08)
    return figure


def save_counterfactual_visuals(samples_dir, records=None):
    """Create a panel for every eligible CF; quantitative PNGs/manifest stay intact."""
    from tqdm import tqdm

    samples_dir = Path(samples_dir).expanduser().resolve()
    if records is None:
        _, _, records = load_samples(samples_dir)
    references = {int(row['index']): row for row in records if row['task'] == 'real'}
    counterfactuals = [row for row in records if row['task'] == 'counterfactual']
    for row in tqdm(counterfactuals, desc='Saving counterfactual panels'):
        index = int(row['index'])
        reference = references[index]
        original = read_image(samples_dir, reference['file_path'], reference['sha256'])
        counterfactual = read_image(samples_dir, row['file_path'], row['sha256'])
        figure = counterfactual_figure(original, counterfactual, row['attribute'], row['source'], row['target'])
        output = samples_dir / 'counterfactual_visuals' / row['attribute'] / f'{index:08d}.png'
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output, dpi=120)
        finally:
            figure.clear()
    return len(counterfactuals)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples-dir', required=True, help='Completed sample cache; no checkpoint is loaded.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    count = save_counterfactual_visuals(args.samples_dir)
    print(f'Saved {count} panels: {Path(args.samples_dir).expanduser().resolve() / "counterfactual_visuals"}')
