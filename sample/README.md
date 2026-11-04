# Sampling

Sampling code lives here; Slurm launchers live in `sample_scripts/`, and saved
images go to `sample_results/`. Metric code remains in `evaluation/`.

From `/vol/biomedic3/tx1215/REPA-E`:

```bash
bash sample_scripts/sample_checkpoint.sh gpus48 \
  checkpoints/EXPERIMENT/checkpoints/0065000.pt
```

Or run Python inside an allocation:

```bash
.venv/bin/python sample/sample_checkpoint.py \
  --checkpoint checkpoints/EXPERIMENT/checkpoints/0065000.pt
```

The sampler loads the checkpoint and raw dataset once. It saves real inputs,
fresh generation, VAE reconstructions, no-edit flow reconstructions, and each
requested counterfactual. Sampling does not load metric feature models.

Default counterfactual targets are uniform random draws, independent of the
source value: view draws from 0–1, density from 0–3, and scanner from 0–4.
The source class is included with the same probability as every other class.
Age draws continuously and uniformly between the minimum and maximum `age`
in the training split's `train.csv`, rather than adding ten years. All other
conditions remain factual. Explicit `--target-view`, `--target-density`,
`--target-scanner`, and `--target-age` values still override the random draws.

Targets are keyed by seed, split-CSV row index and attribute, so the same input
receives the same target across LDM, REPA, REPA-E and checkpoint steps, regardless
of batch size, global random state or requested attribute order. This is random
uniform sampling; finite cohorts need not contain exactly equal target counts.
The training age bounds, training-CSV checksum, policy and actual target-assignment
checksum are recorded in `config.json`.

```text
sample_results/EXPERIMENT/step_0065000/test_n5000_seed1234_steps50_ema_uniform/
  config.json
  indices.json
  samples.csv
  COMPLETE
  logs/sample_checkpoint.JOB_ID.log
  logs/sample_checkpoint.JOB_ID.out
  logs/jobs.tsv
  real/00000123.png
  generation/00000123.png
  reconstruction/vae/00000123.png
  reconstruction/cycle/00000123.png
  counterfactual/view/00000123.png
  counterfactual/density/00000123.png
  counterfactual/scanner/00000123.png
  counterfactual/age/00000123.png
  counterfactual_visuals/view/00000123.png
  counterfactual_visuals/density/00000123.png
  counterfactual_visuals/scanner/00000123.png
  counterfactual_visuals/age/00000123.png
```

Each eligible counterfactual also gets a horizontal visual panel: **original |
counterfactual | difference map**. The images are grayscale with a fixed [0,1]
display range. The difference map uses the existing `RdBu_r` colors and symmetric
limits around zero for `(counterfactual - original) * 255`: red is an increase,
blue is a decrease. A horizontal colorbar shows the scale for each panel. The
middle title includes the intervened attribute and its source/target values.

Panels are rendered from the saved 16-bit PNGs. They are RGB previews, stored
separately from `samples.csv`, and do not enter evaluation. They are enabled by
default; add `--no-save-visuals` to the single-checkpoint sampling command to skip
them. To add or regenerate panels in an existing completed cache without loading
a checkpoint or resampling:

```bash
.venv/bin/python sample/visualize_counterfactuals.py \
  --samples-dir sample_results/EXPERIMENT/step_0065000/test_n5000_seed1234_steps50_ema_uniform
```

`sample_io.py` defines the portable image/manifest format shared with evaluation.
PNG paths are relative to the cache folder. Filenames use stable split-CSV row
indices. The CSV links outputs to factual inputs and records task, conditions,
intervention source/target, and image checksums. Same-target draws are retained
as no-edit controls with a CF row; they are not redrawn or filtered out.
`COMPLETE` is written only after the export finishes successfully.

Images are 16-bit grayscale PNGs, encoded as
`round(clamp(image,0,1)*65535)` and read with the fixed divisor 65535. References
and outputs use the same encoding; no per-image normalization is applied.
Maximum encoding error is about 0.00000763. This introduces a small quantization
difference from the previous in-memory FP32 metrics, so compare methods using
the same saved-image protocol.

Existing export directories are preserved. To rerun an incomplete export, use a
new `--output-dir`. Completed folders can be moved without their checkpoint or
raw dataset because all images and metadata needed for evaluation are included.
New default folders end in `_uniform`, keeping them separate from previous cyclic
exports. Sampling and comparison launchers use this suffix by default. Set
`SAMPLE_SUFFIX=""` on the comparison launcher to evaluate older unsuffixed caches.

Slurm logs and the captured Python output are saved inside each sampling run's
own `logs/` folder. Three-baseline submissions also record that experiment's
job ID in `logs/jobs.tsv`. Python accepts the log-only folder created at
submission and refuses to overwrite existing sample data.

For three baselines, edit settings in `sample_scripts/sample_experiments.sh`:

```bash
bash sample_scripts/sample_experiments.sh gpus48 65000
```

See [evaluation instructions](../evaluation/README.md) to score saved images.
