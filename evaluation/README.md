# Evaluate saved images

Project layout:

```text
sample/           sampling code and image-cache format
sample_scripts/   sampling Slurm launcher
sample_results/   saved images and CSV manifests

evaluation/       metric code and comparison reports
eval_scripts/     evaluation and comparison Slurm launchers
```

The workflow follows `diffusion_counterfactual`: sample PNGs and metadata first,
then score saved images. `evaluate_all.py` does not load the generative checkpoint,
VAE, or raw dataset. Recomputing metrics never generates images again.

## Three-baseline workflow

Sampling has its own launcher in `sample_scripts/`. Edit the three experiment
names and shared settings at the top of that script, then run from
`/vol/biomedic3/tx1215/REPA-E`:

```bash
bash sample_scripts/sample_experiments.sh gpus48 65000
```

This submits three independent one-GPU sampling jobs. Images and metadata are
saved in `sample_results/`; no metric or report jobs are submitted. Omit the step
or use `latest-common` to sample the latest step shared by all three experiments.
Each experiment's logs and job ID are saved in its own sampling folder, under
`logs/` (`sample_checkpoint.JOB_ID.log`, `sample_checkpoint.JOB_ID.out`, and
`jobs.tsv`).

After sampling completes, evaluate the saved images:

```bash
bash eval_scripts/compare_experiments.sh gpus48 65000
```

The evaluation launcher submits three independent metric jobs and one dependent
report job. It never samples images or opens generative checkpoints. Set matching
experiment names and saved-image folder settings at its top. Metric settings
live separately in `EVAL_ARGS`. An explicit sampled step is required.

Both launchers support `gpus24` and an optional final `--dry-run` argument.
Submissions use the training-script pattern: partition, one GPU, log path, and
required dependencies. CPU cores and tasks are assigned automatically by the
cluster. Evaluate only completed caches containing a `COMPLETE` marker.

## One saved-image run

```bash
bash eval_scripts/evaluate_all.sh gpus48 \
  sample_results/EXPERIMENT/step_0065000/test_n5000_seed1234_steps50_ema
```

Direct Python, inside an allocation:

```bash
.venv/bin/python evaluation/evaluate_all.py \
  --samples-dir sample_results/EXPERIMENT/step_0065000/test_n5000_seed1234_steps50_ema \
  --evaluations identity --identity-metrics mae mse ssim \
  --output-dir evaluation/results/my_identity_evaluation
```

Choose a new metric output directory for each reevaluation. Evaluation verifies
metadata, image checksums and fixed image scaling; missing/corrupt files or
incomplete caches fail clearly. Saved images are never modified. Metric feature
models (VGG/LPIPS, Inception, DINO) load only when their metrics are requested.

See [sampling instructions](../sample/README.md) for image layout and precision.

## Counterfactual effectiveness

Effectiveness asks whether an independent image predictor recovers the attribute
value explicitly requested by an intervention. It complements the existing
similarity and distribution metrics. Sampling does not need to be repeated.

### 1. Train and assess an image predictor

Training is a separate, user-launched step. The new scripts do not start training
when imported or when running evaluation. In your chosen GPU session, from the
project root:

```bash
# Inspect the configuration without starting training or submitting a job.
bash train_scripts/train_attribute_predictor.sh local --dry-run

# Start training when ready.
bash train_scripts/train_attribute_predictor.sh local
```

Use `gpus48` or `gpus24` instead of `local` when submitting from a host with
Slurm installed. Trainer options can be appended, for example
`--epochs=30 --output-dir=checkpoints/attribute_predictor/my_run`.
Each run requires a new output directory; logs are kept inside that directory.
The default run is
`checkpoints/attribute_predictor/resnet50_256x192_seed1234/`.

The predictor uses an ImageNet-initialized ResNet-50, separate heads for view
(2 classes), density (4), scanner (5), and continuous age. It consumes only
images; it never receives conditioning attributes as inputs. Real-image reads
use the same breast masking, fixed image scale and 256×192 resizing as sampling,
followed by grayscale-to-RGB repetition and ImageNet normalization.
ImageNet initialization may download weights during training; scoring is offline.

The trainer checks patient IDs and image paths across the complete train,
validation and test CSVs before training. Class weights and age normalization
come only from training data. Validation performance selects the checkpoint and
controls early stopping; the test split is evaluated once after selection.
The selection criterion averages three categorical balanced-error rates and age
MAE divided by the training age standard deviation. This criterion selects the
predictor; it is not a combined counterfactual metric.

Inspect `run.json` and `history.json` before using `best.pt` as a measurement
model. Reports include confusion matrices, per-class support, accuracy,
balanced accuracy, macro F1, age MAE/RMSE, and training-majority/mean-age
baselines. Completing training does not certify that each attribute is predicted
well: assess all heads against these baselines and the intended use. Age may be
harder to recover from images. Domain shift on generated images can affect even
a strong real-image predictor.

Only completed checkpoints with uncapped validation/test reports are accepted
by the evaluator. `--max-train-samples`, `--max-valid-samples`, and
`--max-test-samples` are debug options; any sample cap marks a checkpoint as
ineligible for effectiveness. Evaluation uses the matching held-out test CSV
hash and trained resolution, rejecting train/validation caches or mismatched
splits. Predictor weights remain frozen during evaluation.

### 2. Score saved counterfactuals

After training and reviewing the held-out results, evaluate saved test images:

```bash
.venv/bin/python evaluation/evaluate_all.py \
  --samples-dir sample_results/EXPERIMENT/step_0065000/test_n1000_seed1234_steps50_ema \
  --evaluations identity inception dino effectiveness \
  --attribute-predictor checkpoints/attribute_predictor/resnet50_256x192_seed1234/best.pt \
  --age-tolerance-years 5 \
  --output-dir evaluation/results/EXPERIMENT_with_effectiveness
```

Categorical **target accuracy** is the fraction of eligible counterfactuals
whose predicted edited attribute equals the requested target. **Balanced target
accuracy** averages this rate across represented target categories; support is
recorded explicitly. Age reports **target MAE** and **target RMSE in years**,
plus the fraction within `--age-tolerance-years` (default 5 years, inclusive).
Categorical accuracy is higher-is-better; age errors are lower-is-better. These
scores are not combined into a single number.

Every eligible edit counts, including inputs the predictor initially labels
incorrectly. `input_target` scores the unchanged input against the same requested
target, providing a no-edit diagnostic. `real_prediction` checks predictions
against factual labels on the cached cohort. These factual diagnostics accompany
the checkpoint's held-out quality report. Attributes without interventions are
N/A rather than zero. Other attributes are not scored as preservation targets:
an intervention can legitimately affect causally related attributes.

The run writes `effectiveness/per_sample.csv`, `effectiveness/summary.csv`, and
`effectiveness/summary.json`, and includes the new metric rows in the standard
`summary.csv`. Checkpoint and tensor fingerprints, preprocessing, split hashes,
and held-out performance are retained in `config.json`.

The Slurm launchers can enable effectiveness using the same checkpoint for all
methods:

```bash
ATTRIBUTE_PREDICTOR=checkpoints/attribute_predictor/resnet50_256x192_seed1234/best.pt \
  bash eval_scripts/compare_experiments.sh gpus48 65000
```

`AGE_TOLERANCE_YEARS` optionally overrides the default. The comparison report
adds effectiveness tables, categorical accuracy and age-error panels, factual
diagnostics, and `effectiveness_targets.csv`. Comparisons reject differing
predictor fingerprints or age tolerances. Without a predictor argument, the
existing evaluation workflow stays available.

## Results and reporting

Existing metric definitions are retained: VAE and no-edit-cycle baselines,
CF-versus-input and CF-versus-cycle similarity, and generation/reconstruction/
observational target-group Inception FID and DINO distances.

Individual results remain under `evaluation/results/` for standalone runs.
Comparison launches use `evaluation/comparisons/step_TIMESTAMP_PID/`, containing
individual metric results, logs, `jobs.tsv`, `report.md`, combined CSVs, and
PNG/PDF figures. `COMPARISON_DIR` selects an exact new comparison destination.

Each metric run stores its Slurm and Python logs in its own `logs/` folder. For
a comparison these are `evaluations/ldm/logs/`, `evaluations/repa/logs/`, and
`evaluations/repae/logs/`. The shared report job's log stays in the comparison's
`logs/` folder.

Reports verify shared split indices, identical saved real images, image precision,
sampling settings, metric settings, and feature-model fingerprints. To summarize
already-computed metrics without rerunning either sampling or scoring:

```bash
.venv/bin/python evaluation/summarize_results.py \
  --ldm-results /path/to/ldm_metrics \
  --repa-results /path/to/repa_metrics \
  --repae-results /path/to/repae_metrics \
  --output-dir /path/to/report
```

Similarity alone does not establish successful edits or preserved patient
identity. Counterfactual reference groups are observational. Weighted distances
are means of group distances, not pooled FIDs. Unsupported groups remain N/A.
