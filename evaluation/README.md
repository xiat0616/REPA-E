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

New sampling runs draw categorical targets uniformly from all classes, including
the source class; age targets are uniform within the training split's age range.
Default cache names end in `_ema` (or `_model`), and the target policy is recorded
in `config.json`. To evaluate an earlier suffixed cache, set
`SAMPLE_SUFFIX="_uniform"` when calling `compare_experiments.sh`, or pass its path
to the single-cache evaluator. New and old policies should be reported separately.

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
bash eval_scripts/evaluate_checkpoint.sh gpus48 \
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

Training logs include immediate process startup, split-check stages, model
initialization, epoch/batch progress, validation results, and best-checkpoint
updates. Progress is flushed every 30 seconds by default; adjust it with
`--log-interval-seconds`. Image-path checks can take time on the shared filesystem
and report their own progress before `run.json` is created. Slurm logs begin
when the job starts running. Code changes apply to future launches; a running
Python process keeps its already loaded code.

Logs use readable timestamped messages. During training, Slurm logs show lines
such as:

```text
[22:45:00 UTC] Epoch 1/20 | Batch 120/3715 (3.2%) | loss=1.2345 | mean_loss=1.4567 | elapsed 00:02:00 | epoch ETA 00:59:54 | 64.0 img/s
```

`loss` is the current batch loss; `mean_loss` is the running mean for that epoch.
ETA estimates the remaining training time in the current epoch, excluding
validation. Lines are printed after the first and last batches, and every
`--log-interval-seconds` in between (default 30 seconds). Interactive terminals
show a tqdm bar instead, refreshed about once per second. `--no-progress-bar`
uses the same readable batch lines in terminals. Dataset checks, validation,
and checkpoint updates also use short readable messages. Full configuration,
confusion matrices, baselines and history remain in `run.json`/`history.json`.

The predictor uses an ImageNet-initialized ResNet-50, separate heads for view
(2 classes), density (4), scanner (5), and continuous age. It consumes only
images; it never receives conditioning attributes as inputs. Real-image reads
use the same breast masking, fixed image scale and 256×192 resizing as sampling,
followed by grayscale-to-RGB repetition and ImageNet normalization.
As in the existing EMBED loader, resized values within `1e-6` of `[0,1]` are
clamped to that interval to absorb floating-point roundoff. Nonfinite values,
incorrect shapes and larger range errors fail with the row, image path and
diagnostic details; images are never silently replaced.
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
  --output-dir evaluation/results/EXPERIMENT_with_effectiveness
```

Categorical **target accuracy** is the fraction of eligible counterfactuals
whose predicted edited attribute equals the requested target. **Balanced target
accuracy** averages this rate across represented target categories; support is
recorded explicitly. Age reports **target MAE in years**:
`mean(abs(predicted_age_years - requested_target_age_years))`. The predictor
already returns years, so no extra normalization or scaling is applied.
Categorical accuracy is higher-is-better; age MAE is lower-is-better. These
scores are not combined into a single number.

Every eligible edit counts, including inputs the predictor initially labels
incorrectly. `input_target` scores the unchanged input against the same requested
target, providing a no-edit diagnostic. `real_prediction` checks predictions
against factual labels on the cached cohort. These factual diagnostics accompany
the checkpoint's held-out quality report. Attributes without interventions are
N/A rather than zero. Other attributes are not scored as preservation targets:
an intervention can legitimately affect causally related attributes.

Uniform-target caches retain same-target draws as no-edit controls, and these
controls count in the overall target scores. They can increase target accuracy
without requiring an attribute change; compare against `input_target` and inspect
source/target values in `effectiveness/per_sample.csv`. The comparison report
also verifies that methods used identical actual target assignments and age bounds.

The run writes `effectiveness/per_sample.csv`, `effectiveness/summary.csv`, and
`effectiveness/summary.json`, and includes the new metric rows in the standard
`summary.csv`. Checkpoint and tensor fingerprints, preprocessing, split hashes,
and held-out performance are retained in `config.json`.

The Slurm launchers enable effectiveness by default using
`checkpoints/attribute_predictor/resnet50_256x192_seed1234/best.pt` for all
methods. Set `ATTRIBUTE_PREDICTOR` to use another checkpoint:

```bash
ATTRIBUTE_PREDICTOR=checkpoints/attribute_predictor/resnet50_256x192_seed1234/best.pt \
  bash eval_scripts/compare_experiments.sh gpus48 65000
```

Set `ATTRIBUTE_PREDICTOR=""` to disable effectiveness, including when requesting
a custom `--evaluations` subset that excludes it. The comparison report
adds effectiveness tables, categorical accuracy and age MAE panels, factual
diagnostics, and `effectiveness_targets.csv`. Comparisons reject differing
predictor fingerprints or age metric definitions. With an explicitly empty predictor,
the existing identity, Inception and DINO evaluation workflow stays available.

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
