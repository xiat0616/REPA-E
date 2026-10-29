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
Sampling job IDs are recorded under `sample_results/logs/`.

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

## Results and reporting

Existing metric definitions are retained: VAE and no-edit-cycle baselines,
CF-versus-input and CF-versus-cycle similarity, and generation/reconstruction/
observational target-group Inception FID and DINO distances.

Individual results remain under `evaluation/results/` for standalone runs.
Comparison launches use `evaluation/comparisons/step_TIMESTAMP_PID/`, containing
individual metric results, logs, `jobs.tsv`, `report.md`, combined CSVs, and
PNG/PDF figures. `COMPARISON_DIR` selects an exact new comparison destination.

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
