# Sampling

Sampling code lives here; Slurm launchers live in `sample_scripts/`, and saved
images go to `sample_results/`. Metric code remains in `evaluation/`.

From `/vol/biomedic3/tx1215/REPA-E`:

```bash
bash sample_scripts/sample_all.sh gpus48 \
  checkpoints/EXPERIMENT/checkpoints/0065000.pt
```

Or run Python inside an allocation:

```bash
.venv/bin/python sample/sample_all.py \
  --checkpoint checkpoints/EXPERIMENT/checkpoints/0065000.pt
```

The sampler loads the checkpoint and raw dataset once. It saves real inputs,
fresh generation, VAE reconstructions, no-edit flow reconstructions, and each
requested counterfactual. Sampling does not load metric feature models.

```text
sample_results/EXPERIMENT/step_0065000/test_n5000_seed1234_steps50_ema/
  config.json
  indices.json
  samples.csv
  COMPLETE
  real/00000123.png
  generation/00000123.png
  reconstruction/vae/00000123.png
  reconstruction/cycle/00000123.png
  counterfactual/view/00000123.png
  counterfactual/density/00000123.png
  counterfactual/scanner/00000123.png
  counterfactual/age/00000123.png
```

`sample_io.py` defines the portable image/manifest format shared with evaluation.
PNG paths are relative to the cache folder. Filenames use stable split-CSV row
indices. The CSV links outputs to factual inputs and records task, conditions,
intervention source/target, and image checksums. No-op interventions have no CF
row. `COMPLETE` is written only after the export finishes successfully.

Images are 16-bit grayscale PNGs, encoded as
`round(clamp(image,0,1)*65535)` and read with the fixed divisor 65535. References
and outputs use the same encoding; no per-image normalization is applied.
Maximum encoding error is about 0.00000763. This introduces a small quantization
difference from the previous in-memory FP32 metrics, so compare methods using
the same saved-image protocol.

Existing export directories are preserved. To rerun an incomplete export, use a
new `--output-dir`. Completed folders can be moved without their checkpoint or
raw dataset because all images and metadata needed for evaluation are included.

For three baselines, edit settings in `sample_scripts/sample_experiments.sh`:

```bash
bash sample_scripts/sample_experiments.sh gpus48 65000
```

See [evaluation instructions](../evaluation/README.md) to score saved images.
