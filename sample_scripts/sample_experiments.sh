#!/bin/bash
set -euo pipefail

# Usage: bash sample_scripts/sample_experiments.sh [gpus48|gpus24] [STEP|latest-common] [--dry-run]
partition="${1:-gpus48}"
checkpoint_step="${2:-latest-common}"

project_root="/vol/biomedic3/tx1215/REPA-E"
uv_bin="/vol/biomedic2/tx1215/uv/bin/uv"
checkpoint_dir="${project_root}/checkpoints"
output_dir="${project_root}/sample_results"

# Experiments.
ldm_exp="embed_latent_dit_sits2_256x192"
repa_exp="embed_repa_sits2_256x192"
repae_exp="embed_repae_sits2_256x192"

# Sampling settings shared by all three baselines.
weights="ema"
split="test"
num_samples=5000
num_steps=50
batch_size=4
num_workers=0
seed=1234
cfg_scale=1.0

dry_run=false
if [ "${3:-}" = "--dry-run" ]; then
    dry_run=true
elif [ $# -gt 2 ]; then
    echo "Usage: bash sample_scripts/sample_experiments.sh PARTITION STEP [--dry-run]" >&2
    exit 1
fi

if [ "$partition" != "gpus48" ] && [ "$partition" != "gpus24" ]; then
    echo "Unknown partition: $partition. Use gpus48 or gpus24." >&2
    exit 1
fi
if [ ! -x "$uv_bin" ]; then
    echo "uv executable not found: $uv_bin" >&2
    exit 1
fi
if [ "$ldm_exp" = "$repa_exp" ] || [ "$ldm_exp" = "$repae_exp" ] || [ "$repa_exp" = "$repae_exp" ]; then
    echo "Set three different experiment names." >&2
    exit 1
fi

# Select a checkpoint step present in every experiment, if requested.
if [ "$checkpoint_step" = "latest-common" ]; then
    checkpoint_step=0
    for checkpoint in "${checkpoint_dir}/${ldm_exp}/checkpoints/"*.pt; do
        checkpoint_name="$(basename "$checkpoint")"
        if [[ ! "$checkpoint_name" =~ ^[0-9]+\.pt$ ]]; then
            continue
        fi
        if [ ! -f "${checkpoint_dir}/${repa_exp}/checkpoints/${checkpoint_name}" ] || \
           [ ! -f "${checkpoint_dir}/${repae_exp}/checkpoints/${checkpoint_name}" ]; then
            continue
        fi
        candidate_step=$((10#${checkpoint_name%.pt}))
        if [ "$candidate_step" -gt "$checkpoint_step" ]; then
            checkpoint_step=$candidate_step
        fi
    done
fi

if [[ ! "$checkpoint_step" =~ ^[0-9]+$ ]] || [ "$checkpoint_step" -eq 0 ]; then
    echo "Choose a positive checkpoint step present in all three experiments." >&2
    exit 1
fi
printf -v checkpoint_name '%07d' "$((10#$checkpoint_step))"
ldm_ckpt="${checkpoint_dir}/${ldm_exp}/checkpoints/${checkpoint_name}.pt"
repa_ckpt="${checkpoint_dir}/${repa_exp}/checkpoints/${checkpoint_name}.pt"
repae_ckpt="${checkpoint_dir}/${repae_exp}/checkpoints/${checkpoint_name}.pt"

sample_name="${split}_n${num_samples}_seed${seed}_steps${num_steps}_${weights}"
ldm_samples="${output_dir}/${ldm_exp}/step_${checkpoint_name}/${sample_name}"
repa_samples="${output_dir}/${repa_exp}/step_${checkpoint_name}/${sample_name}"
repae_samples="${output_dir}/${repae_exp}/step_${checkpoint_name}/${sample_name}"

for checkpoint in "$ldm_ckpt" "$repa_ckpt" "$repae_ckpt"; do
    if [ ! -f "$checkpoint" ]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 1
    fi
done
for samples in "$ldm_samples" "$repa_samples" "$repae_samples"; do
    if [ -e "$samples" ]; then
        echo "Sampling folder already exists: $samples" >&2
        echo "Evaluate saved images with eval_scripts/compare_experiments.sh." >&2
        exit 1
    fi
done

ARGS=(
    --weights="$weights"
    --split="$split"
    --num-samples=$num_samples
    --num-steps=$num_steps
    --batch-size=$batch_size
    --num-workers=$num_workers
    --seed=$seed
    --cfg-scale=$cfg_scale
    --save-visuals

    --tasks generation reconstruction counterfactual
    --attributes view density scanner age
)

echo "Sampling comparison"
echo "  LDM          = $ldm_exp"
echo "  REPA         = $repa_exp"
echo "  REPA-E       = $repae_exp"
echo "  checkpoint   = $checkpoint_name"
echo "  partition    = $partition"
echo "  LDM images   = $ldm_samples"
echo "  REPA images  = $repa_samples"
echo "  REPA-E images= $repae_samples"

if [ "$dry_run" = true ]; then
    echo "Dry run: submit three independent one-GPU sampling jobs. No evaluation jobs."
    exit 0
fi

# These calls submit immediately; the three sampling jobs are independent.
ldm_job=$(bash "${project_root}/sample_scripts/sample_checkpoint.sh" \
    "$partition" "$ldm_ckpt" "${ARGS[@]}" --output-dir="$ldm_samples")
printf 'method\tjob_id\timages\nldm\t%s\t%s\n' "$ldm_job" "$ldm_samples" > "${ldm_samples}/logs/jobs.tsv"

repa_job=$(bash "${project_root}/sample_scripts/sample_checkpoint.sh" \
    "$partition" "$repa_ckpt" "${ARGS[@]}" --output-dir="$repa_samples")
printf 'method\tjob_id\timages\nrepa\t%s\t%s\n' "$repa_job" "$repa_samples" > "${repa_samples}/logs/jobs.tsv"

repae_job=$(bash "${project_root}/sample_scripts/sample_checkpoint.sh" \
    "$partition" "$repae_ckpt" "${ARGS[@]}" --output-dir="$repae_samples")
printf 'method\tjob_id\timages\nrepae\t%s\t%s\n' "$repae_job" "$repae_samples" > "${repae_samples}/logs/jobs.tsv"

echo "Submitted sampling jobs"
echo "  LDM          = $ldm_job"
echo "  REPA         = $repa_job"
echo "  REPA-E       = $repae_job"
echo "  job records are saved in each experiment's logs/jobs.tsv"
