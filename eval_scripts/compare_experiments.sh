#!/bin/bash
set -euo pipefail

# Usage: bash eval_scripts/compare_experiments.sh [gpus48|gpus24] STEP [--dry-run]
partition="${1:-gpus48}"
checkpoint_step="${2:-}"

project_root="/vol/biomedic3/tx1215/REPA-E"
uv_bin="/vol/biomedic2/tx1215/uv/bin/uv"
sampling_root="${project_root}/sample_results"
output_dir="${project_root}/evaluation/comparisons"

# Experiments: these names must match the sampled runs.
ldm_exp="embed_latent_dit_sits2_256x192"
repa_exp="embed_repa_sits2_256x192"
repae_exp="embed_repae_sits2_256x192"

# These settings identify the saved-image folders; no sampling is performed.
weights="ema"
split="test"
num_samples=500
num_steps=50
seed=1234

# Metric settings.
batch_size=4
num_workers=0
feature_model="dinov2_vitb14"
save_examples=0

dry_run=false
if [ "${3:-}" = "--dry-run" ]; then
    dry_run=true
elif [ $# -gt 2 ]; then
    echo "Usage: bash eval_scripts/compare_experiments.sh PARTITION STEP [--dry-run]" >&2
    exit 1
fi

if [ "$partition" != "gpus48" ] && [ "$partition" != "gpus24" ]; then
    echo "Unknown partition: $partition. Use gpus48 or gpus24." >&2
    exit 1
fi
if [[ ! "$checkpoint_step" =~ ^[0-9]+$ ]] || [ "$checkpoint_step" -eq 0 ]; then
    echo "Specify the checkpoint step used to create the saved images." >&2
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
printf -v checkpoint_name '%07d' "$((10#$checkpoint_step))"

sample_name="${split}_n${num_samples}_seed${seed}_steps${num_steps}_${weights}"
ldm_samples="${sampling_root}/${ldm_exp}/step_${checkpoint_name}/${sample_name}"
repa_samples="${sampling_root}/${repa_exp}/step_${checkpoint_name}/${sample_name}"
repae_samples="${sampling_root}/${repae_exp}/step_${checkpoint_name}/${sample_name}"

for samples in "$ldm_samples" "$repa_samples" "$repae_samples"; do
    if [ ! -f "${samples}/COMPLETE" ]; then
        echo "No completed sampling run found: $samples" >&2
        exit 1
    fi
done

comparison_name="step_${checkpoint_name}_$(date +%Y%m%d_%H%M%S)_$$"
save_dir="${COMPARISON_DIR:-${output_dir}/${comparison_name}}"
if [ "${save_dir:0:1}" != "/" ]; then
    save_dir="${project_root}/${save_dir}"
fi
if [ -e "$save_dir" ]; then
    echo "Comparison directory already exists: $save_dir" >&2
    exit 1
fi
log_dir="${save_dir}/logs"
ldm_results="${save_dir}/evaluations/ldm"
repa_results="${save_dir}/evaluations/repa"
repae_results="${save_dir}/evaluations/repae"

EVAL_ARGS=(
    --batch-size=$batch_size
    --num-workers=$num_workers
    --save-examples=$save_examples
    --output-root="${save_dir}/evaluations"

    --evaluations identity inception dino
    --identity-metrics mae mse ssim lpips
    --feature-model="$feature_model"
)

SUMMARY_ARGS=(
    --ldm-results="$ldm_results"
    --repa-results="$repa_results"
    --repae-results="$repae_results"
    --output-dir="$save_dir"
)
printf -v REPORT_ARGS '%q ' "${SUMMARY_ARGS[@]}"
printf -v SUMMARY_LOG '%q' "${log_dir}/summary.%j.log"

echo "Saved-image evaluation comparison"
echo "  LDM images   = $ldm_samples"
echo "  REPA images  = $repa_samples"
echo "  REPA-E images= $repae_samples"
echo "  partition    = $partition"
echo "  report       = ${save_dir}/report.md"

if [ "$dry_run" = true ]; then
    echo "Dry run: submit three metric jobs and a dependent summary. No sampling jobs."
    exit 0
fi

mkdir -p "$log_dir"
printf 'stage\tmethod\tjob_id\toutput\n' > "${save_dir}/jobs.tsv"

# Evaluate the three saved-image folders independently.
ldm_eval_job=$(EVAL_DEPENDENCY="" \
    bash "${project_root}/eval_scripts/evaluate_all.sh" "$partition" "$ldm_samples" \
    "${EVAL_ARGS[@]}" --output-dir="$ldm_results")
printf 'evaluate\tldm\t%s\t%s\n' "$ldm_eval_job" "$ldm_results" >> "${save_dir}/jobs.tsv"

repa_eval_job=$(EVAL_DEPENDENCY="" \
    bash "${project_root}/eval_scripts/evaluate_all.sh" "$partition" "$repa_samples" \
    "${EVAL_ARGS[@]}" --output-dir="$repa_results")
printf 'evaluate\trepa\t%s\t%s\n' "$repa_eval_job" "$repa_results" >> "${save_dir}/jobs.tsv"

repae_eval_job=$(EVAL_DEPENDENCY="" \
    bash "${project_root}/eval_scripts/evaluate_all.sh" "$partition" "$repae_samples" \
    "${EVAL_ARGS[@]}" --output-dir="$repae_results")
printf 'evaluate\trepae\t%s\t%s\n' "$repae_eval_job" "$repae_results" >> "${save_dir}/jobs.tsv"

dependency="afterok:${ldm_eval_job}:${repa_eval_job}:${repae_eval_job}"

if [ "$partition" = "gpus48" ]; then
    summary_job=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --dependency=${dependency}
#SBATCH --output=${SUMMARY_LOG}

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail
cd "${project_root}"
export OMP_NUM_THREADS=2

srun "${uv_bin}" run --no-sync python -u evaluation/summarize_results.py ${REPORT_ARGS}
EOF
)

elif [ "$partition" = "gpus24" ]; then
    summary_job=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --dependency=${dependency}
#SBATCH --output=${SUMMARY_LOG}

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail
cd "${project_root}"
export OMP_NUM_THREADS=2

srun "${uv_bin}" run --no-sync python -u evaluation/summarize_results.py ${REPORT_ARGS}
EOF
)
fi

summary_job="${summary_job%%;*}"
if [[ ! "$summary_job" =~ ^[0-9]+$ ]]; then
    echo "Invalid summary job ID: $summary_job" >&2
    exit 1
fi
printf 'summary\tall\t%s\t%s\n' "$summary_job" "${save_dir}/report.md" >> "${save_dir}/jobs.tsv"
echo "Metric jobs: LDM=$ldm_eval_job REPA=$repa_eval_job REPA-E=$repae_eval_job"
echo "Summary job: $summary_job"
echo "Report: ${save_dir}/report.md"
