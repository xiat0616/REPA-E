#!/bin/bash
set -euo pipefail

# Usage: bash eval_scripts/evaluate_all.sh [gpus48|gpus24] SAMPLES_DIR [metric options]
partition="${1:-gpus48}"
samples_dir="${2:-}"

project_root="/vol/biomedic3/tx1215/REPA-E"
uv_bin="/vol/biomedic2/tx1215/uv/bin/uv"
output_root="${project_root}/evaluation/results"
log_dir="${EVAL_LOG_DIR:-${output_root}/logs}"

# These settings affect metrics only; sampling settings are loaded from the cache.
batch_size=4
num_workers=0
feature_model="dinov2_vitb14"
save_examples=0

if [ "$partition" != "gpus48" ] && [ "$partition" != "gpus24" ]; then
    echo "Unknown partition: $partition. Use gpus48 or gpus24." >&2
    exit 1
fi

if [ -z "$samples_dir" ]; then
    echo "Supply the saved-image directory as the second argument." >&2
    exit 1
fi

if [ "${samples_dir:0:1}" != "/" ]; then
    samples_dir="${project_root}/${samples_dir}"
fi

# A dependency allows submission before the sampling job has created its cache.
SUBMIT_ARGS=(--parsable)
if [ -n "${EVAL_DEPENDENCY:-}" ]; then
    if [[ ! "$EVAL_DEPENDENCY" =~ ^[0-9]+(:[0-9]+)*$ ]]; then
        echo "EVAL_DEPENDENCY must contain Slurm job IDs separated by colons." >&2
        exit 1
    fi
    SUBMIT_ARGS+=(--dependency="afterok:${EVAL_DEPENDENCY}")
elif [ ! -f "${samples_dir}/COMPLETE" ]; then
    echo "No completed sampling run found: $samples_dir" >&2
    exit 1
fi

if [ ! -x "$uv_bin" ]; then
    echo "uv executable not found: $uv_bin" >&2
    exit 1
fi

ARGS=(
    --samples-dir="$samples_dir"
    --batch-size=$batch_size
    --num-workers=$num_workers
    --output-root="$output_root"
    --save-examples=$save_examples

    --evaluations identity inception dino
    --identity-metrics mae mse ssim lpips
    --feature-model="$feature_model"
)

if [ $# -gt 2 ]; then
    ARGS+=("${@:3}")
fi

mkdir -p "$log_dir"
echo "Saved-image evaluation" >&2
echo "  samples    = $samples_dir" >&2
echo "  partition  = $partition" >&2
echo "  logs       = $log_dir" >&2

printf -v EVAL_ARGS '%q ' "${ARGS[@]}"

if [ "$partition" = "gpus48" ]; then
    submission=$(sbatch "${SUBMIT_ARGS[@]}" <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output="${log_dir}/evaluate_all.%j.log"

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail
cd "${project_root}"
export TORCH_HOME="${project_root}/cache/torch"
export OMP_NUM_THREADS=2
export TQDM_MININTERVAL=300

nvidia-smi

srun "${uv_bin}" run --no-sync python -u evaluation/evaluate_all.py ${EVAL_ARGS} 2>&1 | tee -a "${log_dir}/evaluate_all.\$SLURM_JOB_ID.out"
EOF
)

elif [ "$partition" = "gpus24" ]; then
    submission=$(sbatch "${SUBMIT_ARGS[@]}" <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output="${log_dir}/evaluate_all.%j.log"

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail
cd "${project_root}"
export TORCH_HOME="${project_root}/cache/torch"
export OMP_NUM_THREADS=2
export TQDM_MININTERVAL=300

nvidia-smi

srun "${uv_bin}" run --no-sync python -u evaluation/evaluate_all.py ${EVAL_ARGS} 2>&1 | tee -a "${log_dir}/evaluate_all.\$SLURM_JOB_ID.out"
EOF
)
fi

job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch job ID from: $submission" >&2
    exit 1
fi

echo "Submitted metric job $job_id" >&2
printf '%s\n' "$job_id"
