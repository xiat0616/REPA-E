#!/bin/bash
set -euo pipefail

# Usage: bash sample_scripts/sample_all.sh [gpus48|gpus24] CHECKPOINT [sample_all.py options]
partition="${1:-gpus48}"
checkpoint="${2:-}"

project_root="/vol/biomedic3/tx1215/REPA-E"
uv_bin="/vol/biomedic2/tx1215/uv/bin/uv"
output_root="${project_root}/sample_results"
log_dir="${SAMPLE_LOG_DIR:-${output_root}/logs}"

weights="ema"
split="test"
num_samples=5000
num_steps=50
batch_size=4
num_workers=0
seed=1234
cfg_scale=1.0

if [ "$partition" != "gpus48" ] && [ "$partition" != "gpus24" ]; then
    echo "Unknown partition: $partition. Use gpus48 or gpus24." >&2
    exit 1
fi

if [ -z "$checkpoint" ]; then
    echo "Supply a checkpoint path as the second argument." >&2
    exit 1
fi

if [ "${checkpoint:0:1}" != "/" ]; then
    checkpoint="${project_root}/${checkpoint}"
fi

if [ ! -f "$checkpoint" ]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
fi

if [ ! -x "$uv_bin" ]; then
    echo "uv executable not found: $uv_bin" >&2
    exit 1
fi

if [ "${log_dir:0:1}" != "/" ]; then
    log_dir="${project_root}/${log_dir}"
fi

ARGS=(
    --checkpoint="$checkpoint"
    --weights="$weights"
    --split="$split"
    --num-samples=$num_samples
    --num-steps=$num_steps
    --batch-size=$batch_size
    --num-workers=$num_workers
    --seed=$seed
    --cfg-scale=$cfg_scale
    --output-root="$output_root"

    --tasks generation reconstruction counterfactual
    --attributes view density scanner age
)

if [ $# -gt 2 ]; then
    ARGS+=("${@:3}")
fi

mkdir -p "$log_dir"

echo "Sampling" >&2
echo "  checkpoint = $checkpoint" >&2
echo "  partition  = $partition" >&2
echo "  logs       = $log_dir" >&2

# Preserve argument boundaries in the submitted job's shell command.
printf -v EVAL_ARGS '%q ' "${ARGS[@]}"

if [ "$partition" = "gpus48" ]; then
    submission=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output="${log_dir}/sample_all.%j.log"

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail

cd "${project_root}"

export TORCH_HOME="${project_root}/cache/torch"
mkdir -p "\$TORCH_HOME"
export OMP_NUM_THREADS=2
export TQDM_MININTERVAL=300

nvidia-smi

srun "${uv_bin}" run --no-sync python -u sample/sample_all.py ${EVAL_ARGS} 2>&1 | tee -a "${log_dir}/sample_all.\$SLURM_JOB_ID.out"
EOF
)

elif [ "$partition" = "gpus24" ]; then
    submission=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output="${log_dir}/sample_all.%j.log"

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail

cd "${project_root}"

export TORCH_HOME="${project_root}/cache/torch"
mkdir -p "\$TORCH_HOME"
export OMP_NUM_THREADS=2
export TQDM_MININTERVAL=300

nvidia-smi

srun "${uv_bin}" run --no-sync python -u sample/sample_all.py ${EVAL_ARGS} 2>&1 | tee -a "${log_dir}/sample_all.\$SLURM_JOB_ID.out"
EOF
)
fi

# sbatch --parsable may append a cluster name after a semicolon.
job_id="${submission%%;*}"
case "$job_id" in
    ''|*[!0-9]*)
        echo "Could not parse sbatch job ID from: $submission" >&2
        exit 1
        ;;
esac

echo "Submitted sampling job $job_id" >&2
printf '%s\n' "$job_id"
