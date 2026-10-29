#!/bin/bash
set -euo pipefail

# Usage: bash sample_scripts/sample_checkpoint.sh [gpus48|gpus24] CHECKPOINT [sample_checkpoint.py options]
partition="${1:-gpus48}"
checkpoint="${2:-}"

project_root="/vol/biomedic3/tx1215/REPA-E"
uv_bin="/vol/biomedic2/tx1215/uv/bin/uv"
output_root="${project_root}/sample_results"

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
    --save-visuals

    --tasks generation reconstruction counterfactual
    --attributes view density scanner age
)

if [ $# -gt 2 ]; then
    ARGS+=("${@:3}")
fi

# Resolve the final output settings, including options supplied by the caller.
save_dir=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    option="${ARGS[i]%%=*}"
    case "$option" in
        --output-dir|--output-root|--split|--num-samples|--num-steps|--seed|--weights)
            if [[ "${ARGS[i]}" = *=* ]]; then
                value="${ARGS[i]#*=}"
            else
                i=$((i + 1))
                value="${ARGS[i]:-}"
            fi
            if [ -z "$value" ]; then
                echo "Missing value for $option." >&2
                exit 1
            fi
            case "$option" in
                --output-dir) save_dir="$value" ;;
                --output-root) output_root="$value" ;;
                --split) split="$value" ;;
                --num-samples) num_samples="$value" ;;
                --num-steps) num_steps="$value" ;;
                --seed) seed="$value" ;;
                --weights) weights="$value" ;;
            esac
            ;;
    esac
done

if [ -z "$save_dir" ]; then
    exp_name="$(basename "$(dirname "$(dirname "$checkpoint")")")"
    checkpoint_name="$(basename "$checkpoint" .pt)"
    sample_name="${split}_n${num_samples}_seed${seed}_steps${num_steps}_${weights}"
    save_dir="${output_root}/${exp_name}/step_${checkpoint_name}/${sample_name}"
fi
if [ "${save_dir:0:1}" != "/" ]; then
    save_dir="${project_root}/${save_dir}"
fi
ARGS+=(--output-dir="$save_dir")
log_dir="${save_dir}/logs"

# Reserve a new run before Slurm opens its log files.
mkdir -p "$(dirname "$save_dir")"
if ! mkdir "$save_dir"; then
    echo "Sampling folder already exists: $save_dir. Choose another --output-dir." >&2
    exit 1
fi
mkdir "$log_dir"

echo "Sampling" >&2
echo "  checkpoint = $checkpoint" >&2
echo "  partition  = $partition" >&2
echo "  images     = $save_dir" >&2
echo "  logs       = $log_dir" >&2

# Preserve argument boundaries in the submitted job's shell command.
printf -v EVAL_ARGS '%q ' "${ARGS[@]}"
printf -v SLURM_LOG '%q' "${log_dir}/sample_checkpoint.%j.log"
printf -v PYTHON_LOG_PREFIX '%q' "${log_dir}/sample_checkpoint."

if [ "$partition" = "gpus48" ]; then
    submission=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=${SLURM_LOG}

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail

cd "${project_root}"

export TORCH_HOME="${project_root}/cache/torch"
mkdir -p "\$TORCH_HOME"
export OMP_NUM_THREADS=2
export TQDM_MININTERVAL=300

nvidia-smi

srun "${uv_bin}" run --no-sync python -u sample/sample_checkpoint.py ${EVAL_ARGS} 2>&1 | tee -a ${PYTHON_LOG_PREFIX}"\$SLURM_JOB_ID.out"
EOF
)

elif [ "$partition" = "gpus24" ]; then
    submission=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=${SLURM_LOG}

source ~/.bashrc
unset LD_LIBRARY_PATH
set -eo pipefail

cd "${project_root}"

export TORCH_HOME="${project_root}/cache/torch"
mkdir -p "\$TORCH_HOME"
export OMP_NUM_THREADS=2
export TQDM_MININTERVAL=300

nvidia-smi

srun "${uv_bin}" run --no-sync python -u sample/sample_checkpoint.py ${EVAL_ARGS} 2>&1 | tee -a ${PYTHON_LOG_PREFIX}"\$SLURM_JOB_ID.out"
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
