#!/bin/bash
# Code-only launcher: training starts only when you explicitly run this script.
# Usage: bash train_scripts/train_attribute_predictor.sh [gpus48|gpus24|local] [trainer options] [--dry-run]
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
partition="${1:-gpus48}"
if [ $# -gt 0 ]; then shift; fi
case "$partition" in
    gpus48|gpus24|local) ;;
    *) echo "Use gpus48, gpus24, or local as the first argument." >&2; exit 1 ;;
esac

python_bin="${project_root}/.venv/bin/python"
output_dir="${project_root}/checkpoints/attribute_predictor/resnet50_256x192_seed1234"
ARGS=(--data-dir=/vol/biodata/data/Mammo/EMBED/pngs/1024x768
      --split-dir=/vol/biomedic3/tx1215/mamo-flow/assets/embed_splits_v1
      --output-dir="$output_dir" --height=256 --width=192
      --backbone=resnet50 --pretrained=imagenet --epochs=20 --patience=5
      --batch-size=64 --num-workers=4 --cpu-threads=2 --seed=1234 --device=cuda:0)
dry_run=false
for option in "$@"; do
    if [ "$option" = "--dry-run" ]; then dry_run=true; else ARGS+=("$option"); fi
done
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" = --output-dir=* ]]; then
        output_dir="${ARGS[i]#*=}"
    elif [ "${ARGS[i]}" = "--output-dir" ]; then
        i=$((i+1)); output_dir="${ARGS[i]:-}"
    fi
done
if [ -z "$output_dir" ]; then echo "--output-dir needs a value." >&2; exit 1; fi
if [ "${output_dir:0:1}" != "/" ]; then output_dir="${project_root}/${output_dir}"; fi
ARGS+=(--output-dir="$output_dir")
if [ ! -x "$python_bin" ]; then echo "Python not found: $python_bin" >&2; exit 1; fi
if [ -e "$output_dir" ]; then echo "Output already exists: $output_dir" >&2; exit 1; fi
printf 'Predictor training (%s), output: %s\n' "$partition" "$output_dir"
printf 'Command: %q -u %q ' "$python_bin" "${project_root}/evaluation/train_attribute_predictor.py"
printf '%q ' "${ARGS[@]}"; printf '\n'
if [ "$dry_run" = true ]; then exit 0; fi
if [ "$partition" != local ] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable here. Submit from a Slurm login host, or use local inside your chosen GPU allocation." >&2
    exit 1
fi
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
mkdir "$output_dir/logs"
export TORCH_HOME="${project_root}/cache/torch"
export OMP_NUM_THREADS=2
if [ "$partition" = local ]; then
    cd "$project_root"
    "$python_bin" -u evaluation/train_attribute_predictor.py "${ARGS[@]}" 2>&1 | tee "$output_dir/logs/train.out"
else
    printf -v command_line '%q ' "$python_bin" -u "${project_root}/evaluation/train_attribute_predictor.py" "${ARGS[@]}"
    printf -v root_quoted '%q' "$project_root"
    job_id=$(sbatch --parsable --partition="$partition" --gres=gpu:1 \
        --job-name=attribute_predictor --output="$output_dir/logs/train.%j.log" \
        --wrap="cd ${root_quoted} && ${command_line}")
    printf '%s\n' "$job_id" > "$output_dir/logs/job_id.txt"
    printf 'Submitted predictor job: %s\n' "$job_id"
fi
