#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/run_tvr_xml_eval.sh MODEL_DIR_NAME SPLIT [XML inference.py args...]"
    echo "Example: bash scripts/run_tvr_xml_eval.sh tvr-video_sub-my_run-2026_04_06_00_00_00 val"
    exit 1
fi

model_dir_name=$1
split=$2
shift 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

xml_root="third_party/TVRetrieval"
results_root="${xml_root}/baselines/crossmodal_moment_localization/results"
model_dir_path="${results_root}/${model_dir_name}"
data_root="${TVR_DATA_DIR:-data/tvr}"
eval_path="${data_root}/tvr_${split}_release.jsonl"

required_paths=(
    "${xml_root}/baselines/crossmodal_moment_localization/inference.py"
    "${model_dir_path}"
    "${model_dir_path}/opt.json"
    "${eval_path}"
)

missing=()
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        missing+=("${path}")
    fi
done

if (( ${#missing[@]} > 0 )); then
    printf 'Missing required XML paths:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

tasks=(VCMR SVMR VR)
if [[ "${split}" == "test_public" ]]; then
    tasks=(VCMR VR)
fi

PYTHONPATH="${repo_root}/${xml_root}${PYTHONPATH:+:${PYTHONPATH}}" \
python3 "${xml_root}/baselines/crossmodal_moment_localization/inference.py" \
    --model_dir "${model_dir_name}" \
    --eval_split_name "${split}" \
    --eval_path "${eval_path}" \
    --tasks "${tasks[@]}" \
    "$@"
