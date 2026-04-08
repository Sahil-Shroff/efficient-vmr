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
default_data_root="data/tvr"
if [[ ! -d "${default_data_root}" && -d "${xml_root}/data" ]]; then
    default_data_root="${xml_root}/data"
fi
data_root="${TVR_DATA_DIR:-${default_data_root}}"
eval_path="${data_root}/tvr_${split}_release.jsonl"

wheel_cuda_lib_path="$(python3 - <<'PY'
import os
import site
from pathlib import Path

for root in site.getusersitepackages(), *site.getsitepackages():
    if not root:
        continue
    base = Path(root)
    lib_dirs = [
        base / "nvidia" / "nvjitlink" / "lib",
        base / "nvidia" / "cusparse" / "lib",
        base / "nvidia" / "cublas" / "lib",
        base / "nvidia" / "cudnn" / "lib",
        base / "nvidia" / "cuda_runtime" / "lib",
        base / "nvidia" / "cuda_nvrtc" / "lib",
    ]
    existing = [str(path) for path in lib_dirs if path.is_dir()]
    if existing:
        print(":".join(existing))
        raise SystemExit(0)
print("")
PY
)"

if [[ -n "${wheel_cuda_lib_path}" ]]; then
    export LD_LIBRARY_PATH="${wheel_cuda_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

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
