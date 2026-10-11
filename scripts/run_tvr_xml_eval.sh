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
model_opt_path="${model_dir_path}/opt.json"
default_data_root="data/tvr"
if [[ ! -d "${default_data_root}" && -d "${xml_root}/data" ]]; then
    default_data_root="${xml_root}/data"
fi
data_root="${TVR_DATA_DIR:-${default_data_root}}"
eval_path="${data_root}/tvr_${split}_release.jsonl"
default_feature_root="data/tvr_feature_release"
if [[ ! -d "${default_feature_root}" && -d "/mnt/tvr_disk/data/tvr_feature_release" ]]; then
    default_feature_root="/mnt/tvr_disk/data/tvr_feature_release"
fi
if [[ ! -d "${default_feature_root}" && -d "/home/jupyter/data/tvr_feature_release" ]]; then
    default_feature_root="/home/jupyter/data/tvr_feature_release"
fi
feature_root="${TVR_FEATURE_ROOT:-${default_feature_root}}"

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
    "${model_opt_path}"
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

readarray -t model_feature_paths < <(
    python3 - "${model_opt_path}" "${feature_root}" <<'PY'
import json
import sys
from pathlib import Path

opt_path = Path(sys.argv[1])
feature_root = Path(sys.argv[2])
saved_opt = json.loads(opt_path.read_text(encoding="utf-8"))

paths = []
for key in ("desc_bert_path", "sub_bert_path", "vid_feat_path"):
    saved_path = saved_opt.get(key)
    if not saved_path:
        paths.append("")
        continue
    basename = Path(saved_path).name
    matches = sorted(feature_root.rglob(basename))
    if not matches:
        raise SystemExit(f"Could not find {basename} under {feature_root}")
    paths.append(str(matches[0]))

print("\n".join(paths))
PY
)

desc_bert_path="${model_feature_paths[0]}"
sub_bert_path="${model_feature_paths[1]}"
vid_feat_path="${model_feature_paths[2]}"

feature_required_paths=("${desc_bert_path}" "${vid_feat_path}")
if [[ -n "${sub_bert_path}" ]]; then
    feature_required_paths+=("${sub_bert_path}")
fi

feature_args=(
    --desc_bert_path "${desc_bert_path}"
    --vid_feat_path "${vid_feat_path}"
)
if [[ -n "${sub_bert_path}" ]]; then
    feature_args+=(--sub_bert_path "${sub_bert_path}")
fi

missing=()
for path in "${feature_required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        missing+=("${path}")
    fi
done

if (( ${#missing[@]} > 0 )); then
    printf 'Missing required XML feature paths:\n' >&2
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
    "${feature_args[@]}" \
    "$@"
