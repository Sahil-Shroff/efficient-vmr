#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash scripts/run_tvr_xml_hybrid_eval.sh MODEL_DIR_NAME SPLIT TOP_K KEEP_RATIO [XML inference.py args...]"
    echo "Example: bash scripts/run_tvr_xml_hybrid_eval.sh tvr-video_sub-my_run-2026_04_08_00_00_00 val 5 0.6"
    exit 1
fi

model_dir_name=$1
split=$2
top_k=$3
keep_ratio=$4
shift 4

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

keep_ratio_tag="${keep_ratio//./}"
hybrid_dir="outputs/tvr/xml_hybrid/dense_top${top_k}_keep${keep_ratio_tag}"
eval_id="hybrid_top${top_k}_keep${keep_ratio_tag}"

wheel_cuda_lib_path="$(python3 - <<'PY'
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

python3 scripts/run_tvr_xml_dense_video_shortlist.py \
    --split "${split}" \
    --top-k "${top_k}" \
    --output-dir "${hybrid_dir}"

shortlist_path="${hybrid_dir}/predictions_${split}_top${top_k}.json"

bash scripts/run_tvr_xml_eval.sh "${model_dir_name}" "${split}" \
    --eval_id "${eval_id}" \
    --external_inference_vr_res_path "${shortlist_path}" \
    --max_vcmr_video "${top_k}" \
    --shared_compact_keep_ratio "${keep_ratio}" \
    --shared_compact_num_spans 3 \
    --context_video_shortlist_path "${shortlist_path}" \
    --context_video_shortlist_topk "${top_k}" \
    "$@"
