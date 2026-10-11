#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/run_tvr_xml_train.sh CTX_MODE VID_FEAT_TYPE [XML train.py args...]"
    echo "Example: bash scripts/run_tvr_xml_train.sh video_sub resnet_i3d --exp_id xml_debug --debug"
    exit 1
fi

ctx_mode=$1
vid_feat_type=$2
shift 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

xml_root="third_party/TVRetrieval"
results_root="${xml_root}/baselines/crossmodal_moment_localization/results"
default_data_root="data/tvr"
if [[ ! -d "${default_data_root}" && -d "${xml_root}/data" ]]; then
    default_data_root="${xml_root}/data"
fi
data_root="${TVR_DATA_DIR:-${default_data_root}}"
default_feature_root="data/tvr_feature_release"
if [[ ! -d "${default_feature_root}" && -d "/mnt/tvr_disk/data/tvr_feature_release" ]]; then
    default_feature_root="/mnt/tvr_disk/data/tvr_feature_release"
fi
if [[ ! -d "${default_feature_root}" && -d "/home/jupyter/data/tvr_feature_release" ]]; then
    default_feature_root="/home/jupyter/data/tvr_feature_release"
fi
feature_root="${TVR_FEATURE_ROOT:-${default_feature_root}}"
device_arg=()

wheel_cuda_lib_path="$(python3 - <<'PY'
import os
import site
from pathlib import Path

candidates = []
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

if [[ -n "${TVR_XML_DEVICE:-}" ]]; then
    device_arg=(--device "${TVR_XML_DEVICE}")
else
    if python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
        device_arg=(--device 0)
    else
        device_arg=(--device -1)
    fi
fi

train_path="${data_root}/tvr_train_release.jsonl"
eval_path="${data_root}/tvr_val_release.jsonl"
video_duration_idx_path="${data_root}/tvr_video2dur_idx.json"
desc_bert_path="${feature_root}/bert_feature/query_only/tvr_query_pretrained_w_query.h5"
sub_bert_path=""
vid_feat_path=""
vid_feat_size=2048
extra_args=()

case "${vid_feat_type}" in
    i3d)
        vid_feat_path="${feature_root}/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5"
        vid_feat_size=1024
        ;;
    resnet)
        vid_feat_path="${feature_root}/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5"
        vid_feat_size=2048
        ;;
    resnet_i3d)
        vid_feat_path="${feature_root}/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5"
        vid_feat_size=3072
        extra_args+=(--no_norm_vfeat)
        ;;
    *)
        echo "Unknown VID_FEAT_TYPE: ${vid_feat_type}"
        exit 1
        ;;
esac

if [[ "${ctx_mode}" == *"sub"* ]] || [[ "${ctx_mode}" == "sub" ]]; then
    desc_bert_path="${feature_root}/bert_feature/sub_query/tvr_query_pretrained_w_sub_query.h5"
    sub_bert_path="${feature_root}/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5"
    extra_args+=(--sub_feat_size 768 --sub_bert_path "${sub_bert_path}")
fi

required_paths=(
    "${xml_root}/baselines/crossmodal_moment_localization/train.py"
    "${train_path}"
    "${eval_path}"
    "${video_duration_idx_path}"
    "${desc_bert_path}"
    "${vid_feat_path}"
)
if [[ -n "${sub_bert_path}" ]]; then
    required_paths+=("${sub_bert_path}")
fi

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

mkdir -p "${results_root}"

PYTHONPATH="${repo_root}/${xml_root}${PYTHONPATH:+:${PYTHONPATH}}" \
python3 "${xml_root}/baselines/crossmodal_moment_localization/train.py" \
    --dset_name=tvr \
    --eval_split_name=val \
    --nms_thd=-1 \
    --results_root="${results_root}" \
    --train_path="${train_path}" \
    --eval_path="${eval_path}" \
    --video_duration_idx_path="${video_duration_idx_path}" \
    --desc_bert_path="${desc_bert_path}" \
    --vid_feat_path="${vid_feat_path}" \
    --clip_length=1.5 \
    --vid_feat_size="${vid_feat_size}" \
    --ctx_mode="${ctx_mode}" \
    --max_ctx_l=100 \
    --max_pred_l=16 \
    "${device_arg[@]}" \
    "${extra_args[@]}" \
    "$@"
