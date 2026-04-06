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
data_root="${TVR_DATA_DIR:-data/tvr}"
feature_root="${TVR_FEATURE_ROOT:-data/tvr_feature_release}"

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
    "${extra_args[@]}" \
    "$@"
