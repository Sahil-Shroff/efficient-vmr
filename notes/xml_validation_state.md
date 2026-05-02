# XML Validation State

Date: 2026-04-08
Branch: `tvr/baselines/xml`

## Environment

- Python: `3.10.17`
- GPU: `Tesla T4`
- NVIDIA driver: `550.90.07`
- PyTorch: `2.5.1+cu124`
- Torch CUDA runtime: `12.4`
- NumPy: `1.25.2`
- h5py: `3.16.0`
- tensorboard: `2.20.0`
- easydict: installed

Wrapper note:

- The XML wrappers prepend the PyTorch wheel CUDA library directories under
  `~/.local/lib/python3.10/site-packages/nvidia/.../lib` to `LD_LIBRARY_PATH`
  so the CUDA 12.4 wheel does not bind against `/usr/local/cuda/lib64`.

## Data Paths Used

- Train queries: `third_party/TVRetrieval/data/tvr_train_release.jsonl`
- Val queries: `third_party/TVRetrieval/data/tvr_val_release.jsonl`
- Duration index: `third_party/TVRetrieval/data/tvr_video2dur_idx.json`
- Feature root: `/home/jupyter/data/tvr_feature_release`
- Query features: `/home/jupyter/data/tvr_feature_release/bert_feature/sub_query/tvr_query_pretrained_w_sub_query.h5`
- Subtitle features: `/home/jupyter/data/tvr_feature_release/bert_feature/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5`
- Video features: `/home/jupyter/data/tvr_feature_release/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5`

## Upstream Files Patched

- `baselines/crossmodal_moment_localization/inference.py`
  - replace deprecated NumPy aliases
  - fix debug-mode query/result length mismatch during inference
- `standalone_eval/eval.py`
  - replace deprecated NumPy boolean aliases
- `baselines/clip_alignment_with_language/proposal_retrieval_dataset.py`
  - replace deprecated NumPy integer aliases
- `utils/text_feature/convert_sub_feature_word_to_clip.py`
  - replace deprecated NumPy integer aliases
- `utils/text_feature/lm_finetuning_on_single_sentences.py`
  - replace deprecated NumPy integer aliases

## Wrapper Files Patched

- `scripts/run_tvr_xml_train.sh`
  - resolve metadata/features from local validation paths
  - select CUDA when available, CPU otherwise
  - prepend PyTorch wheel CUDA library directories
- `scripts/run_tvr_xml_eval.sh`
  - resolve metadata fallback paths
  - prepend PyTorch wheel CUDA library directories

## Runtime Deviation Used For Validation

- Standard official non-debug XML uses in-memory HDF5 loading by default.
- On this host, that configuration was killed before the first validation pass.
- Validation was rerun with `--no_core_driver`, which keeps HDF5 access on disk instead of copying the
  feature files into RAM.
- This is an infrastructure/runtime deviation, not a modeling change:
  - same model
  - same data
  - same official XML train/inference code path
  - same official `video_sub + resnet_i3d` setup
  - only data-loading mode changed so the run fits host memory
