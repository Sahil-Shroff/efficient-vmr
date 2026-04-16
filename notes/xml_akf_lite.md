## XML AKF-Lite Prototype

### Implementation approach

Hook location:

- `third_party/TVRetrieval/baselines/crossmodal_moment_localization/inference.py`
- inside `compute_query2ctx_info(...)`

Why here:

- query embeddings are available through `model.encode_query(...)`
- encoded visual clip features are already in `ctx_info["video_feat1"]` / `ctx_info["video_feat2"]`
- masking here avoids model changes and keeps the original clip grid, temporal order, and tensor shapes

AKF-lite scoring:

- per-query visual clip score
- `score = similarity(query_embedding, visual_clip_embedding)`
- implemented with `compute_visual_clip_scores(...)`

Selection strategies implemented:

- `topk`
- `contiguous`

Primary ablation run:

- `topk`
- keep ratios `1.0`, `0.6`, `0.4`, `0.2`

Masking behavior:

- build a query-conditioned visual clip mask
- suppress dropped clips with `mask_logits(...)`
- subtitle branch is unchanged
- temporal indices remain on the original 1.5s grid

### Important limitation

This prototype is shape-preserving, not compute-pruning.

It masks visual clips after query-conditioned scoring, but XML still:

- loads the full visual sequence
- encodes the full context sequence
- runs span heads at the original sequence length

So this is a clean accuracy ablation, but not yet a real speedup path.

### Files and artifacts

Code:

- `third_party/TVRetrieval/baselines/crossmodal_moment_localization/config.py`
- `third_party/TVRetrieval/baselines/crossmodal_moment_localization/inference.py`

Artifacts:

- `outputs/tvr/xml_akf/clip_scores_val_gt_video.npz`
- `outputs/tvr/xml_akf/akf_topk_ablation.csv`
- `outputs/tvr/xml_akf/akf_topk_ablation.json`

Logs:

- `logs/xml_akf/akf_topk_100.log`
- `logs/xml_akf/akf_topk_60.log`
- `logs/xml_akf/akf_topk_40.log`
- `logs/xml_akf/akf_topk_20.log`
