# TVR Data Layout

This branch expects TVR assets to live outside git under `data/tvr/`.

Recommended layout:

```text
data/
  README.md
  tvr/
    tvr_train_release.jsonl
    tvr_val_release.jsonl
    tvr_test_public_release.jsonl
    tvqa_preprocessed_subtitles.jsonl
    tvr_video2dur_idx.json
    processed/
      subtitles/
        <vid_name>.jsonl
      windows/
        <vid_name>.jsonl
```

Notes:

- `tvr_*_release.jsonl` are the official split files with fields such as `vid_name`, `duration`, `ts`, `desc`, `type`, and `desc_id`.
- `tvqa_preprocessed_subtitles.jsonl` is the official subtitle JSONL used here for subtitle-window baselines.
- `processed/subtitles/` contains normalized per-clip subtitle segments.
- `processed/windows/` contains fixed-width retrieval windows built from those segments.

Nothing under `data/tvr/` is meant to be committed in this branch.
