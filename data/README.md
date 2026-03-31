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
    videos/
      <vid_name>.mp4
    processed/
      subtitles/
        <vid_name>.jsonl
      windows/
        <vid_name>.jsonl
      visual_features/
        <vid_name>.npz
```

Notes:

- `tvr_*_release.jsonl` are the official split files with fields such as `vid_name`, `duration`, `ts`, `desc`, `type`, and `desc_id`.
- `tvqa_preprocessed_subtitles.jsonl` is the official subtitle JSONL used here for subtitle-window baselines.
- `videos/` is optional and should contain local TVR clips when you want to build CLIP frame features.
- `processed/subtitles/` contains normalized per-clip subtitle segments.
- `processed/windows/` contains fixed-width retrieval windows built from those segments.
- `processed/visual_features/` contains compressed CLIP frame embeddings with `timestamps` and `embeddings` arrays per clip.

Nothing under `data/tvr/` is meant to be committed in this branch.
