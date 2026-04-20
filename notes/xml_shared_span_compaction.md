# XML Shared-Span Compaction Baseline

## Feasibility

Joint video+subtitle compaction is feasible for XML in the validated `video_sub` setup as long as both modalities are compacted on the same retained timeline.

Smallest safe insertion point:
- `StartEndEvalDataset._get_item_context(...)`
- compact raw video and subtitle features together before context encoding
- keep `retained_clip_indices` in `meta` so inference can map compact indices back to original clip indices

Why this works:
- XML cross-attention can handle shorter shared sequences at inference time
- the structural blocker from visual-only compaction was the fused `video_sub` span path assuming equal temporal length
- shared compaction preserves that equality

## Mapping

For each video, the compacted timeline stores:

```python
retained_clip_indices = [5, 6, 7, 20, 21, 22]
```

Inference runs on the compacted sequence. When XML predicts compact indices `(start_idx, end_idx)`, inference maps back with:

```python
original_start_idx = retained_clip_indices[start_idx]
original_end_idx = retained_clip_indices[end_idx] + 1
```

The `+1` keeps XML's original exclusive-end time conversion semantics.

Because multiple retained spans are concatenated, inference also applies a contiguity mask so a predicted span cannot cross a dropped gap.

## Cheap selector

This baseline uses feature-space change scoring on the XML clip grid:
- score each clip by adjacent visual feature difference
- greedily keep top contiguous windows
- compact both video and subtitle streams with the same retained indices

No training changes were made.

## Runtime note

Shared compaction reduces real runtime because context encoding and span scoring now run on shorter aligned sequences.

At aggressive compaction (`20%`), some queries produce no valid contiguous span after masking. A tiny eval-only fallback emits one zero-score retained-timeline span so the official evaluator can score the run instead of crashing on an empty prediction list.
