# XML Hybrid Validation

Environment note:
- XML feature root auto-resolved from the mounted secondary disk at `/mnt/tvr_disk/data/tvr_feature_release`.
- `scripts/run_tvr_xml_eval.sh` now detects feature roots locally and overrides stale checkpoint feature paths.
- `scripts/run_tvr_xml_train.sh` now recognizes the same mounted-disk feature root.

Completed runs:
- Full XML baseline: existing validated run reused from `outputs/tvr/xml_hierarchical/dense_shortlist_xml_eval_summary.json`
- Dense -> XML top-5: existing validated run reused from `outputs/tvr/xml_hierarchical/dense_shortlist_xml_eval_summary.json`
- Shared compaction -> XML keep=0.6:
  - command: `bash scripts/run_tvr_xml_eval.sh tvr-video_sub-xml_official_nocore_validation-2026_04_08_19_21_22 val --eval_id shared_compact_keep06 --shared_compact_keep_ratio 0.6 --shared_compact_num_spans 3`
  - runtime: 121s
- Dense -> shared compaction -> XML top-5 keep=0.6:
  - command: `bash scripts/run_tvr_xml_hybrid_eval.sh tvr-video_sub-xml_official_nocore_validation-2026_04_08_19_21_22 val 5 0.6`
  - runtime: 114s

Results:

| Method | VR R@1 | VCMR R@1@0.5 | SVMR R@1@0.5 | Runtime | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full XML | 1.69 | 0.38 | 15.60 | 80.15s | 1.00x |
| Dense -> XML top-5 | 11.56 | 1.43 | 15.60 | 54.05s | 1.48x |
| Shared compaction -> XML keep=0.6 | 1.60 | 0.25 | 11.23 | 121s | 0.66x |
| Dense -> shared compaction -> XML top-5 keep=0.6 | 11.56 | 1.17 | 11.56 | 114s | 0.70x |

Interpretation:
- The mounted-disk feature-path issue is resolved locally in the wrappers.
- Shared compaction alone is slower than the previously measured full XML baseline on this host and drops accuracy materially.
- The combined hybrid run preserves the dense-shortlist VR gains and improves over compaction-only on VCMR/SVMR, but at `114s` it is still slower than both full XML (`80.15s`) and dense -> XML top-5 (`54.05s`).
- On this host, the current shared-span selector is still the limiting factor for runtime value, so hierarchical pruning remains the stronger efficiency story.
