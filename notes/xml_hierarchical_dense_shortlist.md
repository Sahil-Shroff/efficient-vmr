## XML Hierarchical Dense Shortlist Prototype

### Prototype choice

Option A: video shortlist before XML inference.

This is the smallest useful step because XML already has an upstream hook for external video-retrieval results:

- `third_party/TVRetrieval/baselines/crossmodal_moment_localization/inference.py`
  - `load_external_vr_res2(...)`
  - `compute_query2ctx_info(...)`

That lets us hand XML a per-query top-N video list in official `VR` JSON format and restrict `VR` and `VCMR` to those candidates without rewriting the XML model.

### Where the shortlist is applied

Stage 1:

- `scripts/run_tvr_xml_dense_video_shortlist.py`
- dense retrieval over one subtitle document per video
- outputs official-style `VR` predictions at:
  - `outputs/tvr/xml_hierarchical/dense_shortlist/predictions_val_top10.json`

Stage 2:

- `scripts/run_tvr_xml_eval.sh`
- XML inference with:
  - `--external_inference_vr_res_path .../predictions_val_top10.json`
  - `--max_vcmr_video N`

### What XML assumes

The external shortlist file needs:

- top-level `video2idx`
- top-level `VR`
- each query row shaped like:
  - `{"desc_id": ..., "desc": ..., "predictions": [[video_idx, 0, 0, score], ...]}`

### Important systems caveat

This prototype restricts XML candidate scoring at inference time, but it does **not** eliminate XML's full context encoding pass.

`compute_context_info(...)` still encodes all videos first, then `compute_query2ctx_info(...)` applies the shortlist during `VR` / `VCMR` scoring.

So the runtime win here is real but partial:

- less query-side search / fewer candidate videos to rank
- no reduction in the up-front context-feature encoding pass

### Small upstream patch required

`third_party/TVRetrieval/baselines/crossmodal_moment_localization/config.py`

Why:

- test-time config restore was overwriting CLI `--max_vcmr_video` with the saved training value `100`
- that made shortlist eval commands silently run full-size candidate sets

Change:

- allow `max_vcmr_video` to stay overridable at inference time

### Commands used

Dense shortlist:

```bash
wheel_cuda_lib_path="$(python3 - <<'PY'
import site
from pathlib import Path
for root in (site.getusersitepackages(), *site.getsitepackages()):
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
    existing = [str(p) for p in lib_dirs if p.is_dir()]
    if existing:
        print(":".join(existing))
        break
PY
)"
LD_LIBRARY_PATH="$wheel_cuda_lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python scripts/run_tvr_xml_dense_video_shortlist.py --split val --top-k 10
```

Full XML eval reference:

```bash
bash scripts/run_tvr_xml_eval.sh \
  tvr-video_sub-xml_official_nocore_validation-2026_04_08_19_21_22 \
  val \
  --eval_id xmr_dense_full
```

Shortlisted XML eval:

```bash
bash scripts/run_tvr_xml_eval.sh \
  tvr-video_sub-xml_official_nocore_validation-2026_04_08_19_21_22 \
  val \
  --eval_id xmr_dense_top10 \
  --external_inference_vr_res_path outputs/tvr/xml_hierarchical/dense_shortlist/predictions_val_top10.json \
  --max_vcmr_video 10
```
