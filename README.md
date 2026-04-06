# TVR Baselines

This branch is dedicated to **TVR (TV show Retrieval)** baselines.

The focus is:

- subtitle/text-first moment retrieval
- simple, interpretable baselines
- clean evaluation and latency reporting
- a codebase that is easy to extend toward dense retrieval and multimodal fusion later

This repo does **not** use the original TVR codebase as its implementation base. It only uses the TVR paper and official repo as reference for dataset schema, splits, and evaluation conventions.

## Scope

Current baselines:

- subtitle-window BM25 retrieval
- dense subtitle-window retrieval
- BM25 + dense subtitle-window fusion
- dense + visual reranking for `v`-heavy queries
- sanity baselines: random window, full-video span, oracle-best window

Current task setup:

- input: natural-language query + associated `vid_name`
- retrieval target: timestamped subtitle windows from that clip
- output: predicted `(start_time, end_time)`

## Repo Layout

```text
data/
  README.md
scripts/
  inspect_tvr.py
  build_tvr_windows.py
  build_tvr_visual_features.py
  run_tvr_bm25.py
  run_tvr_dense.py
  run_tvr_fusion.py
  run_tvr_oracle.py
  run_tvr_visual_rerank.py
  setup_vm.sh
src/
  data/
    tvr.py
  eval/
    tvr_eval.py
  retrieval/
    bm25.py
    dense.py
    fusion.py
  subtitles/
    parsing.py
    windows.py
  utils/
    io.py
    text.py
    time.py
  visual/
    features.py
```

## TVR Schema

The loader is built around the official query format, for example:

```json
{
  "vid_name": "friends_s01e03_seg02_clip_19",
  "duration": 61.46,
  "ts": [16.48, 33.87],
  "desc": "Phoebe puts one of her ponytails in her mouth.",
  "type": "v",
  "desc_id": 90200
}
```

Normalized query fields used in this repo:

- `query_id`
- `desc_id`
- `vid_name`
- `duration`
- `query`
- `query_type`
- `gold_start_time`
- `gold_end_time`

The subtitle baseline expects timestamped subtitle segments. The official TVR reference repo provides `tvqa_preprocessed_subtitles.jsonl`, where each row contains a `vid_name` and a `sub` list of `{text, start, end}` subtitle items. This repo normalizes those into per-clip JSONL segment files and then builds overlapping retrieval windows.

## Setup

```bash
bash scripts/setup_vm.sh
source .venv/bin/activate
```

## Expected Data Layout

See [data/README.md](/home/ext_sash8218_colorado_edu/efficient-vmr/data/README.md) for the expected external TVR data layout.

In short, place the official TVR metadata and subtitle files under `data/tvr/`.

## Commands

Inspect TVR:

```bash
source .venv/bin/activate
python scripts/inspect_tvr.py --data-dir data/tvr --split val
```

Normalize subtitles and build windows:

```bash
source .venv/bin/activate
python scripts/build_tvr_windows.py \
  --data-dir data/tvr \
  --subtitles-path data/tvr/tvqa_preprocessed_subtitles.jsonl \
  --parsed-dir data/tvr/processed/subtitles \
  --windows-dir data/tvr/processed/windows \
  --window-size-sec 12 \
  --stride-sec 6
```

Run BM25 baseline:

```bash
source .venv/bin/activate
python scripts/run_tvr_bm25.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --top-k 3 \
  --output-dir outputs/tvr/bm25
```

Run dense subtitle-window baseline:

```bash
source .venv/bin/activate
python scripts/run_tvr_dense.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 3 \
  --output-dir outputs/tvr/dense
```

Run BM25 + dense fusion baselines:

```bash
source .venv/bin/activate
python scripts/run_tvr_fusion.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --alphas 0.0,0.25,0.5,0.75,1.0 \
  --rrf-ks 10,60 \
  --bm25-topn-rerank 10,20,50 \
  --output-dir outputs/tvr/fusion
```

Build CLIP frame features from local TVR clips:

```bash
source .venv/bin/activate
python scripts/build_tvr_visual_features.py \
  --data-dir data/tvr \
  --split val \
  --video-dir data/tvr/videos \
  --output-dir data/tvr/processed/visual_features \
  --model-name openai/clip-vit-base-patch32 \
  --sample-stride-sec 1.0
```

Run dense + visual reranking for `v` queries:

```bash
source .venv/bin/activate
python scripts/run_tvr_visual_rerank.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --visual-features-dir data/tvr/processed/visual_features \
  --candidate-top-k 10 \
  --top-k 3 \
  --text-weight 0.6 \
  --visual-weight 0.4 \
  --apply-query-types v \
  --output-dir outputs/tvr/visual_rerank
```

Run sanity/oracle baselines:

```bash
source .venv/bin/activate
python scripts/run_tvr_oracle.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --baseline random_window \
  --output-dir outputs/tvr/oracle
```

```bash
source .venv/bin/activate
python scripts/run_tvr_oracle.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --baseline full_video \
  --output-dir outputs/tvr/oracle
```

```bash
source .venv/bin/activate
python scripts/run_tvr_oracle.py \
  --data-dir data/tvr \
  --split val \
  --windows-dir data/tvr/processed/windows \
  --baseline oracle_window \
  --output-dir outputs/tvr/oracle
```

## Reported Metrics

The evaluator reports:

- mean IoU
- Recall@1 at IoU >= 0.3
- Recall@1 at IoU >= 0.5
- Recall@3 at IoU >= 0.3
- Recall@3 at IoU >= 0.5
- mean absolute start error
- mean absolute end error
- average retrieval latency
- p50 retrieval latency
- p95 retrieval latency

It also provides a breakdown by query type:

- `v`
- `t`
- `vt`

## Assumptions

- The default subtitle source is the official `tvqa_preprocessed_subtitles.jsonl`.
- Subtitle rows contain `vid_name` plus a `sub` list of timestamped segments.
- Window construction currently targets **per-clip moment retrieval**, where retrieval is restricted to the query's ground-truth clip.
- Corpus-level retrieval is intentionally not implemented yet to keep the baseline path simple and readable.
- The dense retriever uses sentence-transformer text embeddings over subtitle windows and shares the same eval pipeline as BM25.
- The visual reranker expects local video clips under `data/tvr/videos/` or precomputed CLIP frame features under `data/tvr/processed/visual_features/`.
- Visual reranking is intended as a targeted extension for visually grounded queries, so it defaults to applying only on query type `v`.

## Next Steps

Good next extensions after this baseline layer:

- add clip-level retrieval before within-clip window localization
- add late fusion with visual features
- compare accuracy/latency tradeoffs across BM25, dense retrieval, and multimodal reranking
