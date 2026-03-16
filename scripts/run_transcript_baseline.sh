#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

python -m src.eval_transcript_bm25 \
  --split "${VIDEOMMLU_SPLIT:-Video_MMLU}" \
  --window-char-len "${WINDOW_CHAR_LEN:-700}" \
  --window-char-stride "${WINDOW_CHAR_STRIDE:-500}" \
  --top-k "${TOP_K:-3}" \
  --output-dir "${OUTPUT_DIR:-outputs/legacy_qa}"
