#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

python -m src.eval.moment_retrieval \
  --split "${VIDEOMMLU_SPLIT:-Video_MMLU}" \
  --windows-dir "${SUBTITLE_WINDOWS_DIR:-data/subtitles/windows}" \
  --top-k "${TOP_K:-3}" \
  --output-dir "${VMR_OUTPUT_DIR:-outputs/vmr}"
