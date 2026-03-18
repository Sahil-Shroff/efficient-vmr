#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate

PYTHONPATH=. streamlit run src/apps/vmr_annotation_app.py --server.headless true
