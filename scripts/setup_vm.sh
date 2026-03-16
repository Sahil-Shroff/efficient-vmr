#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential ffmpeg

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "Setup complete. Activate with: source .venv/bin/activate"
