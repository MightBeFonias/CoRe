#!/usr/bin/env bash
# One-shot environment setup for a Vast.ai instance (run from the repo root).
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$HF_HOME"
grep -q "HF_HOME" ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== Installing dependencies =="
pip install -q -r requirements.txt
pip install -q -U "huggingface_hub[cli]"

echo "== Free disk =="
df -h "$HF_HOME" | tail -1

echo "Setup complete. HF_HOME=$HF_HOME"
