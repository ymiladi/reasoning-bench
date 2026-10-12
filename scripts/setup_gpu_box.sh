#!/usr/bin/env bash
# One-time setup on a Linux GPU box (tested layout: 8x H100 80GB, CUDA 12.x).
# Usage:  export HF_TOKEN=hf_...   then   bash scripts/setup_gpu_box.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_TOKEN:?set HF_TOKEN (needed for the gated GPQA and ZebraLogic datasets)}"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "vllm>=0.11.0"            # Falcon-H1R model card requires vLLM >= 0.11
pip install -r requirements.txt

echo ">> Pre-downloading weights (K2 Think V2 is ~290 GB as F32 on disk; make sure you have the space)"
huggingface-cli download tiiuae/Falcon-H1R-7B
huggingface-cli download IFM/K2-Think-V2

echo ">> Freezing benchmark items into data/"
python -m bench prepare
python -m bench inspect
echo
echo "If GPQA or ZebraLogic failed: open these pages logged in as the HF_TOKEN owner, accept the terms, re-run 'python -m bench prepare --tasks gpqa_diamond zebralogic':"
echo "  https://huggingface.co/datasets/Idavidrein/gpqa"
echo "  https://huggingface.co/datasets/allenai/ZebraLogicBench-private"
