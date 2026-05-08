#!/bin/bash
cd /root/autodl-tmp/cloud_deploy
source .venv/bin/activate
export HF_TOKEN="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"
export HUGGINGFACE_HUB_TOKEN="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"
export WANDB_API_KEY="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"
export DEEPSEEK_API_KEY="sk-96e8fddcb45b466c960f1fde4bc45444"
export HF_ENDPOINT="https://hf-mirror.com"
exec python -u train_all.py --max-hours 8 --fp16 --variants v1 v2 v3 v4 v7 --quick-tokens 180000000 --full-tokens 1 --batch-size 16 --grad-accum 1
