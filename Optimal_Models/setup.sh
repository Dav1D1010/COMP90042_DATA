#!/bin/bash
# =============================================================================
# Climatron Optimal Model — UV Environment Setup for Google Colab
# Usage: bash setup.sh
# This script installs uv, creates a venv, and installs all dependencies.
# Designed to run in < 5 minutes on Colab Free T4.
# =============================================================================
set -e

echo "=== Climatron Optimal Model Setup ==="
echo ""

# ── Step 1: Install uv (fast Python package manager by Astral) ─────────────
# uv is 10-100x faster than pip for dependency resolution and installation.
# On Colab, we install it via the official curl script.
echo "[1/4] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "       uv installed at $(which uv)"
else
    echo "       uv already installed at $(which uv)"
fi

# ── Step 2: Verify Python version ───────────────────────────────────────────
echo "[2/4] Checking Python..."
python3 --version
echo "       OK"

# ── Step 3: Create virtual environment and install dependencies ─────────────
# uv sync reads pyproject.toml and installs exact dependencies with lockfile.
# This replaces pip install -r requirements.txt with faster, reproducible installs.
echo "[3/4] Installing dependencies with uv (this may take 2-3 minutes)..."
uv venv .venv --python=python3
source .venv/bin/activate
uv pip install -r <(grep -E '^\s*"' pyproject.toml | sed 's/"//g' | sed 's/,\s*$//' | sed 's/^\s*//')
echo "       Dependencies installed"

# ── Step 4: Verify GPU and PyTorch ──────────────────────────────────────────
echo "[4/4] Verifying GPU access..."
python3 -c "
import torch
print(f'       PyTorch: {torch.__version__}')
print(f'       CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'       GPU: {props.name}')
    print(f'       VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'       Compute Capability: {props.major}.{props.minor}')
else:
    print('       WARNING: No GPU detected. Training will be CPU-only (very slow).')
"

echo ""
echo "=== Setup complete ==="
echo "Run: python train_optimal.py --max-hours 10 --fp16"
