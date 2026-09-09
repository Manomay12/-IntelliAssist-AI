#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=========================================="
echo " Starting IntelliAssist AI Build on Render"
echo "=========================================="

# 1. Upgrade pip, setuptools, and wheel
echo "==> Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

# 2. Install lightweight CPU-only PyTorch first
# This prevents downloading 2.5GB+ of unused CUDA binaries on Linux / Render free tier
echo "==> Installing CPU-only PyTorch wheels (saving ~2.3 GB disk & RAM)..."
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining application dependencies
echo "==> Installing project requirements..."
pip install --no-cache-dir -r requirements.txt

# 4. Ensure runtime data folders exist
echo "==> Creating required data directories..."
mkdir -p data/uploads data/vector_db data/conversations

echo "=========================================="
echo " IntelliAssist AI Build Completed Successfully"
echo "=========================================="
