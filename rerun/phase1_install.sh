#!/usr/bin/env bash
# Phase 1: provision torch_xla 2.4 venv on the v5e VM.
# Run this INSIDE the v5e VM (after gcloud compute tpus tpu-vm ssh).
#
# See analysis/SETUP.md for the full plan.

set -euo pipefail

echo "==> creating ~/torch-tpu-2.4 venv"
python3 -m venv ~/torch-tpu-2.4
source ~/torch-tpu-2.4/bin/activate

echo "==> upgrading pip"
pip install --upgrade pip

echo "==> installing torch 2.4 + torch_xla[tpu] 2.4 (matches notebook's design)"
pip install 'torch==2.4.0' --index-url https://download.pytorch.org/whl/cpu
pip install 'torch_xla[tpu]==2.4.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html

echo "==> installing profiling-side Python deps (per notebook cell 0 + extras)"
pip install 'transformers==4.45.0' 'accelerate>=0.33' sentencepiece \
            numpy pandas tqdm huggingface_hub

echo "==> smoke tests"
python3 - <<'PY'
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp

print("torch_xla version:", torch_xla.__version__)
print("xla device      :", xm.xla_device())
print("has xp.start_trace:", hasattr(xp, "start_trace"))
print("has xp.stop_trace :", hasattr(xp, "stop_trace"))
print("has torch_xla.sync:", hasattr(torch_xla, "sync"))
print("has torch_xla.device:", hasattr(torch_xla, "device"))
PY

echo ""
echo "==> expected output above:"
echo "    has xp.start_trace: True"
echo "    has xp.stop_trace : True"
echo "    (sync/device may be False on 2.3-style builds; the compat shim in profiler_core.py handles that)"
echo ""
echo "==> Phase 1 done. Next: copy rerun/ scripts and run_profile.py smoke."
