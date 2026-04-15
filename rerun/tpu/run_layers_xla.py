#!/usr/bin/env python3
"""Faithful author-pipeline adapter for TPU profiling.

Runs `profiler.layers.main.main()` exactly as in `profile_layers.sh`, with:
  - `--device xla`
  - `--profile-method perf_counter`  (author-provided method, Timer:54-60/88-93)

Only one thing is injected: `torch.cuda.synchronize` and `torch.cuda.empty_cache`
are monkey-patched to their XLA equivalents at the top of this script.
This avoids touching author source files. Every hot path in
profiler/layers/main.py, profiler/common/timer.py, and models/llama.py runs
as published.

Run from inside the llm_profile/ directory:
    cd ~/llm_profile
    python3 run_layers_xla.py [args...]
"""
import os
import sys

# When invoked by absolute path, sys.path[0] is the script dir, not CWD.
# The author's `from models.llama import ...` and `from profiler.* import ...`
# require CWD (= LLMServingSim/llm_profile/) on sys.path.
sys.path.insert(0, os.getcwd())

# ---- XLA compatibility shim (must run before any profiler import) ----
import torch
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    def _xla_full_sync(*a, **k):
        """CUDA synchronize equivalent: queue + BLOCK until device idle.
        torch_xla.sync() alone is async (mark_step) — must follow with
        wait_device_ops() to match torch.cuda.synchronize() blocking semantics.
        """
        torch_xla.sync()
        xm.wait_device_ops()
    torch.cuda.synchronize = _xla_full_sync
    torch.cuda.empty_cache = lambda: None
    print("[xla_shim] patched torch.cuda.synchronize -> sync + wait_device_ops")
except Exception as e:
    print(f"[xla_shim] NOT patched (import failed: {e})")
    raise

# Default argv if none supplied (single-layer 1B smoke)
_DEFAULT_ARGS = [
    "--hardware", "TPU-v5e-1",
    "--model", "meta-llama/Llama-3.2-1B-Instruct",
    "--num-layers", "1",
    "--tp-size", "1",
    "--warmup", "5",
    "--repeat", "30",
    "--max-len", "64",
    "--device", "xla",
    "--profile-method", "perf_counter",
]
if len(sys.argv) == 1:
    sys.argv = [sys.argv[0]] + _DEFAULT_ARGS
    print(f"[xla_shim] no argv given, using defaults: {_DEFAULT_ARGS}")

# ---- Hand off to author's main ----
from profiler.layers.main import main
main()
