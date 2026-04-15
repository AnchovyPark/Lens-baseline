#!/usr/bin/env python3
"""TPU adapter for author's `profile_attn.sh` / `profiler.attention.main`.

Follows the same philosophy as `run_layers_xla.py`:
  - no author source file is edited
  - monkey-patches applied before any author module imports
  - author's `--profile-method perf_counter` path used (device-agnostic)

Extra patches needed for attention (vs layers):
  1. `flash_attn` package stub prepended to sys.path so the hard import succeeds;
     see `flash_attn_stub/flash_attn/__init__.py` for the SDPA reimplementation.
  2. `torch.cuda.mem_get_info()` faked to v5e's 16 GB HBM (used by
     `profiler/attention/batch_sampling.py:163` for block-budget calc).
  3. `torch.cuda.OutOfMemoryError` remapped to a benign exception class
     since the except clause never fires on TPU (we stay well under budget).

Run from inside `llm_profile/` on the VM:
    cd ~/Lens-baseline/LLMServingSim/llm_profile
    python3 ~/Lens-baseline/rerun/tpu/run_attn_xla.py
"""
import os
import sys

# 1) flash_attn stub ahead of real flash_attn (which doesn't exist on TPU anyway)
STUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn_stub")
if STUB_DIR not in sys.path:
    sys.path.insert(0, STUB_DIR)

# 2) author module imports (profiler.*, models.*) resolve from CWD = llm_profile/
sys.path.insert(0, os.getcwd())

# 3) XLA sync / empty_cache shim
import torch
import torch_xla
import torch_xla.core.xla_model as xm

def _xla_full_sync(*a, **k):
    torch_xla.sync()
    xm.wait_device_ops()

torch.cuda.synchronize = _xla_full_sync
torch.cuda.empty_cache = lambda: None

# 4) memory query shim (v5e HBM = 16 GiB; report 14 GiB free to leave headroom)
_V5E_TOTAL = 16 * (1024 ** 3)
_V5E_FREE = 14 * (1024 ** 3)
torch.cuda.mem_get_info = lambda *a, **k: (_V5E_FREE, _V5E_TOTAL)

# 5) OOM class: never actually raised on TPU, but except clause references it
torch.cuda.OutOfMemoryError = RuntimeError  # safe superclass; nothing matches

print("[xla_shim] flash_attn stubbed | cuda.synchronize -> sync+wait_device_ops")
print(f"[xla_shim] cuda.mem_get_info -> ({_V5E_FREE}, {_V5E_TOTAL}) bytes")

# Verify flash_attn stub is actually our file, not a real installation
import flash_attn
assert getattr(flash_attn, "__version__", "") == "xla-stub", \
    f"expected xla-stub flash_attn, got: {flash_attn.__file__}"
print(f"[xla_shim] flash_attn stub active: {flash_attn.__file__}")

# Default argv: small attention smoke (--max-len=128, batch=1-4)
_DEFAULT_ARGS = [
    "--model", "meta-llama/Llama-3.2-1B-Instruct",
    "--hardware", "TPU-v5e-1",
    "--max-len", "128",
    "--tp-size", "1",
    "--min-batch-size", "1",
    "--max-batch-size", "4",
    "--warmup", "3",
    "--repeat", "5",
    "--device", "xla",
    "--profile-method", "perf_counter",
]
if len(sys.argv) == 1:
    sys.argv = [sys.argv[0]] + _DEFAULT_ARGS
    print(f"[xla_shim] no argv given, using smoke defaults")

# Hand off to author's main
from profiler.attention.main import main
main()
