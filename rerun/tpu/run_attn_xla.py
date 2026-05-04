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

# 4) memory query shim (per-chip HBM). batch_sampling.py:163 uses mem_get_info to
#    decide which (chunk, kv, batch) combinations fit. Pick by --hardware:
#      TPU-v5e-1: 16 GB HBM
#      TPU-v6e-1: 32 GB HBM
_HBM_BY_HW = {
    "TPU-v5e-1": 16 * (1024 ** 3),
    "TPU-v6e-1": 32 * (1024 ** 3),
}
_hw_arg = "TPU-v6e-1"
for i, a in enumerate(sys.argv):
    if a == "--hardware" and i + 1 < len(sys.argv):
        _hw_arg = sys.argv[i + 1]
        break
_HBM_TOTAL = _HBM_BY_HW.get(_hw_arg, 32 * (1024 ** 3))
_HBM_FREE = _HBM_TOTAL - 2 * (1024 ** 3)  # 2 GB headroom for activations / runtime
torch.cuda.mem_get_info = lambda *a, **k: (_HBM_FREE, _HBM_TOTAL)

# 5) OOM class: never actually raised on TPU, but except clause references it
torch.cuda.OutOfMemoryError = RuntimeError  # safe superclass; nothing matches

# 6) AttentionInput.is_valid: the GPU-pipeline version rejects (is_prefill=True,
#    kv_cache_size=0), i.e. cold-start full prefill. BUT the committed
#    TPU-v6e-1 attention.csv in the repo DOES contain (chunk=N, kv=0, batch=1,
#    is_prefill=True) rows (lines starting "32,0,1,True,..."). This indicates
#    the authors' TPU profiler used a more permissive validator. We restore
#    cold-start prefills by patching is_valid to accept kv=0 when is_prefill.
from profiler.attention.attention_input import AttentionInput as _AI
_orig_is_valid = _AI.is_valid
def _tpu_is_valid(self, max_seq_len, max_model_len):
    if self.is_prefill:
        if self.batch_size != 1:
            return False
        if self.prefill_chunk_size == 0:
            return False
        if self.prefill_chunk_size + self.kv_cache_size > max_seq_len:
            return False
        # author GPU version rejects kv_cache_size==0 here; TPU version (per
        # committed perf_models/TPU-v6e-1/.../attention.csv) does NOT.
        return True
    else:
        if self.prefill_chunk_size > 0:
            return False
        if self.kv_cache_size == 0:
            return False
        if self.kv_cache_size > max_model_len:
            return False
        return True
_AI.is_valid = _tpu_is_valid

print("[xla_shim] flash_attn stubbed | cuda.synchronize -> sync+wait_device_ops")
print(f"[xla_shim] cuda.mem_get_info -> ({_V5E_FREE}, {_V5E_TOTAL}) bytes")

# Verify flash_attn stub is actually our file, not a real installation
import flash_attn
assert getattr(flash_attn, "__version__", "") == "xla-stub", \
    f"expected xla-stub flash_attn, got: {flash_attn.__file__}"
print(f"[xla_shim] flash_attn stub active: {flash_attn.__file__}")

# Default argv: v6e × Llama-3.1-8B, max-len 8192, batch=1.
# Covers ShareGPT (~400), CNN (~1.1k), Writing-prompts (~1.2k), arXiv (~3.5–8.5k).
# Author's v6e ipynb sweeps to 2048 only; we extend to 8192 because the simulator's
# DB-lookup mode (default) raises KeyError when (kv > 2048) is queried, and the
# sklearn predictor mode (random forest) clamps at the training boundary — both
# silently break long-context datasets without wider profiling.
# batch=1 keeps the (chunk × kv × batch) sweep tractable; larger batches at
# kv=8192 don't fit anyway (256 batches × 8192 kv × 128KB/tok ≈ 256 GB ≫ 32 GB HBM).
_DEFAULT_ARGS = [
    "--model", "meta-llama/Llama-3.1-8B",
    "--hardware", "TPU-v6e-1",
    "--max-len", "8192",
    "--tp-size", "1",
    "--min-batch-size", "1",
    "--max-batch-size", "1",
    "--warmup", "5",
    "--repeat", "20",
    "--device", "xla",
    "--profile-method", "perf_counter",
]
if len(sys.argv) == 1:
    sys.argv = [sys.argv[0]] + _DEFAULT_ARGS
    print(f"[xla_shim] no argv given, using smoke defaults")

# Hand off to author's main
from profiler.attention.main import main
main()
