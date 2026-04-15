#!/usr/bin/env python3
"""Sparse sweep variant of run_layers_xla.py.

Calls authors' `run_profile()` directly instead of `main()`, so we can pass
custom `input_lengths` (sparse) instead of the dense `range(1, max_len+1)`
the authors hardcode. Same instrumentation, same Timer, same Llama model
code — only difference is the set of input lengths sampled.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

# XLA shim (must run before any author module imports)
import torch
import torch_xla
import torch_xla.core.xla_model as xm
def _xla_full_sync(*a, **k):
    """CUDA synchronize equivalent: queue + BLOCK until device idle."""
    torch_xla.sync()           # mark_step: queue current graph
    xm.wait_device_ops()       # block until all queued ops actually complete
torch.cuda.synchronize = _xla_full_sync
torch.cuda.empty_cache = lambda: None
print("[xla_shim] patched torch.cuda.synchronize -> sync + wait_device_ops")

from profiler.layers.main import run_profile

# Sparse: covers small (overhead-dominated) and large (compute-dominated)
INPUT_LENGTHS = [1, 16, 64, 128, 256, 512, 1024, 2048]

run_profile(
    hardware="TPU-v5e-1",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    num_layers=1,
    input_lengths=INPUT_LENGTHS,
    is_prefill=True,
    tp_size=1,
    device="xla",
    warmup=5,
    repeat=20,
    profile_method="perf_counter",
    csv_append=False,
    verbose=True,
)
