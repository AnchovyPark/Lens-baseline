#!/usr/bin/env python3
"""
Dry-run: profile Llama-3.2-1B-Instruct on TPU v5e.

Produces a sparse sweep CSV at:
  ~/perf_models/TPU-v5e-1/meta-llama_Llama-3.2-1B-Instruct.csv

This is smoke-test territory: PREFILL_STEP and DECODE_STEP are coarse so
the run completes in minutes, not hours.
"""
import os
import sys

# torch_xla profiler server (notebook cell 8 pattern)
os.environ.setdefault("XLA_PROFILER_PORT", "9012")

from profiler_core import run_profile

HARDWARE      = "TPU-v5e-1"
MODEL         = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LAYERS    = 1           # profile one decoder layer, extrapolate
DEVICE        = "xla"
WARMUP        = 3
REPEAT        = 5

# Coarse sweep for dry run; dense sweep (step=1) comes later.
PREFILL_MAX   = 2048
PREFILL_STEP  = 256
DECODE_MAX    = 2048
DECODE_STEP   = 256

OUTPUT_DIR    = os.path.expanduser("~/perf_models_raw")

input_prefill = list(range(0, PREFILL_MAX + 1, PREFILL_STEP))
kv_decode     = list(range(0, DECODE_MAX + 1, DECODE_STEP))
print(f"prefill configs: {len(input_prefill)}  decode configs: {len(kv_decode)}")
print(f"output dir: {OUTPUT_DIR}/{HARDWARE}/")

# HF token: prefer cached login; fall back to env var if present.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Start profiler server
import time
import torch_xla.debug.profiler as xp
# Hold server reference alive for the whole run — else GC closes the socket
_PROF_SERVER = None
try:
    _PROF_SERVER = xp.start_server(int(os.environ["XLA_PROFILER_PORT"]))
    print(f"profiler server started on port {os.environ['XLA_PROFILER_PORT']}")
except Exception as e:
    print(f"[warn] xp.start_server failed: {e}")
time.sleep(1.0)  # let gRPC socket bind

out_csv = run_profile(
    hardware=HARDWARE,
    model_name=MODEL,
    num_layers=NUM_LAYERS,
    input_lengths=tuple(input_prefill),
    kv_cache_lengths=tuple(kv_decode),
    device_flag=DEVICE,
    warmup=WARMUP,
    repeat=REPEAT,
    csv_append=False,
    verbose=True,
    out_dir=OUTPUT_DIR,
    hf_token=HF_TOKEN,
    progress=True,
    flush_every=20,
)

print(f"\n[DONE] raw csv: {out_csv}")
