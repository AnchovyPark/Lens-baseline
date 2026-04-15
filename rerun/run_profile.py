#!/usr/bin/env python3
"""Single driver for Phase 3 (smoke) / Phase 4 (sparse) / Phase 5 (dense) runs.

Usage:
    python3 run_profile.py smoke    # Phase 3: 1 config, ~2 min
    python3 run_profile.py sparse   # Phase 4: 18 configs, ~15 min
    python3 run_profile.py dense    # Phase 5: 256 configs, ~60 min

Output: ~/perf_models_raw/<HARDWARE>/<sanitized_model>.csv
(The notebook's own schema; conversion to LLMServingSim's per-tp{N}/layers.csv
format happens in a separate post-processing step on the local Mac.)
"""
from __future__ import annotations
import os
import sys

# torch_xla 2.x in-process profiler server — must listen before xp.start_trace runs.
os.environ.setdefault("XLA_PROFILER_PORT", "9012")

from profiler_core import run_profile

# ---- fixed across all presets ----
HARDWARE = "TPU-v5e-1"
MODEL = os.environ.get("PROFILE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
DEVICE = "xla"
NUM_LAYERS = 1  # profile 1 decoder layer; notebook scales by original_num_layers
OUTPUT_DIR = os.path.expanduser("~/perf_models_raw")

PRESETS = {
    "smoke":  dict(PREFILL_MAX=1,    PREFILL_STEP=1,   DECODE_MAX=1,    DECODE_STEP=1,   WARMUP=3,  REPEAT=5),
    "sparse": dict(PREFILL_MAX=2048, PREFILL_STEP=256, DECODE_MAX=2048, DECODE_STEP=256, WARMUP=3,  REPEAT=5),
    # sparse30: same coverage as sparse but REPEAT=30 so small configs still
    # produce device events within the trace window (REPEAT=5 missed them).
    "sparse30": dict(PREFILL_MAX=2048, PREFILL_STEP=256, DECODE_MAX=2048, DECODE_STEP=256, WARMUP=5, REPEAT=30),
    "dense":  dict(PREFILL_MAX=2048, PREFILL_STEP=16,  DECODE_MAX=2048, DECODE_STEP=16,  WARMUP=10, REPEAT=30),
}


def main(preset: str) -> None:
    if preset not in PRESETS:
        sys.exit(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    p = PRESETS[preset]

    input_prefill = list(range(0, p["PREFILL_MAX"] + 1, p["PREFILL_STEP"]))
    kv_decode = list(range(0, p["DECODE_MAX"] + 1, p["DECODE_STEP"]))
    print(f"[preset={preset}] prefill configs: {len(input_prefill)}, "
          f"decode configs: {len(kv_decode)}, "
          f"warmup={p['WARMUP']}, repeat={p['REPEAT']}")

    # Hold profiler-server handle alive for the whole run (GC'd handle = closed socket)
    import torch_xla.debug.profiler as xp
    _server_handle = None
    try:
        _server_handle = xp.start_server(int(os.environ["XLA_PROFILER_PORT"]))
        print(f"profiler server bound on port {os.environ['XLA_PROFILER_PORT']}")
    except Exception as e:
        print(f"[warn] xp.start_server failed: {e}")

    out_csv = run_profile(
        hardware=HARDWARE,
        model_name=MODEL,
        num_layers=NUM_LAYERS,
        input_lengths=tuple(input_prefill),
        kv_cache_lengths=tuple(kv_decode),
        device_flag=DEVICE,
        warmup=p["WARMUP"],
        repeat=p["REPEAT"],
        csv_append=False,
        verbose=(preset == "smoke"),   # only spam stdout on smoke
        out_dir=OUTPUT_DIR,
        hf_token=os.environ.get("HF_TOKEN", ""),
        progress=True,
        flush_every=20,
    )
    print(f"\n[DONE] raw csv: {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
