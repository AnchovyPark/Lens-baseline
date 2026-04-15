"""Regenerate a clean profiler_core.py from the reference notebook.

Unlike the earlier `/tmp/v5e_prof/profiler_core.py`, this version uses the
notebook's original `xp.start_trace` / `xp.stop_trace` / Chrome-JSON trace flow
(intended for torch_xla ≤ 2.4) and adds graceful fallbacks for the
`torch_xla.sync()` / `torch_xla.device()` shorthand API in case the target
env only has the older `xm.mark_step()` / `xm.xla_device()` forms.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(
    "/Users/parkjuhyun/Desktop/baseline/LLMServingSim/llm_profile/"
    "perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb"
)
OUT_PATH = Path("/Users/parkjuhyun/Desktop/baseline/rerun/profiler_core.py")

HEADER = '''"""Profiler core (auto-generated from LLMServingSim/.../llm_profiler_tpu.ipynb
cells 1 + 4 + 5 + 6). Intended for torch_xla 2.4 which still ships xp.start_trace
and writes Chrome trace JSON (so the notebook's original parser works as-is).

A thin compatibility shim is added at top so the same file can also run against
older torch_xla (2.3) where torch_xla.sync() / torch_xla.device() didn't exist
yet — those fall back to xm.mark_step() / xm.xla_device().
"""
import os, sys, time, threading
import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    from torch_xla import runtime as xr
    _XLA_AVAILABLE = True
except Exception as e:
    print("XLA import error:", e)
    _XLA_AVAILABLE = False

# Shim: torch_xla.sync / torch_xla.device were added in 2.4 as aliases.
# On 2.3 fall back to the older names.
if _XLA_AVAILABLE:
    if not hasattr(torch_xla, "sync"):
        torch_xla.sync = xm.mark_step   # type: ignore[attr-defined]
    if not hasattr(torch_xla, "device"):
        torch_xla.device = xm.xla_device  # type: ignore[attr-defined]

'''


def main() -> None:
    nb = json.load(open(NB_PATH))
    parts = [HEADER]
    # cell 4: profiler core
    parts.append(''.join(nb['cells'][4]['source']))
    parts.append("\n")
    # cell 5: validation helpers
    parts.append(''.join(nb['cells'][5]['source']))
    parts.append("\n")
    # cell 6: scaling factor patch
    parts.append(''.join(nb['cells'][6]['source']))
    parts.append("\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("".join(parts))

    n_lines = sum(1 for _ in open(OUT_PATH))
    print(f"wrote {OUT_PATH} ({n_lines} lines)")


if __name__ == "__main__":
    main()
