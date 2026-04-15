"""Extract device-plane latencies from sparse30 Chrome trace files.

For each (input_len, kv_len) config we compute:
  - per-forward device time = sum of SyncTensorsGraph.N durations / count
  - total device time = sum of all pid=3 event durations / repeats

This gives us real compute-time scaling to compare against the notebook's
broken host-side numbers (which stayed flat ~220 µs per layer regardless
of input size).
"""
from __future__ import annotations
import gzip
import json
import glob
from collections import Counter
from pathlib import Path

TRACE_ROOT = Path("/Users/parkjuhyun/Desktop/baseline/v5e_rerun/xla_trace")
REPEAT = 30
WARMUP = 5

# sparse30 preset: PREFILL_STEP=256, DECODE_STEP=256, range(0, 2049, 256).
# Notebook builds pairs: (il if il<=0 else 1, 0) for each prefill, then
# (1, kl if kl<=0 else 1) for each decode. So file order matches:
PREFILLS = [(max(1, x), 0) for x in range(0, 2049, 256)]    # 9 configs
DECODES  = [(1, max(1, x)) for x in range(0, 2049, 256)]    # 9 configs
CONFIGS = PREFILLS + DECODES


def parse_one(trace_path: str) -> dict:
    """Return summary for one Chrome trace file."""
    with gzip.open(trace_path, "rt") as f:
        data = json.load(f)
    evs = data.get("traceEvents", [])

    # pid=3 is /device:TPU:0 (verified earlier via process_name metadata)
    tpu_x = [ev for ev in evs if ev.get("pid") == 3 and ev.get("ph") == "X"]

    sync_events = [ev for ev in tpu_x if ev.get("name", "").startswith("SyncTensorsGraph")]
    sync_durs = [ev.get("dur", 0) for ev in sync_events]

    # Group all device events by leading name prefix (fusion / copy-start / copy-done / SyncTensorsGraph)
    by_kind: Counter[str] = Counter()
    kind_dur_sum: dict[str, float] = {}
    for ev in tpu_x:
        name = ev.get("name", "")
        kind = name.split(".")[0].split("(")[0] if "." in name or "(" in name else name
        kind = kind[:32]
        by_kind[kind] += 1
        kind_dur_sum[kind] = kind_dur_sum.get(kind, 0.0) + ev.get("dur", 0)

    total_device_us = sum(ev.get("dur", 0) for ev in tpu_x)
    return {
        "tpu_event_count": len(tpu_x),
        "sync_count": len(sync_events),
        "sync_sum_us": sum(sync_durs),
        "sync_per_forward_us": sum(sync_durs) / max(1, len(sync_events)),
        "total_device_us": total_device_us,
        "kind_counts": dict(by_kind),
        "kind_durs_us": kind_dur_sum,
    }


def main() -> None:
    files = sorted(glob.glob(f"{TRACE_ROOT}/plugins/profile/*/*.trace.json.gz"))
    assert len(files) == len(CONFIGS), f"got {len(files)} files for {len(CONFIGS)} configs"

    print(f"{'idx':>3}  {'input':>6} {'kv_cache':>8}  "
          f"{'sync#':>5} {'per_fwd(µs)':>12} {'total_dev(µs)':>14} "
          f"{'events':>7}")
    print("-" * 72)

    rows = []
    for idx, (fp, (il, kv)) in enumerate(zip(files, CONFIGS)):
        r = parse_one(fp)
        rows.append((il, kv, r))
        print(f"{idx:3d}  {il:6d} {kv:8d}  "
              f"{r['sync_count']:5d} "
              f"{r['sync_per_forward_us']:12.1f} "
              f"{r['total_device_us']:14.1f} "
              f"{r['tpu_event_count']:7d}")

    # Show event-kind distribution for a representative config
    print("\nEvent-kind distribution at input=2048, kv=0:")
    idx_2048 = CONFIGS.index((2048, 0))
    r = rows[idx_2048][2]
    for kind, cnt in sorted(r["kind_counts"].items(), key=lambda x: -x[1]):
        tot = r["kind_durs_us"][kind]
        print(f"  {kind:32s}  count={cnt:4d}  total={tot:10.1f} µs  "
              f"avg={tot/max(1,cnt):8.2f} µs")

    print("\nSame for smallest prefill (input=1, kv=0):")
    idx_smallest = CONFIGS.index((1, 0))
    r = rows[idx_smallest][2]
    for kind, cnt in sorted(r["kind_counts"].items(), key=lambda x: -x[1])[:12]:
        tot = r["kind_durs_us"][kind]
        print(f"  {kind:32s}  count={cnt:4d}  total={tot:10.1f} µs  "
              f"avg={tot/max(1,cnt):8.2f} µs")


if __name__ == "__main__":
    main()
