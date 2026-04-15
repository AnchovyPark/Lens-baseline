#!/usr/bin/env python3
"""Convert GPU-schema attention.csv (from author's profile_attn.sh output)
to TPU-schema expected by `predictor/main.py:67-87` TPU branch.

The author's GPU pipeline writes:
    time_stats.attn_prefill.{min,max,mean,median,std}
    time_stats.attn_decode.{min,max,mean,median,std}
    (values in MILLISECONDS — from Timer's perf_counter *1e3)

Author's TPU predictor expects the TPU-schema:
    mean_ns, p50_ns, p90_ns, max_ns    (values in NANOSECONDS)

This aligns with the hmchoi TODO in build_sklearn_predictor_and_pred.py:33:
    "for TPU profiled result, we don't need to scale it"
i.e. TPU attention.csv ships ns directly, skipping the *1e6 scale step the GPU
pipeline applies during predict_and_save.

So the transform is: for each row, pick the non-NaN prefill or decode
time_stats.*.{stat} (stat in {mean, median, max}), multiply by 1e6 (ms → ns),
and write TPU schema. Author's predictor only trains on `p50_ns` for TPU;
the other stat columns (mean_ns, p90_ns, max_ns) are included for schema
parity with the committed TPU-v6e-1 attention.csv.
"""
from __future__ import annotations
import csv
import math
import sys

MS_TO_NS = 1_000_000


def _to_float(x: str):
    if x == "" or x is None:
        return float("nan")
    try:
        v = float(x)
    except ValueError:
        return float("nan")
    return v if not math.isnan(v) else float("nan")


def _pick(row: dict, stat: str) -> float:
    """Return whichever of prefill/decode stat is populated (non-NaN)."""
    p = _to_float(row.get(f"time_stats.attn_prefill.{stat}", ""))
    d = _to_float(row.get(f"time_stats.attn_decode.{stat}", ""))
    if not math.isnan(p):
        return p
    if not math.isnan(d):
        return d
    return float("nan")


def convert(src: str, dst: str) -> None:
    out_rows = []
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean_ms = _pick(row, "mean")
            med_ms = _pick(row, "median")
            max_ms = _pick(row, "max")
            if math.isnan(med_ms):
                continue  # SDPA OOM / skipped — drop this row
            out_rows.append({
                "prefill_chunk_size": int(row["prefill_chunk_size"]),
                "kv_cache_size": int(row["kv_cache_size"]),
                "batch_size": int(row["batch_size"]),
                "is_prefill": row["is_prefill"],
                "mean_ns": mean_ms * MS_TO_NS if not math.isnan(mean_ms) else "",
                "p50_ns": med_ms * MS_TO_NS,
                # Timer only records {min,max,mean,median,std}. p90 is not directly
                # available; we use max as a conservative upper-bound proxy.
                "p90_ns": max_ms * MS_TO_NS if not math.isnan(max_ms) else "",
                "max_ns": max_ms * MS_TO_NS if not math.isnan(max_ms) else "",
            })

    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prefill_chunk_size", "kv_cache_size", "batch_size", "is_prefill",
                "mean_ns", "p50_ns", "p90_ns", "max_ns",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    n_prefill = sum(1 for r in out_rows if r["is_prefill"] == "True")
    n_decode = len(out_rows) - n_prefill
    p50_vals = [r["p50_ns"] for r in out_rows]
    print(f"wrote {len(out_rows)} rows to {dst}")
    print(f"prefill rows: {n_prefill}, decode rows: {n_decode}")
    print(f"p50_ns range: [{min(p50_vals):.0f}, {max(p50_vals):.0f}] ns")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: convert_attn_to_tpu_schema.py <input.csv> <output.csv>")
    convert(sys.argv[1], sys.argv[2])
