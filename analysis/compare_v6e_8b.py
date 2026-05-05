#!/usr/bin/env python3
"""Compare LLMServingSim v6e × 8B simulator output to real vLLM-TPU measurements.

Real measurements: inference_result/Llama-3.1-8B-Instruct__bs{1,2,4}_tp1_seq2048/results_*.csv
                   columns: id,input_len,output_len,status,ttft_ms,tpot_ms,e2e_ms,...
Simulator output:  LLMServingSim_ispass26/output/v6e_8b_baseline/{ds}_bs{N}.csv
                   columns (per LLMServingSim v2): instance_id, request_id, model,
                   input, output, arrival, end_time, latency, queuing_delay, TTFT, TPOT, ITL

Joins by request id (0..49 in both files; vLLM script and JSONL preserve order).
Reports: median TTFT/TPOT for both, signed % error per metric per (dataset, batch).

Usage:
    python3 analysis/compare_v6e_8b.py
"""
from __future__ import annotations
import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = ROOT / "inference_result"
SIM_ROOT = ROOT / "LLMServingSim_ispass26" / "output" / "v6e_8b_baseline"

DATASETS = ("sharegpt", "cnn", "writing_prompts")
BATCHES = (1, 2, 4)


def _read_real(ds: str, bs: int) -> dict[int, dict]:
    """bs=1 has per-request ttft_ms/tpot_ms/e2e_ms.
    bs={2,4} only have per-batch batch_e2e_ms (different schema).
    Returns dict keyed by request id (bs=1) or batch run_id (bs>=2).
    """
    fp = REAL_ROOT / f"Llama-3.1-8B-Instruct__bs{bs}_tp1_seq2048" / f"results_{ds}.csv"
    out = {}
    with open(fp, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "OK":
                continue
            if "ttft_ms" in row:  # bs=1 per-request
                out[int(row["id"])] = {
                    "kind": "per_request",
                    "input_len": int(row["input_len"]),
                    "output_len": int(row["output_len"]),
                    "ttft_ms": float(row["ttft_ms"]),
                    "tpot_ms": float(row["tpot_ms"]),
                    "e2e_ms": float(row["e2e_ms"]),
                }
            else:  # bs>=2 per-batch
                out[int(row["run_id"])] = {
                    "kind": "per_batch",
                    "batch_e2e_ms": float(row["batch_e2e_ms"]),
                }
    return out


def _read_sim(ds: str, bs: int) -> dict[int, dict]:
    """Sim CSV columns (LLMServingSim v2): 'instance id','request id',model,input,
    output,arrival,end_time,latency,queuing_delay,TTFT,TPOT,ITL.

    Sim's TTFT field = queuing_delay + prefill_compute. Real measurement's
    ttft_ms is per-request prefill compute only (queue wait excluded). To
    compare apples-to-apples, return ttft_ms = (TTFT - queuing_delay)/1e6.
    """
    fp = SIM_ROOT / f"{ds}_bs{bs}.csv"
    if not fp.exists():
        return {}
    out = {}
    with open(fp, newline="") as f:
        for row in csv.DictReader(f):
            rid = int(row.get("request id", row.get("request_id", row.get("id", -1))))
            ttft_ns = float(row.get("TTFT", 0))
            tpot_ns = float(row.get("TPOT", 0))
            qd_ns = float(row.get("queuing_delay", 0))
            lat_ns = float(row.get("latency", 0))
            out[rid] = {
                "ttft_ms": (ttft_ns - qd_ns) / 1e6,
                "ttft_inc_queue_ms": ttft_ns / 1e6,
                "tpot_ms": tpot_ns / 1e6,
                "e2e_ms": lat_ns / 1e6,
                "queuing_delay_ms": qd_ns / 1e6,
            }
    return out


def _stat(vals, fn):
    return fn(vals) if vals else float("nan")


def main():
    print("=== bs=1: per-request TTFT/TPOT (sim TTFT computed as TTFT_field − queuing_delay) ===")
    print(f"{'dataset':<18} {'bs':>3} | "
          f"{'TTFT real':>10} {'TTFT sim':>10} {'%err':>7} | "
          f"{'TPOT real':>10} {'TPOT sim':>10} {'%err':>7} | "
          f"{'n':>4}")
    print("-" * 90)
    for ds in DATASETS:
        bs = 1
        real = _read_real(ds, bs)
        sim = _read_sim(ds, bs)
        if not real or not sim or next(iter(real.values()))["kind"] != "per_request":
            print(f"{ds:<18} {bs:>3} | missing data")
            continue
        common = sorted(set(real.keys()) & set(sim.keys()))
        ttft_r = [real[i]["ttft_ms"] for i in common]
        ttft_s = [sim[i]["ttft_ms"] for i in common]
        tpot_r = [real[i]["tpot_ms"] for i in common]
        tpot_s = [sim[i]["tpot_ms"] for i in common]
        tr, ts = statistics.median(ttft_r), statistics.median(ttft_s)
        pr, ps = statistics.median(tpot_r), statistics.median(tpot_s)
        terr = (ts - tr) / tr * 100.0 if tr else 0.0
        perr = (ps - pr) / pr * 100.0 if pr else 0.0
        print(f"{ds:<18} {bs:>3} | "
              f"{tr:>8.2f}ms {ts:>8.2f}ms {terr:>+6.1f}% | "
              f"{pr:>8.2f}ms {ps:>8.2f}ms {perr:>+6.1f}% | "
              f"{len(common):>4}")

    print()
    print("=== bs={2,4}: total wall-time comparison (real has only per-batch e2e) ===")
    print(f"{'dataset':<18} {'bs':>3} | "
          f"{'real total (s)':>14} {'sim total (s)':>14} {'%err':>7} | "
          f"{'real med batch':>14} {'sim med req e2e':>16} | "
          f"{'n_real':>6} {'n_sim':>5}")
    print("-" * 110)
    for ds in DATASETS:
        for bs in (2, 4):
            real = _read_real(ds, bs)
            sim = _read_sim(ds, bs)
            if not real or not sim:
                print(f"{ds:<18} {bs:>3} | missing data")
                continue
            real_batch_e2e = [v["batch_e2e_ms"] for v in real.values() if v["kind"] == "per_batch"]
            sim_e2e = [v["e2e_ms"] for v in sim.values()]
            real_total_s = sum(real_batch_e2e) / 1000.0
            # sim runs all 50 reqs in parallel-with-batching; total wall = max(end_time) ≈ max(arrival+latency).
            # Since arrivals are burst at ns≈0, max(latency) ≈ max(end_time) = total wall.
            sim_total_s = max(sim_e2e) / 1000.0 if sim_e2e else 0.0
            err = (sim_total_s - real_total_s) / real_total_s * 100.0 if real_total_s else 0.0
            real_med = statistics.median(real_batch_e2e)
            sim_med = statistics.median(sim_e2e)
            print(f"{ds:<18} {bs:>3} | "
                  f"{real_total_s:>12.2f}s {sim_total_s:>12.2f}s {err:>+6.1f}% | "
                  f"{real_med:>12.2f}ms {sim_med:>14.2f}ms | "
                  f"{len(real_batch_e2e):>6} {len(sim_e2e):>5}")


if __name__ == "__main__":
    main()
