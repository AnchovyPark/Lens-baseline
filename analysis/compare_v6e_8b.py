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
    fp = REAL_ROOT / f"Llama-3.1-8B-Instruct__bs{bs}_tp1_seq2048" / f"results_{ds}.csv"
    out = {}
    with open(fp, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "OK":
                continue
            out[int(row["id"])] = {
                "input_len": int(row["input_len"]),
                "output_len": int(row["output_len"]),
                "ttft_ms": float(row["ttft_ms"]),
                "tpot_ms": float(row["tpot_ms"]),
                "e2e_ms": float(row["e2e_ms"]),
            }
    return out


def _read_sim(ds: str, bs: int) -> dict[int, dict]:
    fp = SIM_ROOT / f"{ds}_bs{bs}.csv"
    if not fp.exists():
        return {}
    out = {}
    with open(fp, newline="") as f:
        for row in csv.DictReader(f):
            rid = int(row.get("request_id", row.get("id", -1)))
            ttft_ns = float(row.get("TTFT", row.get("ttft", 0)))
            tpot_ns = float(row.get("TPOT", row.get("tpot", 0)))
            lat_ns = float(row.get("latency", 0))
            out[rid] = {
                "ttft_ms": ttft_ns / 1e6,
                "tpot_ms": tpot_ns / 1e6,
                "e2e_ms": lat_ns / 1e6,
            }
    return out


def _stat(vals, fn):
    return fn(vals) if vals else float("nan")


def main():
    print(f"{'dataset':<18} {'bs':>3} | "
          f"{'TTFT real (med)':>16} {'TTFT sim (med)':>16} {'%err':>7} | "
          f"{'TPOT real (med)':>16} {'TPOT sim (med)':>16} {'%err':>7} | "
          f"{'n':>4}")
    print("-" * 130)

    for ds in DATASETS:
        for bs in BATCHES:
            real = _read_real(ds, bs)
            sim = _read_sim(ds, bs)
            common = sorted(set(real.keys()) & set(sim.keys()))
            if not common:
                print(f"{ds:<18} {bs:>3} | (no overlapping ids — sim file missing or empty)")
                continue
            ttft_real = [real[i]["ttft_ms"] for i in common]
            ttft_sim = [sim[i]["ttft_ms"] for i in common]
            tpot_real = [real[i]["tpot_ms"] for i in common]
            tpot_sim = [sim[i]["tpot_ms"] for i in common]

            t_r = statistics.median(ttft_real)
            t_s = statistics.median(ttft_sim)
            p_r = statistics.median(tpot_real)
            p_s = statistics.median(tpot_sim)
            t_err = (t_s - t_r) / t_r * 100.0 if t_r else 0.0
            p_err = (p_s - p_r) / p_r * 100.0 if p_r else 0.0
            print(f"{ds:<18} {bs:>3} | "
                  f"{t_r:>14.2f}ms {t_s:>14.2f}ms {t_err:>+6.1f}% | "
                  f"{p_r:>14.2f}ms {p_s:>14.2f}ms {p_err:>+6.1f}% | "
                  f"{len(common):>4}")


if __name__ == "__main__":
    main()
