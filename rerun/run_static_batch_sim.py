#!/usr/bin/env python3
"""Replay each real static-batching run as a separate LLMServingSim invocation.

Real measurements at bs={2,4} use STATIC batching: each row of results_*.csv is
one batch (group of N requests submitted together, runs until all finish before
next batch starts). LLMServingSim's --max-batch=N flag implements CONTINUOUS
batching (max N concurrent, slot freed = next admitted). Comparing static real
to continuous sim is apples-to-oranges (the prior comparator showed -65~-84%
"error" entirely from regime mismatch).

This script reproduces the static-batching regime in sim by spawning ONE sim
invocation PER REAL BATCH:
  - For each row in real results_<ds>.csv (run_id=0..N-1):
    - Build a mini JSONL of just that batch's K requests (input_lens[i], output_lens[i])
    - Run main.py with --num-req=K --max-batch=K --dataset <mini.jsonl>
    - The sim's wall-time for that mini run = max(end_time) across K requests
  - Sum all per-batch sim wall-times → static-mode sim total
  - Compare to sum(real.batch_e2e_ms)

Output:
  <out_root>/<ds>_bs<bs>/
    batch_NNN.csv               per-batch sim output (5 cols × K rows)
    batch_NNN_input.jsonl       per-batch sim input (K rows)
    summary.json                aggregated comparison

Usage (from inside LLMServingSim_ispass26/):
  python3 ../rerun/run_static_batch_sim.py \\
    --real-root ../inference_result \\
    --cluster-config cluster_config/single_node_tpuv6e_8b.json \\
    --out-root output/v6e_8b_static \\
    --datasets sharegpt cnn writing_prompts \\
    --batch-sizes 2 4

Wallclock estimate per (dataset, bs): ~50 batches × 1~5 min/batch ≈ 1~4 hours.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _parse_lens(s: str) -> list[int]:
    """results_*.csv stores input_lens / output_lens as stringified Python lists."""
    return list(map(int, ast.literal_eval(s)))


def _build_mini_jsonl(
    out_path: Path,
    input_lens: list[int],
    output_lens: list[int],
    vocab_size: int = 128256,
    seed: int = 42,
) -> None:
    """Write one JSONL line per request. arrival_time_ns=0 so all start together."""
    rng = random.Random(seed)
    with open(out_path, "w") as f:
        for il, ol in zip(input_lens, output_lens):
            rec = {
                "input_toks": il,
                "output_toks": ol,
                "arrival_time_ns": 0.0,
                "input_tok_ids": [rng.randrange(vocab_size) for _ in range(il)],
                "output_tok_ids": [rng.randrange(vocab_size) for _ in range(ol)],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_sim(
    cluster_config: str,
    jsonl_path: Path,
    output_csv: Path,
    num_req: int,
    max_batch: int,
    max_num_batched_tokens: int = 2048,
    log_level: str = "WARNING",
) -> float:
    """Invoke main.py and return sim wall-time in ms (max end_time across reqs)."""
    cmd = [
        sys.executable, "main.py",
        "--cluster-config", cluster_config,
        "--dataset", str(jsonl_path),
        "--output", str(output_csv),
        "--max-batch", str(max_batch),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--num-req", str(num_req),
        "--log-level", log_level,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    end_times_ns: list[float] = []
    with open(output_csv) as f:
        for row in csv.DictReader(f):
            end_times_ns.append(float(row["end_time"]))
    return max(end_times_ns) / 1e6  # ns → ms


def _read_real_batches(results_csv: Path) -> list[dict]:
    """Parse static-batching real CSV. Each row is one batch."""
    out = []
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "OK":
                continue
            out.append({
                "run_id": int(row["run_id"]),
                "batch_size": int(row["batch_size"]),
                "input_lens": _parse_lens(row["input_lens"]),
                "output_lens": _parse_lens(row["output_lens"]),
                "batch_e2e_ms": float(row["batch_e2e_ms"]),
            })
    return out


def replay_static_batching(
    real_root: Path,
    cluster_config: str,
    out_root: Path,
    dataset: str,
    bs: int,
    max_num_batched_tokens: int = 2048,
) -> dict:
    real_csv = real_root / f"Llama-3.1-8B-Instruct__bs{bs}_tp1_seq2048" / f"results_{dataset}.csv"
    if not real_csv.exists():
        return {"error": f"missing {real_csv}"}

    batches = _read_real_batches(real_csv)
    out_dir = out_root / f"{dataset}_bs{bs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_batch = []
    t_start = time.time()
    for i, b in enumerate(batches):
        jsonl = out_dir / f"batch_{i:03d}_input.jsonl"
        sim_csv = out_dir / f"batch_{i:03d}.csv"
        _build_mini_jsonl(jsonl, b["input_lens"], b["output_lens"])
        try:
            sim_ms = _run_sim(
                cluster_config=cluster_config,
                jsonl_path=jsonl,
                output_csv=sim_csv,
                num_req=len(b["input_lens"]),
                max_batch=bs,
                max_num_batched_tokens=max_num_batched_tokens,
            )
        except subprocess.CalledProcessError as e:
            sim_ms = float("nan")
            print(f"  [batch {i}] FAILED: {e.stderr.decode()[:300]}", file=sys.stderr)

        per_batch.append({
            "run_id": b["run_id"],
            "input_lens": b["input_lens"],
            "output_lens": b["output_lens"],
            "real_batch_e2e_ms": b["batch_e2e_ms"],
            "sim_batch_e2e_ms": sim_ms,
            "abs_err_ms": (sim_ms - b["batch_e2e_ms"]) if sim_ms == sim_ms else None,
            "pct_err": ((sim_ms - b["batch_e2e_ms"]) / b["batch_e2e_ms"] * 100.0) if (sim_ms == sim_ms and b["batch_e2e_ms"] > 0) else None,
        })

        elapsed = time.time() - t_start
        print(f"  [{dataset} bs{bs}] batch {i+1}/{len(batches)}  "
              f"real={b['batch_e2e_ms']:.1f}ms  sim={sim_ms:.1f}ms  "
              f"elapsed={elapsed:.0f}s")

    valid = [p for p in per_batch if p["sim_batch_e2e_ms"] == p["sim_batch_e2e_ms"]]
    real_total_s = sum(p["real_batch_e2e_ms"] for p in valid) / 1000.0
    sim_total_s = sum(p["sim_batch_e2e_ms"] for p in valid) / 1000.0

    summary = {
        "dataset": dataset,
        "bs": bs,
        "n_batches_real": len(batches),
        "n_batches_sim_ok": len(valid),
        "real_total_s": real_total_s,
        "sim_total_s": sim_total_s,
        "total_pct_err": (sim_total_s - real_total_s) / real_total_s * 100.0 if real_total_s else None,
        "per_batch_pct_err": {
            "median": statistics.median([p["pct_err"] for p in valid if p["pct_err"] is not None]) if valid else None,
            "mean": statistics.mean([p["pct_err"] for p in valid if p["pct_err"] is not None]) if valid else None,
            "stdev": statistics.stdev([p["pct_err"] for p in valid if p["pct_err"] is not None]) if len(valid) > 1 else None,
        },
        "per_batch": per_batch,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-root", type=Path, required=True,
                    help="path to inference_result/")
    ap.add_argument("--cluster-config", required=True,
                    help="cluster config relative to LLMServingSim cwd, e.g. cluster_config/single_node_tpuv6e_8b.json")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="output dir, e.g. output/v6e_8b_static")
    ap.add_argument("--datasets", nargs="+", default=["sharegpt", "cnn", "writing_prompts"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    overall = {}
    for ds in args.datasets:
        for bs in args.batch_sizes:
            print(f"\n========== {ds} bs={bs} ==========")
            summary = replay_static_batching(
                real_root=args.real_root,
                cluster_config=args.cluster_config,
                out_root=args.out_root,
                dataset=ds,
                bs=bs,
                max_num_batched_tokens=args.max_num_batched_tokens,
            )
            overall[f"{ds}_bs{bs}"] = {
                "n_batches_sim_ok": summary.get("n_batches_sim_ok"),
                "real_total_s": summary.get("real_total_s"),
                "sim_total_s": summary.get("sim_total_s"),
                "total_pct_err": summary.get("total_pct_err"),
                "per_batch_median_pct_err": summary.get("per_batch_pct_err", {}).get("median"),
            }

    with open(args.out_root / "overall_summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'config':<28} {'#batch':>7} {'real total (s)':>16} {'sim total (s)':>16} {'total %err':>12} {'med %err':>10}")
    print("-" * 100)
    for k, v in overall.items():
        print(f"{k:<28} {v.get('n_batches_sim_ok',0):>7} "
              f"{v.get('real_total_s',0):>14.2f}s "
              f"{v.get('sim_total_s',0):>14.2f}s "
              f"{v.get('total_pct_err',0):>+11.1f}% "
              f"{v.get('per_batch_median_pct_err',0):>+9.1f}%")


if __name__ == "__main__":
    main()
