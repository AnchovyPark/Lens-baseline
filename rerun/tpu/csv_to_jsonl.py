#!/usr/bin/env python3
"""Convert pre-sampled (id, input_len, output_len) CSV → LLMServingSim .jsonl.

LLMServingSim's request format (see dataset/sharegpt_req100_rate10_llama.jsonl):
    {"input_toks": int, "output_toks": int, "arrival_time_ns": float,
     "input_tok_ids": [int...], "output_tok_ids": [int...]}

For --max-batch=1 + no prefix caching runs, the simulator only reads token COUNTS
(`input_toks`, `output_toks`). The token IDs and arrival times don't affect the
TTFT/TPOT measurements when batch=1 (every request executes in isolation).
We still populate them so the file matches the schema; arrival_time_ns is laid
out via Poisson at a fixed rate (default 10 req/s, mirroring author's
sharegpt_parser.py:23).

Usage:
    csv_to_jsonl.py <input.csv> <output.jsonl> [--rate 10] [--seed 42]
"""
from __future__ import annotations
import argparse
import csv
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_jsonl")
    ap.add_argument("--rate", type=float, default=10.0,
                    help="Mean request arrival rate (req/sec) for Poisson process")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-size", type=int, default=128256,
                    help="Llama-3 family vocab size for random token IDs")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    arrival_ns = 0.0  # first request at t=0
    n = 0

    with open(args.input_csv, newline="") as fin, open(args.output_jsonl, "w") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            in_toks = int(row["input_len"])
            out_toks = int(row["output_len"])
            in_ids = [rng.randrange(args.vocab_size) for _ in range(in_toks)]
            out_ids = [rng.randrange(args.vocab_size) for _ in range(out_toks)]
            rec = {
                "input_toks": in_toks,
                "output_toks": out_toks,
                "arrival_time_ns": arrival_ns,
                "input_tok_ids": in_ids,
                "output_tok_ids": out_ids,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Poisson interarrival: exponential(rate). Convert sec → ns.
            interval_s = rng.expovariate(args.rate)
            arrival_ns += interval_s * 1_000_000_000
            n += 1

    print(f"wrote {n} requests to {args.output_jsonl}")
    print(f"final arrival time: {arrival_ns / 1e9:.2f} sec")


if __name__ == "__main__":
    main()
