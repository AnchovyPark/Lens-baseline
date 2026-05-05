# Project Memory

Last updated: 2026-05-05.

## Goal — UPDATED 2026-05-05 (CRITICAL)

**This project's only purpose is to faithfully reproduce LLMServingSim v2** (casys-kaist
fork — branch ispass26-artifact or main v1.1.0) using the author's code AS-IS.

**It is NOT** a benchmark of the simulator's accuracy, nor an effort to improve its
accuracy. We do not evaluate, "make it look better", or "make it look worse".

### Hard rules (do not violate)

- ❌ NEVER apply external calibration constants (sf / scaling factor / multiplicative
  correction) to perf-model CSVs or to the simulator's outputs.
- ❌ NEVER edit `layers.csv` / `attention.csv` to align simulator output to real
  measurements.
- ❌ NEVER tune `cluster_config` (HBM bw, link bw, etc.) to match real numbers — keep
  author/spec values exactly.
- ❌ NEVER add value judgments ("this is acceptable" / "this is too inaccurate") to
  baseline reports. Just present raw numbers.
- ✅ Build the simulator with the author's instructions; run it with the author's perf
  data; report whatever it outputs as-is.
- ✅ Any discrepancy with real measurements is the simulator's intrinsic behavior.
  Do not paper over it.

**Comparison is allowed only as observation**, never as something to be "fixed" by
local hacks. If the user asks "why does it differ", explain mechanisms. Do not
silently apply corrections.

### What we report
- Output of `python3 main.py ...` directly (the CSV the simulator writes).
- Comparison vs `inference_result/` real vLLM-TPU measurements as a side-by-side table,
  raw numbers, no editorial.

### What we don't do
- No `*.scaled_X.YY` perf files.
- No `fit_sf_*.py`, no `per_batch_sf*.py`, no calibration scripts.
- No "after applying X, error becomes Y" framing.

(Past mistake on 2026-05-04: I fitted sf=1.19 from real-vs-sim ratios and rewrote
perf CSVs ×sf, then reported "now error is 1.5%". User flagged this as wrong on
2026-05-05. All sf artifacts deleted.)

## Repo layout (2026-04-28)

```
baseline/
├── LLMServingSim/                 ← v1 (legacy, kept for reference)
├── LLMServingSim 2/               ← v2 unzipped (legacy, prefer LLMServingSim_ispass26)
├── LLMServingSim_ispass26/        ← FRESH CLONE of casys-kaist/LLMServingSim
│                                    branch ispass26-artifact (commit f4ab208, 2026-03-13).
│                                    THIS IS THE WORKING COPY going forward.
│                                    `git submodule update --init --recursive` already done.
├── ONNXim/                        ← cycle-level sim, separate
├── inference_result/              ← REAL vLLM-TPU measurements (ground truth)
│   └── Llama-3.1-8B-Instruct__bs{1,2,4}_tp1_seq2048/
│       {results,eval,summary}_{sharegpt,cnn,writing_prompts}.{csv,json}
├── dataset/                       ← pre-sampled (input_len, output_len) CSVs
├── rerun/tpu/                     ← OUR profiling/conversion tools
│   ├── llm_profiler_tpu.py        ← .py port of author's TPU notebook (NEW, 2026-04-28)
│   ├── run_attn_xla.py            ← attention profiler adapter (default v6e×8B max-len 8192)
│   ├── csv_to_jsonl.py            ← eval CSV → LLMServingSim JSONL
│   ├── convert_attn_to_tpu_schema.py  ← GPU schema → TPU schema
│   └── run_v6e_8b_baseline.sh     ← shell driver: 9 sims (3 dataset × 3 batch)
├── analysis/
│   └── compare_v6e_8b.py          ← simulator vs real measurement comparator
└── memory.md                      ← THIS FILE
```

## NPU + model scope (current session 2026-04-28)

**Active baseline target**:
- **Hardware**: TPU v6e-1 (single chip, 32 GB HBM, 1640 GB/s HBM bw)
- **Model**: meta-llama/Llama-3.1-8B (perf_models has it; the real measurement uses
  Llama-3.1-8B-Instruct — same architecture, OK to compare)
- **TP**: 1
- **max_seq**: 2048 (constrained by perf_models predictor coverage)
- **batch sweep**: {1, 2, 4} — power-of-2 only, real hardware can't go higher reliably
  on v6e-1 + 8B due to lm_head + activation memory (user empirically saw v5e×TP4×8B
  cap at batch ~8-16; single-chip v6e-1 likely caps at batch ~4-8).

**Datasets**: ShareGPT, CNN-DailyMail, Writing-prompts (50 samples each).
arXiv excluded — needs kv > 2048 which author's perf data doesn't cover.

**Deferred**: 1B/3B on v5e (model_config exists), Inf2 (needs new profiler script).

## Author's TPU profiler — found this session (CRITICAL)

`LLMServingSim_ispass26/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb`

Hidden in perf_models/, not under profiler/. This is the AUTHORITATIVE TPU profiler:
- Uses `xp.start_trace` + Chrome trace JSON parsing for EXCLUSIVE device time per tag
- Stack-aware exclusive computation (parent − children)
- Fits sklearn RandomForestRegressor predictor over the sweep
- `validate_and_scale` runs full forward end-to-end with perf_counter, fits sf
- For v6e × Llama-3.1-8B author got sf ≈ 0.95 (5% correction)
- Author sweeps step=1 dense, max-len=2048, batch 1~256 (decode side)

We ported the notebook to `rerun/tpu/llm_profiler_tpu.py` (CLI-driven, no Colab/Drive deps).

**NEVER do this**: per-op `Timer` with sync-after-each-op (our previous broken approach
gave sf ≈ 0.064 — 16× error — because each sync added 200-400 µs overhead × 160 ops/forward).

## Author's perf data limits (v6e × Llama-3.1-8B / tp1, pre-shipped)

| dimension | hard cap |
|---|---|
| single prefill chunk | ≤ 2048 tokens |
| **decode kv (input + cumulative output)** | **≤ 2048** (predictor stops there) |
| decode batch | predictor: ≤ 256, but real HBM-feasible: ~50 at full kv=2048 |
| simulator behavior outside cap | DB-mode: KeyError crash. Predictor-mode: random forest clamps at training boundary (silently wrong) |

**Implication for scenarios we can simulate RIGHT NOW with author's data alone**
(zero additional profiling):
- max sequence length (input + output) ≤ 2048
- batch sweep up to ≤ 256 in simulator (HBM-realistic ≤ 8)
- Covers ShareGPT, CNN, Writing-prompts safely
- arXiv (kv up to ~8.5k) requires extended profiling

## XLA bucketing — important methodology note

Author method uses **NO bucketing** — each (input_len, kv_len) shape triggers fresh
XLA recompilation. Step=1 dense sweep means thousands of recompiles. For 8B at
max-len=8192 step=1 this is multi-day; for max-len=2048 step=1 it's ~6-8 hours on v6e.

Simulator HAS bucketing internally:
- `_kv_cache_prediction_granularity = 64` (round kv up to 64)
- `_prefill_chunk_size_prediction_granularity = 32`
- attention.csv lookup uses these rounded keys
- BUT layers.csv lookup is exact — sparse profiling breaks layers.csv direct lookup

So **layers.csv must be step=1** (or simulator patches needed). Attention can be sparser.

## Real measurements (inference_result/)

Source: simple vLLM-TPU script (no external scheduler), all 50 requests sent burst.
Meta says `device: tpu-v5e` but per user this is a bug — actual device is **v6e**.

Per `summary_*.json` from bs1:
- ShareGPT: TTFT median 16.3 ms, TPOT median 14.8 ms
- CNN: TTFT median 30.9 ms, TPOT median 14.9 ms
- Writing-prompts: TTFT median 16.2 ms, TPOT median 14.7 ms

## Cluster config + JSONL prepared (ready to simulate)

```
LLMServingSim_ispass26/cluster_config/single_node_tpuv6e_8b.json
LLMServingSim_ispass26/dataset/v6e_8b_baseline/{sharegpt,cnn,writing_prompts}.jsonl
  ← all arrival_ns=0 (burst, mirrors vLLM script reality)
```

## Local execution status

**Mac CANNOT run LLMServingSim** — no Docker, astra-sim binary not built. Sandbox
also blocks `pip install` of external repos (chakra) into venv.

**Plan moving forward**: server. User has lab server with Docker setup from prior
session. Steps for server:

```bash
git clone <Lens-baseline fork>      # or git pull
cd Lens-baseline/LLMServingSim_ispass26
git submodule update --init --recursive
sudo bash docker.sh                 # or build via venv:
  python3 -m venv .venv && source .venv/bin/activate
  pip install astra-sim/extern/graph_frontend/chakra
  bash compile.sh
# then:
bash ../rerun/tpu/run_v6e_8b_baseline.sh   # adjust REPO path inside
python3 ../analysis/compare_v6e_8b.py
```

## Code modifications this session

- `rerun/tpu/llm_profiler_tpu.py` — NEW. Notebook port. Default `--hardware TPU-v6e-1`,
  `--model meta-llama/Llama-3.1-8B`, `--prefill-max 8192 --decode-max 8192` (extended
  beyond notebook's 2048 to cover arXiv-length contexts when we eventually re-profile).
- `rerun/tpu/run_attn_xla.py` — modified. HBM shim parameterized by `--hardware`
  (v5e=16GB, v6e=32GB). Default argv switched to v6e × Llama-3.1-8B × max-len 8192.
- `LLMServingSim_ispass26/cluster_config/single_node_tpuv6e_8b.json` — NEW.
- `rerun/tpu/run_v6e_8b_baseline.sh` — NEW (server runner).
- `analysis/compare_v6e_8b.py` — NEW (simulator vs real comparator).

## Where to pick up next session

1. SSH to lab server (or boot v6e VM if going for fresh profiling).
2. Pull this repo: `git clone https://github.com/AnchovyPark/Lens-baseline.git`.
3. Init submodules in `LLMServingSim_ispass26/` and build astra-sim (docker-based,
   `astrasim/tutorial-micro2024` image works).
4. Run `rerun/tpu/run_v6e_8b_baseline.sh` (adjust REPO path inside).
5. Run `analysis/compare_v6e_8b.py` to get real-vs-sim **observation** table.
6. **Report raw numbers only. Do not "fix" any discrepancy via sf, perf-data
   rewriting, or cluster_config tweaking. Per Goal section above.**

## Baseline state (2026-05-05, sim-as-shipped)

Pure LLMServingSim v2 with author-shipped TPU-v6e-1 perf data, no calibration:

bs=1 per-request medians:
- sharegpt:        TTFT 11.98 ms (real 16.28), TPOT 12.31 ms (real 14.80), E2E 3261 ms (real 3922)
- cnn:             TTFT 18.02 ms (real 30.91), TPOT 12.61 ms (real 14.88), E2E  796 ms (real  952)
- writing_prompts: TTFT 11.88 ms (real 16.17), TPOT 12.39 ms (real 14.74)

bs=2/4 wall-time and per-request comparison done, kept as-is in
`LLMServingSim_ispass26/output/v6e_8b_baseline/`. Schema mismatch (real has per-batch
batch_e2e_ms; sim has per-request) noted as observation; no special transform applied.

Note: `output/v6e_8b_baseline/sharegpt_bs1.csv` is currently MISSING because it was
deleted during a prior (now-discarded) sf experiment. Re-run with author's pristine
perf data needed to restore the cell.

## Key decisions / conventions

- TPU profiler = author's notebook approach (xp.start_trace + Chrome trace exclusive),
  NOT our broken Timer-per-op approach. Memory: feedback_tpu_profiling.md
- Datasets stay on (input_len, output_len) format; csv_to_jsonl.py adds dummy token
  IDs and arrival times. arrival=burst (rate ~1e8) for vLLM-script comparison.
- Batch sweep is power-of-2 only on TPU (XLA bucketing convention).
- Comparison metric: median TTFT, median TPOT; signed % error.
- Don't trust simulator output for kv > 2048 with author's pre-shipped data —
  it either crashes (DB mode) or silently clamps (predictor mode).
