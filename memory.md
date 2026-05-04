# Project Memory

Last updated: 2026-04-28.

## Goal

Compare **TTFT / TPOT** of LLM inference across NPUs using two simulators in this repo:

- **LLMServingSim v2** — serving-level simulator. Native TTFT/TPOT output. Consumes
  per-layer latency CSVs under `<repo>/llm_profile/perf_models/{hardware}/{model}/tp{N}/`.
- **ONNXim** — cycle-level NPU simulator. Cross-check / refine LLMServingSim's per-layer latencies.

Primary output: per-request TTFT/TPOT from LLMServingSim, validated against real vLLM-TPU measurements.

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
3. Init submodules in `LLMServingSim_ispass26/` and build astra-sim.
4. Run `rerun/tpu/run_v6e_8b_baseline.sh` (adjust REPO path inside).
5. Run `analysis/compare_v6e_8b.py` to get real-vs-sim error table.
6. If errors are large → deep-dive (HBM model, latency model, batch handling).
7. If acceptable → expand scope: re-profile v6e × 8B with our extended sweep
   (max-len 8192, batches 1-8) to cover arXiv + writing-prompts long-tail.

## Key decisions / conventions

- TPU profiler = author's notebook approach (xp.start_trace + Chrome trace exclusive),
  NOT our broken Timer-per-op approach. Memory: feedback_tpu_profiling.md
- Datasets stay on (input_len, output_len) format; csv_to_jsonl.py adds dummy token
  IDs and arrival times. arrival=burst (rate ~1e8) for vLLM-script comparison.
- Batch sweep is power-of-2 only on TPU (XLA bucketing convention).
- Comparison metric: median TTFT, median TPOT; signed % error.
- Don't trust simulator output for kv > 2048 with author's pre-shipped data —
  it either crashes (DB mode) or silently clamps (predictor mode).
