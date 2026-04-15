# Project Memory

Last updated: 2026-04-15.

## Goal

Compare **TTFT / TPOT** of LLM inference across NPUs using the two simulators
already vendored in this repo:

- **LLMServingSim v2** — serving-level simulator (`LLMServingSim/`). Native
  TTFT/TPOT output. Consumes per-layer latency CSVs under
  `LLMServingSim/llm_profile/perf_models/{hardware}/{model}/tp{N}/`.
- **ONNXim** — cycle-level NPU simulator (`ONNXim/`). Usable as a cycle-accurate
  backend to cross-check / refine LLMServingSim's layer latencies.

The primary output is per-request **TTFT / TPOT** from LLMServingSim.

## NPU scope

- **TPU v5e** — hardware access available (project `llm-inference-468910`, zone
  `us-west4-a`, VM `pjh-prof-1c`). Primary target.
- **AWS Inferentia2** — hardware access available. Secondary target; needs a
  port of the TPU profiler notebook to `torch-neuronx`.
- **TPU v6e** and **AWS Trainium1** — candidates only; no hardware access yet.
  Bring in later.

## Models

- **Llama-3.2-1B-Instruct** and **Llama-3.2-3B-Instruct**, chosen because 1B/3B
  fit on a single v5e chip in BF16 (16 GB HBM). Llama-3.1-8B does not fit v5e-1
  without tensor-parallel sharding (weights alone = 16 GB).
- LLMServingSim has built-in configs only for Llama-3.1-8B/70B; we'll add
  `model_config/meta-llama/Llama-3.2-{1B,3B}.json` (HF config mirrors) when
  setting up the simulator runs. No code changes expected — tokenizers are
  shared across Llama-3.1/3.2, datasets already tokenized with the 3.2 tokenizer.

## Datasets

Already sampled by the user (seed=43, n=50 per dataset, max_context=8192):

- `dataset/Llama-3.2-{1B,3B}-Instruct/eval_{sharegpt,cnn,arxiv}.csv`
  — columns: `id, input_len, output_len`. That's all LLMServingSim needs
  (request content doesn't matter; only lengths + arrival pattern).
- Length profiles:
  - ShareGPT: short in (median 24), medium out (median 299)
  - CNN/DailyMail: long in (median 876), short out (median 68)
  - arxiv: very long in (median 4873), short out (median 189)

CSV → LLMServingSim `.jsonl` conversion (add `arrival_time_ns`, random
`input_tok_ids`) is still pending — planned to happen after profiles exist.

## Profiling status

**Dry run #1 on v5e failed** (2026-04-14). Full writeup in
`analysis/FINDINGS.md`. TL;DR:
- Ran the reference notebook
  (`LLMServingSim/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb`) on
  torch_xla 2.5.
- torch_xla 2.5 removed `xp.start_trace`/`xp.stop_trace` and switched trace
  output from Chrome JSON to raw xplane.pb.
- After patching the API and parsing xplane.pb ourselves, we discovered
  host-side `xp.Trace` durations in 2.5 only capture Python IR-build time
  (~50 µs flat across config sizes) — not device compute time.
- Raw traces + scripts preserved under `v5e_trace_dryrun/`.

**Next step**: re-profile with torch_xla 2.4 (which still has `xp.start_trace`
and emits correlated Chrome trace JSON). Full plan in `analysis/SETUP.md`.
TPU is currently off (user will turn it back on when we resume).

## Definition of "pure hardware" TTFT

User confirmed: measuring TTFT/TPOT in the isolated-request regime, not the
serving-queue regime. Use `--max-batch 1` (force sequential execution) so
arrival rate / queueing effects drop out. Re-visit batched serving behavior
later, not now.

## Repo structure notes

- `LLMServingSim/` and `ONNXim/` were initially registered as git submodules
  with no content ever populated. User converted them to regular tracked
  directories. The `.gitmodules` inside each still exists, referencing
  unfetched sub-submodules:
  - `LLMServingSim/astra-sim/` (empty; from `casys-kaist/astra-sim`)
  - `ONNXim/extern/{onnx,protobuf,booksim,ramulator2,torch2timeloop}` (empty)
  
  These need to be cloned manually when building the simulators. The TPU
  profiling pipeline does not depend on them.

- `LLMServingSim/llm_profile/perf_models/TPU-v5e-1/meta-llama/Llama-3.2-1B-Instruct/tp1/layers.csv`
  currently holds a **placeholder** generated from dry-run #1's wrong host-side
  numbers. Will be overwritten by the proper re-profile. Kept locally only
  (not tracked via the repo because LLMServingSim was a submodule at the time
  it was written — though after conversion it would be tracked if re-added).

## Key decisions logged

- **Model**: Llama-3.2-3B-Instruct + 1B-Instruct (both, 3B primary)
- **NPUs for first comparison**: v5e alone initially; extend to Inf2/v6e/Trn1
  as access is granted
- **Instrumentation**: XLA profiler via `xp.Trace` (notebook's design), not
  wall-clock per-layer — user explicitly rejected the "wall-clock only"
  fallback path
- **TP**: 1 for all 1B/3B runs (fits on a single chip). TP>1 profiling would
  need extending the TPU notebook with JAX mesh / torch_xla SPMD sharding
  (deferred; only becomes relevant if we want 8B on v5e via v5e-4 slice)

## Where to pick up next session

1. User turns v5e VM on.
2. Execute `analysis/SETUP.md` Phase 1 (provision `~/torch-tpu-2.4/` venv).
3. Execute Phases 2-3 (smoke test with 1 config on 1B).
4. If green, Phase 4 (sparse sweep) → Phase 5 (dense sweep).
5. Resume dataset CSV→JSONL conversion + cluster_config authoring once real
   profiles are in hand (tasks #3, #6 in the task list).
