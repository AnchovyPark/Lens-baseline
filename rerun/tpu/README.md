# TPU adaptation of LLMServingSim's official profiling pipeline

The authors' README (`LLMServingSim/llm_profile/README.md` §"Adding a new model
or hardware") directs users to:

1. Use the `models/` scripts (instrumented with `record_function`/`Timer`
   context managers).
2. Set the hardware name in `profile_layers.sh` / `profile_attn.sh`.
3. Run the profiling + predictor build.

`profile_layers.sh` is GPU-hardcoded (`--device cuda`, `CUDA_VISIBLE_DEVICES=0`,
and the underlying `profiler/layers/main.py` calls `torch.cuda.synchronize()`).
For TPU we route through the authors' existing `perf_counter` profile method
(Timer:54-60, 88-93) and inject a *one-liner* compatibility shim that redirects
`torch.cuda.synchronize` → `torch_xla.sync`. No author source files are edited.

## Files in this directory

- `run_layers_xla.py` — thin driver. Applies the shim, then calls the authors'
  `profiler.layers.main.main()` with `--device xla --profile-method perf_counter`.
- `README.md` — this file.

## Running

On the TPU v5e VM:

```bash
# Prereqs: ~/torch-tpu-2.8 venv with torch==2.8.0, torch_xla[tpu]==2.8.0,
# transformers==4.45.0, plus the usual (accelerate, sentencepiece, ...).
# HF login already done.

# Upload llm_profile/ (use author's layout) + this driver:
# (done from local; see rerun/tpu/deploy.sh)

source ~/torch-tpu-2.8/bin/activate
cd ~/llm_profile
python3 run_layers_xla.py --max-len 64   # smoke
# then:
python3 run_layers_xla.py --max-len 2048 # full prefill sweep
```

Output lands in the authors' canonical path:
`perf_models/TPU-v5e-1/meta-llama/Llama-3.2-1B-Instruct/tp1/layers.csv`

## Why `perf_counter` and not `record_function`?

The authors expose four profile methods in `profiler/common/timer.py`:

- `cuda_event` — CUDA-only
- `kineto` — uses `ProfilerActivity.CUDA`, GPU-only
- `perf_counter` — `torch.cuda.synchronize() + time.perf_counter()`
  → **device-agnostic** once synchronize is XLA-aware
- `record_function` (default on GPU) — uses `RecordFunctionTracer` which
  correlates `user_annotation` events against `cuda_runtime` kernel-launch
  events by CUPTI correlation ID. CUPTI has no TPU/XLA counterpart that
  surfaces through `torch.profiler`, so this path won't attribute device
  time on TPU even if we rewrote the activities/category filters.

`perf_counter` is the only authors-supported method that gives meaningful
per-layer timings on TPU. It is legitimate authors' code — the shim just
teaches it to wait on the XLA queue instead of CUDA.
