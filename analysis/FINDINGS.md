# v5e Dry-run Findings (2026-04-14)

## What happened

Ran `LLMServingSim/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb`'s
profiler core (cell 4 + 5 + 6) on TPU v5e for `meta-llama/Llama-3.2-1B-Instruct`
with sparse sweep (PREFILL_STEP=256, DECODE_STEP=256, 18 configs total).

Execution **completed** (3m 22s, 18/18 configs). Output CSV was produced at
`~/perf_models_raw/TPU-v5e-1/meta-llama_Llama-3.2-1B-Instruct.csv` but contains
only the header — zero data rows.

## Root causes

### 1. `xp.start_trace` / `xp.stop_trace` removed in torch_xla 2.5

The notebook was written against an older torch_xla that shipped these APIs.
I patched `profiler_core.py` to use the replacement `xp.trace(duration_ms=...)`
via a background thread. That got the runtime to actually write trace files.

### 2. Trace output format changed from Chrome JSON → raw xplane.pb

`xp.trace()` on 2.5 writes `/tmp/xla_trace/plugins/profile/<ts>/*.xplane.pb`
(protobuf). The notebook's `_load_events()` parser explicitly looks for
`.trace.json.gz` / `.json` — our files don't match so `events = []` and every
config yielded `estimated: 0.0`.

I wrote a raw protobuf xplane reader (`xplane_reader.py`) that correctly parses
the files without needing the TF schema. That recovered the event data.

### 3. **The real problem**: host-side `xp.Trace` durations ≠ device compute time

With the parser working, host plane (`/host:CPU`) events for our tags are all
present (`self_attn/q_proj`, `mlp/gate_proj`, etc.) but their durations look
like this (Llama-3.2-1B, v5e):

| Tag | prefill 1 tok | prefill 1280 tok | decode kv=2048 |
|-----|--------------:|-----------------:|---------------:|
| `self_attn` (parent) | 1100 µs | 1142 µs | 1357 µs |
| `self_attn/q_proj` | 54 µs | 54 µs | 55 µs |
| `mlp/gate_proj` | 51 µs | 54 µs | 56 µs |
| `embedding` | 34 µs | 37 µs | 37 µs |

Prefill 1 token vs 1280 tokens is **~1.04× different**. Actual compute scales
~1280×. → **These are Python IR-build times, not device compute.**

torch_xla is lazy: `model(input)` only builds the XLA graph; actual TPU work
happens on the next `torch_xla.sync()`. The notebook calls sync **outside** the
per-op `xp.Trace` contexts, so each tag's duration captures only the IR-build
slice, not any device work.

On the TPU plane (`/device:TPU:0`) the real HLO op durations **are** recorded,
but the events carry stats `{hlo_op, _a, flow, id}` with zero reference back to
our Python-side tags. In torch_xla 2.5, `xp.Trace` does **not** propagate to
XLA op_metadata on the device.

## Why did the original notebook (on TPU-v6e-1) produce non-zero data?

Two non-exclusive hypotheses:

1. The chrome-trace post-processor (still present in older torch_xla) did the
   host-event ↔ device-event correlation before writing the JSON. Our 2.5
   run never got that post-processing — we got raw xplane.pb.
2. Older torch_xla may have synced more aggressively inside the trace scope,
   which would make host durations absorb device wait time.

Either way: **the fix is to use a torch_xla version whose profiler output is
already correlated (Chrome trace JSON) rather than raw xplane.pb**. That is
the downgrade-to-2.4 plan in `SETUP.md`.

## Artifacts in this directory

- `xplane_reader.py` — raw-wire-format reader for xplane.pb (no TF dep).
- `inspect_xplane.py` — dumps plane structure + event names.
- `inspect_durations.py` — per-tag duration stats across configs.
- `inspect_tpu_plane.py` — scans TPU plane events for tag correlation.
- `extract_layers_csv.py` — generates a LLMServingSim-format `layers.csv` from
  the (still wrong) host-side durations. Kept as the last-mile pipeline step so
  that once Phase 3+ re-profile produces correct numbers, the same script just
  works.
- `SETUP.md` — re-profiling plan (what to do next).

First-attempt raw data is in `../v5e_trace_dryrun/` (18 × 240 KB xplane.pb + log).
