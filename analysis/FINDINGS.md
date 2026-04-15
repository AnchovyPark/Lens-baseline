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

---

## 2026-04-15 update: the deeper root cause

Tested `torch_xla==2.4.0` and `torch_xla==2.8.0` (the latter matching what the
notebook's cell-0 output shows the original author ran: `torch: 2.8.0+cpu`).

Findings:

- **2.4**: `xp.start_trace` / `xp.stop_trace` removed. Only `trace_detached`
  and `xp.trace(duration_ms=...)` remain. Wrote only `xplane.pb`.
- **2.8**: `xp.start_trace` / `xp.stop_trace` restored. Writes **both**
  `.trace.json.gz` (Chrome trace) and `.xplane.pb`.

Even so, on 2.8 the per-layer latency numbers extracted by the notebook's
parser **still do not scale with input length** (q_proj: 225 µs at input=1,
218 µs at input=2048). The notebook reads events from `pid=701` which is the
host-side `/host:CPU` plane — compiler passes + `xp.Trace` scope markers with
IR-build-time durations.

The **real device compute time is on `pid=3` = `/device:TPU:0`**, carried by
HLO events (`SyncTensorsGraph.N`, `fusion.N`, `copy-*`). Those do scale
proportionally:

| input_len | SyncTensorsGraph (per forward) |
|-----------|-------------------------------:|
|  768      | 10.1 ms |
| 1024      | 12.0 ms |
| 1280      | 13.6 ms |
| 1536      | 15.9 ms |
| 1792      | 17.3 ms |
| 2048      | **19.5 ms** |

But pid=3 events carry no Python-scope back-reference — only HLO op names
like `fusion.42`. `xp.Trace` does not propagate scope metadata to XLA
op_metadata on the device side for this torch_xla/libtpu combination.

Small configs (input ≤ 512, all decode configs) produced zero device events
on pid=3 — the trace capture window closes before the (fast) work reaches
the profiler. Next run should use `REPEAT=30` or explicit longer trace
duration.

## Path forward (TPU-free analysis phase)

Correlating pid=3 device events back to Python layer scopes (`q_proj`,
`gate_proj`, etc.) has three candidate approaches, none of which need
another TPU run:

- **Timestamp correlation** — host-side `xp.Trace` events have a time window;
  device events falling within that window belong to that scope. Fragile
  (lazy dispatch means device work lags host window).
- **HLO shape correlation** — each `fusion.N` event's `args.long_name`
  contains the HLO shape (`bf16[2048,2048]` etc). Llama-3.2 layer shapes are
  unique enough to identify q_proj / k_proj / gate_proj / etc. from the
  matmul shapes. Precise but requires a shape catalog per model.
- **FLOPS-based split** — measure total `SyncTensorsGraph` time per config,
  then split proportionally by each layer's theoretical FLOP count. Approximate
  but simple. Good baseline; lose fine-grained accuracy for latency-bound
  ops (attention memory-reads dominate at long context, not FLOPS).

HLO shape correlation is likely the right answer for a real profile run, but
FLOPS split is the fast path to proving the full LLMServingSim pipeline
end-to-end before investing more TPU time.

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
