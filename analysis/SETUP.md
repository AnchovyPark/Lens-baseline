# v5e Re-profiling Setup & Plan

This file walks through resuming profiling on TPU v5e after the first attempt
produced unusable per-layer durations (host-side `xp.Trace` events only captured
Python IR-build time in `torch_xla 2.5`, not device compute).

The plan restores the notebook's original profiling design by **downgrading
torch_xla to a version that still emits Chrome trace JSON** (with host↔device
correlation baked in), and then re-runs the existing
`LLMServingSim/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb` logic
nearly as-is.

---

## Source of truth for env setup

The notebook's cell 0 (LLMServingSim repo) specifies:

```
pip install "torch" --index-url https://download.pytorch.org/whl/cpu
pip install "torch_xla[tpu]" -f https://storage.googleapis.com/libtpu-releases/index.html
pip install "transformers" "accelerate>=0.33" sentencepiece
```

No version pins — that's what originally broke things. We pin only what's
necessary to bring the deprecated `xp.start_trace` / `xp.stop_trace` API back:

**Target**: `torch==2.4.0` + `torch_xla[tpu]==2.4.0`.
Rationale: 2.4 is the last release that shipped **both** the old trace API
(`xp.start_trace`/`xp.stop_trace`) and the new shorthand (`torch_xla.sync()`).
2.5 removed the old API; going older than 2.4 would require additional patches
(`torch_xla.sync()` → `xm.mark_step()`, etc.).

Fallback if 2.4 lookup fails at runtime: 2.3.0 (has old trace API, requires the
two API-shim patches noted below).

---

## TPU VM: `pjh-prof-1c` (zone us-west4-a, project llm-inference-468910)

Already on it:
- `~/torch-tpu/` — venv with torch_xla 2.5 (from failed first attempt); leave as-is.
- `~/v5e_prof/` — old profiler scripts + logs from first attempt.
- `~/perf_models_raw/TPU-v5e-1/*.csv` — empty CSV from first attempt (headers only).

New venv will be `~/torch-tpu-2.4/` — does not touch anything above.

---

## Phase-by-phase execution

### Phase 0 — commit & push (local, no TPU)

Done as part of this commit: analysis scripts, first-attempt traces, this doc.

### Phase 1 — provision torch_xla 2.4 env on v5e

```bash
gcloud compute tpus tpu-vm start pjh-prof-1c --zone=us-west4-a --project=llm-inference-468910

gcloud compute tpus tpu-vm ssh pjh-prof-1c --zone=us-west4-a --project=llm-inference-468910
# On VM:
python3 -m venv ~/torch-tpu-2.4
source ~/torch-tpu-2.4/bin/activate
pip install --upgrade pip

# Notebook's install lines, with versions pinned:
pip install 'torch==2.4.0' --index-url https://download.pytorch.org/whl/cpu
pip install 'torch_xla[tpu]==2.4.0' -f https://storage.googleapis.com/libtpu-releases/index.html
pip install 'transformers==4.45.0' 'accelerate>=0.33' sentencepiece

# Notebook also uses these during profiling / validation:
pip install numpy pandas tqdm huggingface_hub

# Smoke tests:
python3 -c "
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
print('torch_xla:', torch_xla.__version__)
print('device:', xm.xla_device())
print('has xp.start_trace:', hasattr(xp, 'start_trace'))
print('has xp.stop_trace:', hasattr(xp, 'stop_trace'))
print('has torch_xla.sync:', hasattr(torch_xla, 'sync'))
"
```

**Pass criteria**: all four boolean prints `True`, device is `xla:0`.

**Fail → Plan B**: use `torch_xla==2.5.0` but patch `xla_timed_wrapper` to call
`torch_xla.sync()` at the end of every wrapped forward. See `rerun/plan_b.md` — to
be written only if Phase 1 fails.

HF login (same token as before, cache persists from first attempt so may be a
no-op — verify):

```bash
huggingface-cli whoami
# if not logged in:
huggingface-cli login
```

### Phase 2 — regenerate & sync profiler scripts (local)

```bash
# On local Mac:
cd /Users/parkjuhyun/Desktop/baseline
python3 rerun/build_profiler_core.py   # re-extracts profiler_core.py from notebook,
                                        # without the torch_xla 2.5 workarounds

gcloud compute tpus tpu-vm scp --zone=us-west4-a --project=llm-inference-468910 \
  rerun/profiler_core.py rerun/run_dryrun_1config.py \
  pjh-prof-1c:~/v5e_prof_2.4/
```

### Phase 3 — 1-config smoke test (TPU, ~2 min)

```bash
# On VM:
source ~/torch-tpu-2.4/bin/activate
cd ~/v5e_prof_2.4
python3 run_dryrun_1config.py
```

**Pass criteria**:
- At least one `*.trace.json.gz` file appears under `/tmp/xla_trace/`.
- Parsing it produces non-zero latencies for our tags
  (`q_proj`, `gate_proj`, etc.).
- Numbers are on the order of microseconds (device compute), not 50 µs flat
  (the Python-overhead signature).

If the output still looks like Python overhead (everything ~50 µs), abort and
switch to Plan B.

### Phase 4 — sparse sweep (TPU, ~10-15 min)

`PREFILL_STEP=256, DECODE_STEP=256` → 9 prefill + 9 decode = 18 configs, same
sweep the first attempt used. Just verifies the sweep completes and generates
a usable `layers.csv`.

### Phase 5 — dense sweep (TPU, ~45-60 min)

`PREFILL_STEP=16, DECODE_STEP=16, WARMUP=10, REPEAT=30` → matches the coverage
of the existing `TPU-v6e-1/meta-llama/Llama-3.1-8B/tp1/layers.csv`. This is the
final data that LLMServingSim will consume.

After this, Phase 6 validates by re-using the notebook's Cell 10/11
(`validate_and_scale`) to compare estimated vs. measured end-to-end latency and
apply a global scale factor.

---

## Cost estimate

| Phase | TPU time |
|-------|----------|
| 0     | — (local) |
| 1     | ~10 min (installs) |
| 2     | — (local) |
| 3     | ~2 min |
| 4     | ~15 min |
| 5     | ~60 min |
| Total | **~90 min of v5e** |

Plus one-time HF model download (~2.5 GB for 1B-Instruct) already cached from
the first attempt.

---

## Why NOT Plan B by default

Keeping torch_xla 2.5 and bolting `torch_xla.sync()` inside every `xp.Trace`
works (host duration absorbs device-wait time) but:
- Serializes every projection — forward pass is no longer async-overlapped.
- Introduces sync overhead on each of ~10 tags per layer; skews small-op
  durations upward (sync itself is not free).
- Diverges from the reference notebook; harder to reconcile numbers with the
  existing TPU-v6e-1 profile.

The 2.4 downgrade keeps us on the original design with no code surgery beyond
potentially two `torch_xla.sync → xm.mark_step` swaps (if 2.4 lacks the new
shorthand, which is unlikely).
