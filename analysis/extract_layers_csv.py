"""Extract a LLMServingSim-format layers.csv from xplane.pb trace files.

NOTE on accuracy: host-side xp.Trace events in torch_xla 2.5 measure Python IR
build time, NOT TPU device compute time. The durations emitted here are
order-of-magnitude wrong (Python overhead, not real compute). They're placeholders
that prove the pipeline — real numbers need a re-profile that calls
torch_xla.sync() inside each xp.Trace (see rerun_with_sync.py).

The layer_name normalization follows what the notebook's
`_map_tags_to_results` does:
  - self_attn/q_proj → q_proj, etc.
  - mlp/gate_proj → gate_proj, etc.
  - self_attn (exclusive) → attn
  - mlp (exclusive) → act_fn

Config identification: each xplane.pb file covers one (input_len, kv_len) config.
We infer which by reading run.log and matching timestamp order.
"""
from __future__ import annotations
import csv
import glob
import os
import re
from collections import defaultdict, namedtuple
from pathlib import Path
from statistics import median

from xplane_reader import parse_xspace

# ---- paths ----
TRACE_ROOT = Path("/Users/parkjuhyun/Desktop/baseline/v5e_trace_dryrun/xla_trace")
RUN_LOG = Path("/Users/parkjuhyun/Desktop/baseline/v5e_trace_dryrun/run.log")
OUT_DIR = Path(
    "/Users/parkjuhyun/Desktop/baseline/LLMServingSim/llm_profile/perf_models/"
    "TPU-v5e-1/meta-llama/Llama-3.2-1B-Instruct/tp1"
)

# ---- the xp.Trace tags we emitted in profiler_core.py's patch_llama_decoder_layer ----
PARENT_TAGS = {"self_attn", "mlp"}
LEAF_TAGS = {
    "input_layernorm", "post_layernorm", "final_layernorm", "embedding", "lm_head",
    "self_attn/q_proj", "self_attn/k_proj", "self_attn/v_proj", "self_attn/o_proj",
    "mlp/gate_proj", "mlp/up_proj", "mlp/down_proj",
}

# canonical layer names in the LLMServingSim layers.csv schema
TAG_TO_LAYER = {
    "input_layernorm": "input_layernorm",
    "post_layernorm": "post_layernorm",
    "final_layernorm": "final_layernorm",
    "embedding": "embedding",
    "lm_head": "lm_head",
    "self_attn/q_proj": "q_proj",
    "self_attn/k_proj": "k_proj",
    "self_attn/v_proj": "v_proj",
    "self_attn/o_proj": "o_proj",
    "mlp/gate_proj": "gate_proj",
    "mlp/up_proj": "up_proj",
    "mlp/down_proj": "down_proj",
}


def parse_run_log_configs(log_path: Path) -> list[tuple[int, int]]:
    """Return list of (input_len, kv_len) in order they appeared in the log."""
    configs: list[tuple[int, int]] = []
    pat = re.compile(r"input_len:\s*(\d+),\s*kv_len:\s*(\d+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                configs.append((int(m.group(1)), int(m.group(2))))
    return configs


def extract_per_config_latencies(xplane_path: str) -> dict[str, float]:
    """Given one xplane.pb, return {layer_name: median_duration_ns} for our layers.

    Uses exclusive-time logic for parent tags (self_attn, mlp) the way the
    notebook's _exclusive_total did, but simplified: parent exclusive =
    parent_total - sum(children_totals) averaged per repeat.
    """
    planes = parse_xspace(xplane_path)
    host = next((p for p in planes if p.name == "/host:CPU"), None)
    if host is None:
        return {}

    # collect all durations per tag (ps)
    per_tag_durs: dict[str, list[int]] = defaultdict(list)
    for line in host.lines:
        for ev in line.events:
            md = host.event_metadata.get(ev.metadata_id)
            if md is None:
                continue
            if md.name in LEAF_TAGS or md.name in PARENT_TAGS:
                per_tag_durs[md.name].append(ev.duration_ps)

    # median per tag in nanoseconds (ps → ns = /1000)
    med_ns = {tag: median(durs) / 1000.0 for tag, durs in per_tag_durs.items() if durs}

    out: dict[str, float] = {}
    for tag, layer in TAG_TO_LAYER.items():
        if tag in med_ns:
            out[layer] = med_ns[tag]

    # Parent exclusive: self_attn - (q+k+v+o) → 'attn' in CSV
    if "self_attn" in med_ns:
        children = sum(med_ns.get(t, 0.0) for t in
                       ("self_attn/q_proj", "self_attn/k_proj",
                        "self_attn/v_proj", "self_attn/o_proj"))
        out["attn"] = max(0.0, med_ns["self_attn"] - children)

    # mlp exclusive: mlp - (gate+up+down) → 'act_fn' in CSV (includes SiLU + elementwise)
    if "mlp" in med_ns:
        children = sum(med_ns.get(t, 0.0) for t in
                       ("mlp/gate_proj", "mlp/up_proj", "mlp/down_proj"))
        out["act_fn"] = max(0.0, med_ns["mlp"] - children)

    # rope is not separately tagged → zero for now (notebook's schema still expects it)
    out["rope"] = 0.0

    return out


def main() -> None:
    xplanes = sorted(glob.glob(f"{TRACE_ROOT}/plugins/profile/*/*.xplane.pb"))
    configs = parse_run_log_configs(RUN_LOG)

    print(f"xplane files: {len(xplanes)}")
    print(f"configs from run.log: {len(configs)}")
    if len(xplanes) != len(configs):
        print(f"  WARN: count mismatch — using min({len(xplanes)}, {len(configs)})")

    n = min(len(xplanes), len(configs))
    rows: list[dict] = []

    # Standard layer list matching existing perf_models/TPU-v6e-1 layers.csv schema
    LAYER_ORDER = [
        "input_layernorm", "rope", "embedding",
        "q_proj", "k_proj", "v_proj", "attn", "o_proj",
        "post_layernorm",
        "gate_proj", "up_proj", "act_fn", "down_proj",
        "final_layernorm", "lm_head",
    ]

    for i in range(n):
        input_len, kv_len = configs[i]
        latencies_ns = extract_per_config_latencies(xplanes[i])
        for layer in LAYER_ORDER:
            if layer in latencies_ns:
                rows.append({
                    "layer_name": layer,
                    "input": input_len,
                    "kv_cache": kv_len,
                    "tp_size": 1,
                    "latency(ns)": int(round(latencies_ns[layer])),
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "layers.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {out_path}")

    # quick preview
    print(f"\npreview (first 18 rows):")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 18:
                break
            print(f"  {line.rstrip()}")


if __name__ == "__main__":
    main()
