"""Dig into our tagged events on /host:CPU: what are their durations?

If durations ~µs → just Python overhead (useless for layer latency).
If durations ~ms → they include device waiting time (usable).
Also dump stats to see if any device correlation exists.
"""
from __future__ import annotations
import glob
from collections import defaultdict
from statistics import median

from xplane_reader import parse_xspace

TRACE_ROOT = "/Users/parkjuhyun/Desktop/baseline/v5e_trace_dryrun/xla_trace"
OUR_TAGS = {
    "input_layernorm", "post_layernorm", "final_layernorm", "embedding", "lm_head",
    "self_attn", "self_attn/q_proj", "self_attn/k_proj", "self_attn/v_proj", "self_attn/o_proj",
    "mlp", "mlp/gate_proj", "mlp/up_proj", "mlp/down_proj", "mlp/act_fn",
}


def inspect_file(path: str) -> None:
    planes = parse_xspace(path)
    host = next((p for p in planes if p.name == "/host:CPU"), None)
    if host is None:
        return
    # Gather durations per tag
    tag_durs: dict[str, list[int]] = defaultdict(list)
    tag_stat_samples: dict[str, list[str]] = defaultdict(list)
    for line in host.lines:
        for ev in line.events:
            md = host.event_metadata.get(ev.metadata_id)
            if md is None:
                continue
            if md.name not in OUR_TAGS:
                continue
            tag_durs[md.name].append(ev.duration_ps)
            if len(tag_stat_samples[md.name]) < 1:
                for s in ev.stats:
                    smd = host.stat_metadata.get(s.metadata_id)
                    sname = smd.name if smd else f"?{s.metadata_id}"
                    val = (
                        s.str_value if s.str_value is not None
                        else s.int64_value if s.int64_value is not None
                        else s.uint64_value if s.uint64_value is not None
                        else s.double_value
                    )
                    tag_stat_samples[md.name].append(f"{sname}={val!r:.100s}" if isinstance(val, str) else f"{sname}={val}")

    if not tag_durs:
        return
    print(f"\n==== {path.split('/')[-2]} ====")
    print(f"{'tag':<25} {'count':>6} {'median_us':>12} {'mean_us':>12} {'max_us':>12}")
    for tag in sorted(tag_durs):
        ds = tag_durs[tag]
        ps_to_us = 1e-6
        print(f"{tag:<25} {len(ds):>6} {median(ds)*ps_to_us:>12.3f} "
              f"{(sum(ds)/len(ds))*ps_to_us:>12.3f} {max(ds)*ps_to_us:>12.3f}")
    # Show a couple of stat samples
    print("  sample stats:")
    for tag in list(tag_stat_samples)[:3]:
        print(f"    {tag}: {tag_stat_samples[tag]}")


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{TRACE_ROOT}/plugins/profile/*/*.xplane.pb"))
    # Inspect first few configs + one decode config
    targets = [paths[0], paths[5], paths[-1]] if len(paths) >= 6 else paths[:3]
    for p in targets:
        inspect_file(p)
