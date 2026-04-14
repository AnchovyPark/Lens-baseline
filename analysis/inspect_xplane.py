"""Dump every plane's structure + top events to see what torch_xla 2.5 wrote.

Focus question: do our xp.Trace() tags (e.g. 'self_attn/q_proj') show up anywhere
in the trace data — as event names, as stat values, or as metadata?
"""
from __future__ import annotations
import sys
import glob
from collections import Counter

from xplane_reader import parse_xspace, iter_events

TRACE_ROOT = "/Users/parkjuhyun/Desktop/baseline/v5e_trace_dryrun/xla_trace"
OUR_TAGS = {
    "input_layernorm", "post_layernorm", "final_layernorm", "embedding", "lm_head",
    "self_attn", "self_attn/q_proj", "self_attn/k_proj", "self_attn/v_proj", "self_attn/o_proj",
    "mlp", "mlp/gate_proj", "mlp/up_proj", "mlp/down_proj", "mlp/act_fn",
}


def inspect_one(path: str, show_stats: bool = False) -> None:
    print(f"\n==== {path} ====")
    planes = parse_xspace(path)
    print(f"total planes: {len(planes)}")
    for i, p in enumerate(planes):
        total_events = sum(len(l.events) for l in p.lines)
        print(f"\n  plane[{i}] name={p.name!r} id={p.id} lines={len(p.lines)} events={total_events}")
        print(f"     event_metadata entries: {len(p.event_metadata)}")
        print(f"     stat_metadata entries:  {len(p.stat_metadata)}")
        if total_events == 0:
            continue
        # Count event-name frequencies
        name_counts: Counter[str] = Counter()
        for line in p.lines:
            for ev in line.events:
                md = p.event_metadata.get(ev.metadata_id)
                nm = md.name if md else f"?id{ev.metadata_id}"
                name_counts[nm] += 1
        top = name_counts.most_common(8)
        print(f"     top event names:")
        for nm, c in top:
            print(f"       {c:5d}  {nm[:140]}")
        # Search for any of our tags in event names
        our_hits = [nm for nm in name_counts if any(tag in nm for tag in OUR_TAGS)]
        if our_hits:
            print(f"     *** OUR TAG HITS in event names: {our_hits[:10]}")
        # Search our tags in stat metadata NAMES
        stat_md_hits = [md.name for md in p.stat_metadata.values() if any(tag in md.name for tag in OUR_TAGS)]
        if stat_md_hits:
            print(f"     *** OUR TAG HITS in stat_metadata names: {stat_md_hits[:10]}")
        # Search stat VALUES (strings) for our tags — this is where op_metadata / annotations live
        tag_hits_in_stats: Counter[str] = Counter()
        stat_value_samples: dict[str, str] = {}
        for _, _, _, _, stats in iter_events(p):
            for sname, sval in stats:
                if isinstance(sval, str):
                    for tag in OUR_TAGS:
                        if tag in sval:
                            tag_hits_in_stats[tag] += 1
                            stat_value_samples.setdefault(tag, f"{sname} = {sval[:200]}")
        if tag_hits_in_stats:
            print(f"     *** OUR TAG HITS in stat VALUES: {dict(tag_hits_in_stats.most_common())}")
            for tag, sample in list(stat_value_samples.items())[:3]:
                print(f"         e.g. tag={tag!r}: {sample}")


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{TRACE_ROOT}/plugins/profile/*/*.xplane.pb"))
    if not paths:
        sys.exit("no xplane.pb found")
    # Inspect the first two in detail
    for p in paths[:2]:
        inspect_one(p)
    print(f"\n(total files: {len(paths)})")
