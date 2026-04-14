"""Inspect /device:TPU:0 plane events to see if our tags are attached as stats.

If XLA op_metadata propagation works, HLO ops on the device should have a
stat (e.g. 'scope' or 'tf_op') pointing back to our xp.Trace tags.
"""
from __future__ import annotations
import glob
from collections import Counter, defaultdict

from xplane_reader import parse_xspace

TRACE_ROOT = "/Users/parkjuhyun/Desktop/baseline/v5e_trace_dryrun/xla_trace"
OUR_TAG_FRAGMENTS = ["input_layernorm", "post_layernorm", "final_layernorm",
                     "embedding", "lm_head", "self_attn", "mlp", "q_proj",
                     "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                     "down_proj", "act_fn"]


def inspect(path: str) -> None:
    planes = parse_xspace(path)
    tpu = next((p for p in planes if p.name == "/device:TPU:0"), None)
    if tpu is None:
        return
    print(f"\n==== {path.split('/')[-2]} ====")
    print(f"TPU plane: {len(tpu.lines)} lines, {sum(len(l.events) for l in tpu.lines)} events")
    print(f"stat_metadata count: {len(tpu.stat_metadata)}")

    # First: what stat names exist across all device events?
    stat_name_counts: Counter[str] = Counter()
    stat_name_has_our_tag: Counter[str] = Counter()
    sample_by_stat: dict[str, str] = {}
    tag_to_events: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for line in tpu.lines:
        for ev in line.events:
            ev_md = tpu.event_metadata.get(ev.metadata_id)
            ev_name = ev_md.name[:80] if ev_md else "?"
            for s in ev.stats:
                smd = tpu.stat_metadata.get(s.metadata_id)
                sname = smd.name if smd else f"?{s.metadata_id}"
                stat_name_counts[sname] += 1
                val_str = s.str_value if s.str_value is not None else \
                          str(s.int64_value) if s.int64_value is not None else \
                          str(s.uint64_value) if s.uint64_value is not None else \
                          str(s.double_value)
                # Check if any our-tag fragment appears in this stat value
                if val_str:
                    for frag in OUR_TAG_FRAGMENTS:
                        if frag in val_str:
                            stat_name_has_our_tag[sname] += 1
                            sample_by_stat.setdefault(sname, f"event={ev_name!r} stat_value={val_str[:160]!r}")
                            tag_to_events[frag].append((ev_name, ev.duration_ps))
                            break

    print(f"\n  all stat names on TPU plane ({len(stat_name_counts)}):")
    for name, count in stat_name_counts.most_common(20):
        mark = "  *** contains our tag ***" if name in stat_name_has_our_tag else ""
        print(f"    {count:5d}  {name}{mark}")

    if stat_name_has_our_tag:
        print(f"\n  !!! device events with our tags in stat values:")
        for sname, count in stat_name_has_our_tag.most_common():
            print(f"    stat={sname!r}: {count} matches")
            print(f"       sample: {sample_by_stat[sname]}")
        print(f"\n  tags → device event count (first 5 tags):")
        for tag in list(tag_to_events.keys())[:10]:
            evs = tag_to_events[tag]
            print(f"    {tag!r}: {len(evs)} device events, sum_duration={sum(d for _,d in evs)} ps")


if __name__ == "__main__":
    paths = sorted(glob.glob(f"{TRACE_ROOT}/plugins/profile/*/*.xplane.pb"))
    inspect(paths[0])       # prefill 1 token
    inspect(paths[4])       # prefill 1024 tokens (more compute)
    inspect(paths[-1])      # decode kv=2048
