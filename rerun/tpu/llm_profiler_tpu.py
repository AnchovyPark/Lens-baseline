#!/usr/bin/env python3
"""TPU layers profiler — port of authors' Colab notebook to a CLI script.

Source: LLMServingSim_ispass26/llm_profile/perf_models/TPU-v6e-1/llm_profiler_tpu.ipynb
(committed by the authors as their official TPU profiling pipeline.)

Differences from the notebook:
  - Drive mount / Colab UI removed; output goes to a local directory.
  - HF_TOKEN read from env (not Colab secrets).
  - Three modes selectable via --mode {profile, validate, all}.
  - Defaults match the notebook's "RUN PROFILE" / "RUN VALIDATION" cells but every
    knob is overridable on the command line.

Methodology (UNCHANGED from notebook):
  - Wraps every per-op forward with xp.Trace(tag) on TPU.
  - Replays `repeat` forwards inside xp.start_trace / xp.stop_trace.
  - Parses the dumped Chrome trace (.trace.json.gz under /tmp/xla_trace/plugins/profile/)
    and computes EXCLUSIVE device time per tag (stack-aware: parent − child).
  - Writes per-(input_len, kv_len, layer_name) rows to <out_dir>/<hardware>/<model>.csv.
  - validate_and_scale runs the full forward end-to-end with perf_counter syncs,
    fits sf = median(measured / estimated), then scales latency(ns) in place.

Why this is correct on TPU (and our previous Timer-based approach was not):
  - Timer + perf_counter wraps each op individually → forces ~160 syncs per forward.
    Each sync incurs 200-400 µs overhead → measurement is dominated by the syncs.
  - xp.Trace + Chrome trace exposes device-side event durations directly, no per-op
    sync needed. Only one sync at the end of the whole `repeat` loop.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import os
import statistics
import sys
import time
from collections import defaultdict, namedtuple
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- XLA bootstrap (must come before model import for some torch_xla builds) ----------
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    from torch_xla import runtime as xr

    _XLA_AVAILABLE = True
    print(f"[xla] torch_xla OK, device_type={xr.device_type()}, device={torch_xla.device()}")
except Exception as e:
    _XLA_AVAILABLE = False
    print(f"[xla] import failed: {e}")


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# ============================================================================
# Per-op timing wrapper
# ============================================================================

def xla_timed_wrapper(tag: str, fn, use_xla: bool = False):
    def wrapped(*args, **kwargs):
        if not use_xla:
            with torch.autograd.profiler.record_function(tag):
                return fn(*args, **kwargs)
        with xp.Trace(tag):
            return fn(*args, **kwargs)
    return wrapped


# ============================================================================
# KV-cache builders (from notebook, Llama + PhiMoE)
# ============================================================================

def create_llama_past_key_values(config, kv_len, device):
    from transformers.models.llama.modeling_llama import DynamicCache, LlamaRotaryEmbedding

    num_layers = config.num_hidden_layers
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    dtype = torch.bfloat16

    key_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)
    value_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)

    rope = LlamaRotaryEmbedding(config=config, device=device)
    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)
    cos, sin = rope(dummy_x, position_ids)

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(key_states, value_states, layer_idx, {
            "cos": cos, "sin": sin, "cache_position": position_ids,
        })
    return cache


def create_phimoe_past_key_values(config, kv_len, device):
    from transformers.models.llama.modeling_llama import DynamicCache
    from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding

    num_layers = config.num_hidden_layers
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    dtype = torch.bfloat16

    key_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)
    value_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)

    rope = PhimoeRotaryEmbedding(config=config)
    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    cos, sin = rope(dummy_x, kv_len)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(key_states, value_states, layer_idx, {
            "cos": cos, "sin": sin, "cache_position": position_ids,
        })
    return cache


# ============================================================================
# Per-arch patching (from notebook, verbatim)
# ============================================================================

def patch_llama_decoder_layer(layer, use_xla=False):
    sa, mlp = layer.self_attn, layer.mlp
    sa.q_proj.forward = xla_timed_wrapper("self_attn/q_proj", sa.q_proj.forward, use_xla)
    sa.k_proj.forward = xla_timed_wrapper("self_attn/k_proj", sa.k_proj.forward, use_xla)
    sa.v_proj.forward = xla_timed_wrapper("self_attn/v_proj", sa.v_proj.forward, use_xla)
    sa.o_proj.forward = xla_timed_wrapper("self_attn/o_proj", sa.o_proj.forward, use_xla)
    sa.forward = xla_timed_wrapper("self_attn", sa.forward, use_xla)

    mlp.gate_proj.forward = xla_timed_wrapper("mlp/gate_proj", mlp.gate_proj.forward, use_xla)
    mlp.up_proj.forward = xla_timed_wrapper("mlp/up_proj", mlp.up_proj.forward, use_xla)
    mlp.down_proj.forward = xla_timed_wrapper("mlp/down_proj", mlp.down_proj.forward, use_xla)
    mlp.forward = xla_timed_wrapper("mlp", mlp.forward, use_xla)

    layer.input_layernorm.forward = xla_timed_wrapper(
        "input_layernorm", layer.input_layernorm.forward, use_xla)
    layer.post_attention_layernorm.forward = xla_timed_wrapper(
        "post_layernorm", layer.post_attention_layernorm.forward, use_xla)


def patch_opt_decoder_layer(layer, use_xla=False):
    sa = layer.self_attn
    sa.q_proj.forward = xla_timed_wrapper("self_attn/q_proj", sa.q_proj.forward, use_xla)
    sa.k_proj.forward = xla_timed_wrapper("self_attn/k_proj", sa.k_proj.forward, use_xla)
    sa.v_proj.forward = xla_timed_wrapper("self_attn/v_proj", sa.v_proj.forward, use_xla)
    sa.out_proj.forward = xla_timed_wrapper("self_attn/o_proj", sa.out_proj.forward, use_xla)
    sa.forward = xla_timed_wrapper("self_attn", sa.forward, use_xla)

    layer.fc1.forward = xla_timed_wrapper("mlp/fc1", layer.fc1.forward, use_xla)
    layer.activation_fn = xla_timed_wrapper("mlp/act_fn", layer.activation_fn, use_xla)
    layer.fc2.forward = xla_timed_wrapper("mlp/fc2", layer.fc2.forward, use_xla)

    layer.self_attn_layer_norm.forward = xla_timed_wrapper(
        "input_layernorm", layer.self_attn_layer_norm.forward, use_xla)
    layer.final_layer_norm.forward = xla_timed_wrapper(
        "post_layernorm", layer.final_layer_norm.forward, use_xla)


def patch_phimoe_decoder_layer(layer, use_xla=False):
    sa = layer.self_attn
    sa.q_proj.forward = xla_timed_wrapper("self_attn/q_proj", sa.q_proj.forward, use_xla)
    sa.k_proj.forward = xla_timed_wrapper("self_attn/k_proj", sa.k_proj.forward, use_xla)
    sa.v_proj.forward = xla_timed_wrapper("self_attn/v_proj", sa.v_proj.forward, use_xla)
    sa.o_proj.forward = xla_timed_wrapper("self_attn/o_proj", sa.o_proj.forward, use_xla)
    sa.forward = xla_timed_wrapper("self_attn", sa.forward, use_xla)

    moe = getattr(layer, "block_sparse_moe", None)
    if moe is not None:
        moe.forward = xla_timed_wrapper("mlp", moe.forward, use_xla)
        if hasattr(moe, "gate"):
            moe.gate.forward = xla_timed_wrapper("mlp/gate", moe.gate.forward, use_xla)
        try:
            import transformers.models.phimoe.modeling_phimoe as pm
            if hasattr(pm, "sparsemixer"):
                pm.sparsemixer = xla_timed_wrapper("mlp/sparsemixer", pm.sparsemixer, use_xla)
        except ImportError:
            pass
        experts = getattr(moe, "experts", None)
        if experts is not None:
            for expert in experts:
                if hasattr(expert, "w1"):
                    expert.w1.forward = xla_timed_wrapper("mlp/expert.w1", expert.w1.forward, use_xla)
                if hasattr(expert, "w3"):
                    expert.w3.forward = xla_timed_wrapper("mlp/expert.w2", expert.w3.forward, use_xla)
                if hasattr(expert, "w2"):
                    expert.w2.forward = xla_timed_wrapper("mlp/expert.w3", expert.w2.forward, use_xla)

    layer.input_layernorm.forward = xla_timed_wrapper(
        "input_layernorm", layer.input_layernorm.forward, use_xla)
    layer.post_attention_layernorm.forward = xla_timed_wrapper(
        "post_layernorm", layer.post_attention_layernorm.forward, use_xla)


def patch_model(model, config, use_xla=False):
    archs = [a.lower() for a in getattr(config, "architectures", [])]
    arch = archs[0] if archs else ""
    if "llama" in arch:
        for lyr in model.model.layers:
            patch_llama_decoder_layer(lyr, use_xla=use_xla)
        model.model.embed_tokens.forward = xla_timed_wrapper(
            "embedding", model.model.embed_tokens.forward, use_xla)
        model.model.norm.forward = xla_timed_wrapper(
            "final_layernorm", model.model.norm.forward, use_xla)
    elif "opt" in arch:
        for lyr in model.model.decoder.layers:
            patch_opt_decoder_layer(lyr, use_xla=use_xla)
        model.model.decoder.embed_tokens.forward = xla_timed_wrapper(
            "embedding", model.model.decoder.embed_tokens.forward, use_xla)
        model.model.decoder.final_layer_norm.forward = xla_timed_wrapper(
            "final_layernorm", model.model.decoder.final_layer_norm.forward, use_xla)
    elif "phimoe" in arch:
        for lyr in model.model.layers:
            patch_phimoe_decoder_layer(lyr, use_xla=use_xla)
        model.model.embed_tokens.forward = xla_timed_wrapper(
            "embedding", model.model.embed_tokens.forward, use_xla)
        model.model.norm.forward = xla_timed_wrapper(
            "final_layernorm", model.model.norm.forward, use_xla)
    else:
        raise NotImplementedError(f"Unsupported arch: {archs}")
    model.lm_head.forward = xla_timed_wrapper("lm_head", model.lm_head.forward, use_xla)


# ============================================================================
# Chrome trace parsing — exclusive device time per tag
# ============================================================================

def _find_trace_files(logdir: str) -> List[str]:
    hits = []
    for root, _, files in os.walk(logdir):
        for fn in files:
            if fn.endswith(".trace.json.gz") or fn.endswith(".json"):
                hits.append(os.path.join(root, fn))
    return hits


def _latest_trace_files(logdir: str) -> List[str]:
    run_dirs = []
    profile_root = os.path.join(logdir, "plugins", "profile")
    if os.path.isdir(profile_root):
        for d in os.listdir(profile_root):
            full = os.path.join(profile_root, d)
            if os.path.isdir(full):
                run_dirs.append(full)
    if not run_dirs:
        return _find_trace_files(logdir)
    latest = max(run_dirs, key=os.path.getmtime)
    hits = []
    for root, _, files in os.walk(latest):
        for fn in files:
            if fn.endswith(".trace.json.gz") or fn.endswith(".json"):
                hits.append(os.path.join(root, fn))
    return hits


def _load_events(logdir_or_file: str):
    files = _latest_trace_files(logdir_or_file) if os.path.isdir(logdir_or_file) else [logdir_or_file]
    evs = []
    for fp in files:
        try:
            opener = gzip.open if fp.endswith(".gz") else open
            with opener(fp, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        arr = data.get("traceEvents", data if isinstance(data, list) else [])
        for ev in arr:
            if isinstance(ev, dict) and ev.get("ph") == "X" and "dur" in ev and "ts" in ev:
                evs.append({
                    "name": ev.get("name", ""),
                    "pid": ev.get("pid", -1),
                    "tid": ev.get("tid", -1),
                    "ts": ev.get("ts", 0),
                    "dur": ev.get("dur", 0),
                })
    return evs


def _exclusive_total(events):
    """Stack-aware exclusive time per tag (parent − children) in microseconds."""
    Marker = namedtuple("Marker", ["t", "type", "name"])
    by_thread = defaultdict(list)
    for ev in events:
        s = ev["ts"]
        e = ev["ts"] + ev["dur"]
        by_thread[(ev["pid"], ev["tid"])].append(Marker(s, "start", ev["name"]))
        by_thread[(ev["pid"], ev["tid"])].append(Marker(e, "end", ev["name"]))

    exclusive = defaultdict(float)
    for markers in by_thread.values():
        if not markers:
            continue
        markers.sort(key=lambda m: (m.t, 0 if m.type == "end" else 1))
        stack: List[str] = []
        prev = markers[0].t
        for m in markers:
            if stack:
                exclusive[stack[-1]] += (m.t - prev)
            prev = m.t
            if m.type == "start":
                stack.append(m.name)
            else:
                if stack and stack[-1] == m.name:
                    stack.pop()
                elif m.name in stack:
                    while stack and stack[-1] != m.name:
                        stack.pop()
                    if stack and stack[-1] == m.name:
                        stack.pop()
    return exclusive


def _map_tags_to_results(exclusive_us: dict, arch: str):
    ex = exclusive_us
    out = {}
    for k in ("embedding", "final_layernorm", "lm_head", "input_layernorm", "post_layernorm"):
        if k in ex:
            out[k] = ex[k]
    if "self_attn/q_proj" in ex: out["q_proj"] = ex["self_attn/q_proj"]
    if "self_attn/k_proj" in ex: out["k_proj"] = ex["self_attn/k_proj"]
    if "self_attn/v_proj" in ex: out["v_proj"] = ex["self_attn/v_proj"]
    if "self_attn/o_proj" in ex: out["o_proj"] = ex["self_attn/o_proj"]
    if "self_attn" in ex:        out["attn"] = ex["self_attn"]

    if "llama" in arch:
        if "mlp/gate_proj" in ex: out["gate_proj"] = ex["mlp/gate_proj"]
        if "mlp/up_proj"   in ex: out["up_proj"]   = ex["mlp/up_proj"]
        if "mlp/down_proj" in ex: out["down_proj"] = ex["mlp/down_proj"]
        if "mlp" in ex:           out["act_fn"]    = ex["mlp"]
    elif "opt" in arch:
        if "mlp/fc1"    in ex: out["fc1"]    = ex["mlp/fc1"]
        if "mlp/act_fn" in ex: out["act_fn"] = ex["mlp/act_fn"]
        if "mlp/fc2"    in ex: out["fc2"]    = ex["mlp/fc2"]
    elif "phi" in arch:
        if "mlp/gate"        in ex: out["gate"]        = ex["mlp/gate"]
        if "mlp/sparsemixer" in ex: out["sparsemixer"] = ex["mlp/sparsemixer"]
        if "mlp/expert.w1"   in ex: out["expert.w1"]   = ex["mlp/expert.w1"]
        if "mlp/expert.w2"   in ex: out["expert.w2"]   = ex["mlp/expert.w2"]
        if "mlp/expert.w3"   in ex: out["expert.w3"]   = ex["mlp/expert.w3"]
        if "mlp" in ex:             out["act_fn"]      = ex["mlp"]
    return out


# ============================================================================
# Device resolution
# ============================================================================

def resolve_device(device_flag: str):
    device_flag = (device_flag or "").lower()
    if device_flag in ("xla", "tpu") and _XLA_AVAILABLE:
        try:
            dev = torch_xla.device()
            return "xla", dev
        except Exception as e:
            print(f"[warn] could not initialize XLA device ({e})")
    if device_flag in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


# ============================================================================
# Main profiling loop
# ============================================================================

def run_profile(
    hardware: str,
    model_name: str,
    num_layers: int,
    input_lengths: Sequence[int],
    kv_cache_lengths: Sequence[int],
    out_dir: str,
    device_flag: str = "xla",
    warmup: int = 5,
    repeat: int = 20,
    csv_append: bool = True,
    verbose: bool = True,
    hf_token: str = "",
    flush_every: int = 100,
):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    out_dir_hw = os.path.join(out_dir, hardware)
    os.makedirs(out_dir_hw, exist_ok=True)
    out_path = os.path.join(out_dir_hw, f"{_sanitize_model_name(model_name)}.csv")
    fieldnames = ["hardware", "model", "layer_name", "input", "kv_cache", "latency(ns)"]

    def _ensure_header(path, overwrite=False):
        if overwrite or (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    _ensure_header(out_path, overwrite=(not csv_append))

    dev_kind, device = resolve_device(device_flag)
    if verbose:
        print(f"[device] kind={dev_kind}, device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token or None)
    dtype = torch.bfloat16 if (dev_kind == "xla") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=hf_token or None).eval()
    original_num_layers = model.config.num_hidden_layers

    if num_layers is not None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = model.model.layers[:num_layers]
        elif hasattr(model, "model") and hasattr(model.model, "decoder"):
            model.model.decoder.layers = model.model.decoder.layers[:num_layers]
        model.config.num_hidden_layers = num_layers
    model.to(device)

    use_xla_timers = (dev_kind == "xla")
    patch_model(model, model.config, use_xla=use_xla_timers)

    # (input, kv) pair sweep: notebook semantics — prefill (kv=0) then decode (input=1)
    pairs: List[Tuple[int, int]] = []
    for il in input_lengths:
        if il <= 0:
            il = 1
        pairs.append((il, 0))
    for kl in kv_cache_lengths:
        if kl <= 0:
            kl = 1
        pairs.append((1, kl))

    arch_tag = (model.config.architectures[0].lower() if model.config.architectures else "")
    outer = tqdm(pairs, desc="Profiling configs", unit="cfg")
    buffered_rows = []
    since_last_flush = 0

    for input_len, kv_len in outer:
        outer.set_postfix_str(f"in={input_len}, kv={kv_len}")
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, input_len), device=device)

        if "llama" in arch_tag:
            past_key_values = create_llama_past_key_values(model.config, kv_len, device)
        elif "phi" in arch_tag:
            past_key_values = create_phimoe_past_key_values(model.config, kv_len, device)
        else:
            kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            past_key_values = (
                torch.zeros((1, kv_heads, kv_len, head_dim), device=device, dtype=dtype),
                torch.zeros((1, kv_heads, kv_len, head_dim), device=device, dtype=dtype),
            )

        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_ids, past_key_values=past_key_values, use_cache=True)
            if use_xla_timers:
                torch_xla.sync()
            elif dev_kind == "cuda":
                torch.cuda.synchronize()

        results_us = {}
        if use_xla_timers:
            trace_dir = "/tmp/xla_trace"
            os.makedirs(trace_dir, exist_ok=True)
            for fn in os.listdir(trace_dir):
                fp = os.path.join(trace_dir, fn)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass

            xp.start_trace(log_dir=trace_dir)
            for _ in range(repeat):
                if kv_len == 0:
                    pkv = None
                else:
                    if "llama" in arch_tag:
                        pkv = create_llama_past_key_values(model.config, kv_len, device)
                    elif "phi" in arch_tag:
                        pkv = create_phimoe_past_key_values(model.config, kv_len, device)
                    else:
                        pkv = past_key_values
                with torch.no_grad():
                    _ = model(input_ids, past_key_values=pkv, use_cache=True)
            torch_xla.sync()
            xp.stop_trace()

            events = _load_events(trace_dir)
            exclusive_us = _exclusive_total(events)
            for k, v in exclusive_us.items():
                results_us[k] = float(v) / max(1, repeat) / num_layers
        # cuda/cpu fallback intentionally left out — TPU is the use case here.

        results = _map_tags_to_results(results_us, arch=arch_tag)

        if "llama" in arch_tag:
            block_comps = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj",
                           "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj"]
        elif "phi" in arch_tag:
            block_comps = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj",
                           "post_layernorm", "gate", "sparsemixer", "expert.w1", "expert.w2", "expert.w3"]
        else:
            block_comps = ["input_layernorm", "q_proj", "k_proj", "v_proj", "qk_matmul", "softmax",
                           "sv_matmul", "o_proj", "post_layernorm", "fc1", "act_fn", "fc2"]
        common_comps = ["embedding", "final_layernorm", "lm_head"]

        if verbose:
            block_ns = sum(int(max(results.get(c, 0.0), 0.0) * 1000.0) for c in block_comps)
            common_ns = sum(int(max(results.get(c, 0.0), 0.0) * 1000.0) for c in common_comps)
            est = common_ns + block_ns * original_num_layers / max(1, num_layers)
            print(f"in={input_len}, kv={kv_len}, estimated={est:.0f} ns")

        for comp in block_comps + common_comps:
            if comp in results:
                buffered_rows.append({
                    "hardware": hardware,
                    "model": model_name,
                    "layer_name": comp,
                    "input": input_len,
                    "kv_cache": kv_len,
                    "latency(ns)": int(max(results[comp], 0.0) * 1000.0),
                })

        since_last_flush += 1
        if since_last_flush >= max(1, flush_every):
            with open(out_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(buffered_rows)
                f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
            buffered_rows.clear()
            since_last_flush = 0

        gc.collect()

    if buffered_rows:
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(buffered_rows)
            f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass

    print(f"[ok] CSV written: {out_path}")
    return out_path


# ============================================================================
# Validation + scaling (from notebook)
# ============================================================================

def _csv_path(out_dir, hardware, model_name):
    return os.path.join(out_dir, hardware, f"{_sanitize_model_name(model_name)}.csv")


def measure_generation_latency(model, input_length=10, output_length=5,
                               warmup=3, repeat=5, verbose=False):
    device = next(model.parameters()).device
    input_ids = torch.randint(0, model.config.vocab_size, (1, input_length), device=device)

    for _ in range(warmup):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        torch_xla.sync()
        pkv = out.past_key_values
        last = input_ids[:, -1:]
        for _ in range(max(0, output_length)):
            with torch.no_grad():
                out = model(input_ids=last, past_key_values=pkv, use_cache=True)
            torch_xla.sync()
            pkv = out.past_key_values
            last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    total_ns = []
    for _ in range(repeat):
        iter_ns = 0
        torch_xla.sync()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=None, use_cache=True)
        torch_xla.sync()
        iter_ns += (time.perf_counter_ns() - t0)

        pkv = out.past_key_values
        last = input_ids[:, -1:]
        for _ in range(1, max(1, output_length)):
            torch_xla.sync()
            t1 = time.perf_counter_ns()
            with torch.no_grad():
                out = model(input_ids=last, past_key_values=pkv, use_cache=True)
            torch_xla.sync()
            iter_ns += (time.perf_counter_ns() - t1)
            pkv = out.past_key_values
            last = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        total_ns.append(iter_ns)

    dt_ms = statistics.median(total_ns) / 1e6
    if verbose:
        print(f"[measure] in={input_length}, out={output_length} -> {dt_ms:.2f} ms")
    return dt_ms


def estimate_total_latency(out_dir, hardware, model_name, num_layers,
                           input_length, output_length, csv_path=None, verbose=False):
    config = AutoConfig.from_pretrained(model_name)
    latency_db = defaultdict(dict)
    csv_path = csv_path or _csv_path(out_dir, hardware, model_name)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["input"]), int(row["kv_cache"]))
            latency_db[key][row["layer_name"]] = int(row["latency(ns)"])

    arch = model_name.lower()
    if "llama" in arch:
        comps = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj",
                 "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj"]
    elif "phi" in arch:
        comps = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj",
                 "post_layernorm", "gate", "sparsemixer"]
    else:
        comps = ["input_layernorm", "q_proj", "k_proj", "v_proj", "qk_matmul", "softmax",
                 "sv_matmul", "o_proj", "post_layernorm", "fc1", "act_fn", "fc2"]

    total_ns = 0
    pre_key = (input_length, 0)
    if pre_key not in latency_db:
        raise ValueError(f"missing latency for input={input_length}, kv=0 in {csv_path}")
    pre = latency_db[pre_key]
    total_ns += num_layers * sum(pre.get(c, 0) for c in comps)
    for c in ("embedding", "final_layernorm", "lm_head"):
        total_ns += pre.get(c, 0)

    if "phi" in arch:
        n_experts = getattr(config, "num_local_experts", 1)
        moe_comps = ["expert.w1", "expert.w2", "act_fn", "expert.w3"]
        total_ns += num_layers * n_experts * sum(pre.get(c, 0) for c in moe_comps)

    for i in range(max(0, output_length - 1)):
        kv = input_length + i
        dec_key = (1, kv)
        if dec_key not in latency_db:
            raise ValueError(f"missing latency for input=1, kv={kv} in {csv_path}")
        dec = latency_db[dec_key]
        total_ns += num_layers * sum(dec.get(c, 0) for c in comps)
        total_ns += dec.get("lm_head", 0)
        if "phi" in arch:
            total_ns += num_layers * n_experts * sum(dec.get(c, 0) for c in moe_comps)
        for c in ("embedding", "final_layernorm", "lm_head"):
            total_ns += dec.get(c, 0)
        total_ns += 10e6 * i  # 10 ms compile time per decode step (notebook heuristic)

    if verbose:
        print(f"[estimate] {total_ns/1e6:.3f} ms")
    return total_ns / 1e6


def validate_latency_estimation(out_dir, hardware, model_name, num_layers=None,
                                input_lengths=(10, 20), output_lengths=(2, 4),
                                warmup=3, repeat=10, verbose=False, csv_path=None,
                                hf_token: str = ""):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, token=hf_token or None).eval()
    if num_layers is not None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = model.model.layers[:num_layers]
        elif hasattr(model, "model") and hasattr(model.model, "decoder"):
            model.model.decoder.layers = model.model.decoder.layers[:num_layers]
    model.to(torch.device("xla"))

    rows = []
    for il in input_lengths:
        for ol in output_lengths:
            il = max(1, il)
            ol = max(1, ol)
            m_meas = measure_generation_latency(model, il, ol, warmup=warmup, repeat=repeat)
            m_est = estimate_total_latency(out_dir, hardware, model_name,
                                           num_layers or model.config.num_hidden_layers,
                                           il, ol, csv_path=csv_path)
            err = (m_est - m_meas) / max(1e-9, m_meas) * 100.0
            rows.append((il, ol, m_meas, m_est, err))
            if verbose:
                print(f"[validate] in={il}, out={ol}  measured={m_meas:.2f} ms  "
                      f"estimated={m_est:.2f} ms  Δ={err:+.1f}%")
    return rows


def _compute_scaling_factor(rows, method="median"):
    ratios = [meas / est for (_, _, meas, est, _) in rows if est and est > 0]
    if not ratios:
        return 1.0
    if method == "mean":
        return float(sum(ratios) / len(ratios))
    return float(statistics.median(ratios))


def _compute_error_stats(rows):
    errs = [err for (*_, err) in rows if err is not None]
    if not errs:
        return {"n": 0, "mean_signed_pct_err": 0.0, "mean_abs_pct_err": 0.0,
                "median_abs_pct_err": 0.0}
    abs_errs = [abs(e) for e in errs]
    return {"n": len(errs),
            "mean_signed_pct_err": float(sum(errs) / len(errs)),
            "mean_abs_pct_err": float(sum(abs_errs) / len(abs_errs)),
            "median_abs_pct_err": float(statistics.median(abs_errs))}


def scale_latency_csv(out_dir, hardware, model_name, scaling_factor=1.0, overwrite=False):
    src = _csv_path(out_dir, hardware, model_name)
    dst = src if overwrite else src.replace(".csv", f".scaled_{scaling_factor:.2f}.csv")
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if not overwrite and "scaled_latency(ns)" not in fieldnames:
        fieldnames.append("scaled_latency(ns)")
    for row in rows:
        val = int(row["latency(ns)"])
        if overwrite:
            row["latency(ns)"] = int(val * scaling_factor)
        else:
            row["scaled_latency(ns)"] = int(val * scaling_factor)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] scaled CSV: {dst}")
    return dst


def validate_and_scale(out_dir, hardware, model_name, num_layers=None,
                       input_lengths=(1,), output_lengths=(1, 2),
                       warmup=5, repeat=3, verbose=True, method="median",
                       overwrite=False, csv_path=None, hf_token: str = ""):
    rows = validate_latency_estimation(
        out_dir=out_dir, hardware=hardware, model_name=model_name,
        num_layers=num_layers, input_lengths=input_lengths, output_lengths=output_lengths,
        warmup=warmup, repeat=repeat, verbose=verbose, csv_path=csv_path, hf_token=hf_token)
    sf = _compute_scaling_factor(rows, method=method)
    err_stats = _compute_error_stats(rows)
    if verbose:
        print(f"[scale] fitted scaling_factor = {sf:.4f} (method={method})")
        print(f"[error] n={err_stats['n']}  mean_abs={err_stats['mean_abs_pct_err']:.2f}%  "
              f"median_abs={err_stats['median_abs_pct_err']:.2f}%  "
              f"mean_signed={err_stats['mean_signed_pct_err']:+.2f}%")
    scaled_path = scale_latency_csv(out_dir=out_dir, hardware=hardware, model_name=model_name,
                                    scaling_factor=sf, overwrite=overwrite)
    return sf, scaled_path, rows, err_stats


# ============================================================================
# CLI
# ============================================================================

def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["profile", "validate", "all"], default="all")
    ap.add_argument("--hardware", default="TPU-v6e-1")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--num-layers", type=int, default=1,
                    help="truncate model to this many layers for profiling (notebook default=1)")
    ap.add_argument("--out-dir", default="./perf_models",
                    help="root dir; CSV goes to <out-dir>/<hardware>/<model>.csv")
    ap.add_argument("--device", default="xla", choices=["xla", "tpu", "cuda", "gpu", "cpu"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=30)
    # Default sweep extended to 8192 (vs notebook's 2048) so layers.csv covers
    # arXiv / writing-prompts contexts. Notebook step=1 dense remains the default
    # to match author's methodology — override with --prefill-step / --decode-step
    # for sparse sweeps (e.g. step=8 for ~8x speedup with minor predictor accuracy loss).
    ap.add_argument("--prefill-max", type=int, default=8192)
    ap.add_argument("--prefill-step", type=int, default=1)
    ap.add_argument("--decode-max", type=int, default=8192)
    ap.add_argument("--decode-step", type=int, default=1)
    ap.add_argument("--input-lengths", default="",
                    help="comma-separated override for prefill sweep (else range(0, prefill-max+1, prefill-step))")
    ap.add_argument("--kv-lengths", default="",
                    help="comma-separated override for decode sweep (else range(0, decode-max+1, decode-step))")
    ap.add_argument("--csv-append", action="store_true",
                    help="append to existing CSV instead of overwriting (default: overwrite)")
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))

    # validate-only knobs (extended to 8192 to validate predictor behavior in
    # the long-context region too, matching the wider sweep)
    ap.add_argument("--val-input-lengths", default="1,128,512,1024,2048,4096,8192")
    ap.add_argument("--val-output-lengths", default="1")
    ap.add_argument("--val-num-layers", type=int, default=None,
                    help="num_layers to use during validation; defaults to model's full layer count")
    ap.add_argument("--scale-method", choices=["median", "mean"], default="median")
    ap.add_argument("--scale-overwrite", action="store_true",
                    help="overwrite latency(ns) in place (else add scaled_latency(ns) column)")

    args = ap.parse_args()

    if args.mode in ("profile", "all"):
        if args.input_lengths:
            input_prefill = _parse_int_list(args.input_lengths)
        else:
            input_prefill = list(range(0, args.prefill_max + 1, args.prefill_step))
        if args.kv_lengths:
            kv_decode = _parse_int_list(args.kv_lengths)
        else:
            kv_decode = list(range(0, args.decode_max + 1, args.decode_step))

        print(f"[profile] input lens: {len(input_prefill)}, kv lens: {len(kv_decode)}")

        # Optional XLA profiler server (matches notebook PROF_PORT=9012)
        if _XLA_AVAILABLE:
            try:
                xp.start_server(int(os.environ.get("XLA_PROFILER_PORT", "9012")))
            except Exception:
                pass

        run_profile(
            hardware=args.hardware,
            model_name=args.model,
            num_layers=args.num_layers,
            input_lengths=input_prefill,
            kv_cache_lengths=kv_decode,
            out_dir=args.out_dir,
            device_flag=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            csv_append=args.csv_append,
            verbose=False,
            hf_token=args.hf_token,
            flush_every=args.flush_every,
        )

    if args.mode in ("validate", "all"):
        val_in = _parse_int_list(args.val_input_lengths)
        val_out = _parse_int_list(args.val_output_lengths)
        validate_and_scale(
            out_dir=args.out_dir,
            hardware=args.hardware,
            model_name=args.model,
            num_layers=args.val_num_layers,
            input_lengths=tuple(val_in),
            output_lengths=tuple(val_out),
            warmup=args.warmup,
            repeat=args.repeat,
            verbose=True,
            method=args.scale_method,
            overwrite=args.scale_overwrite,
            hf_token=args.hf_token,
        )


if __name__ == "__main__":
    main()
