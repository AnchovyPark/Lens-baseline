"""Profiler core (auto-generated from LLMServingSim/.../llm_profiler_tpu.ipynb
cells 1 + 4 + 5 + 6). Intended for torch_xla 2.4 which still ships xp.start_trace
and writes Chrome trace JSON (so the notebook's original parser works as-is).

A thin compatibility shim is added at top so the same file can also run against
older torch_xla (2.3) where torch_xla.sync() / torch_xla.device() didn't exist
yet — those fall back to xm.mark_step() / xm.xla_device().
"""
import os, sys, time, threading
import torch

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    from torch_xla import runtime as xr
    _XLA_AVAILABLE = True
except Exception as e:
    print("XLA import error:", e)
    _XLA_AVAILABLE = False

# Shim: torch_xla.sync / torch_xla.device were added in 2.4 as aliases.
# On 2.3 fall back to the older names.
if _XLA_AVAILABLE:
    if not hasattr(torch_xla, "sync"):
        torch_xla.sync = xm.mark_step   # type: ignore[attr-defined]
    if not hasattr(torch_xla, "device"):
        torch_xla.device = xm.xla_device  # type: ignore[attr-defined]

# ====================== TPU-aware micro-profiler core (inline) ======================
import os, gc, csv, time, math, types, json, gzip, statistics
from typing import List
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# -------- Lazy imports per-arch to avoid ImportError --------
def _llama_modules():
    import transformers.models.llama.modeling_llama as llm
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, DynamicCache
    return llm, LlamaRotaryEmbedding, DynamicCache

def _phimoe_modules():
    import transformers.models.phimoe.modeling_phimoe as pm
    from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding
    return pm, PhimoeRotaryEmbedding


# -------- XLA model wrapper --------
# torch_xla forbids mark_step()/sync() *inside* xp.Trace scopes
# ("RuntimeError: Expecting scope to be empty"). So we use the notebook's
# original design: call sync ONCE after the full repeat loop, outside all
# Trace scopes. Per-layer device attribution then relies on XLA's scope
# propagation into device HLO events (parsed from the xplane output).
def xla_timed_wrapper(tag, fn, use_xla: bool = False):
    def wrapped(*args, **kwargs):
        if not use_xla:
            with torch.autograd.profiler.record_function(tag):
                return fn(*args, **kwargs)
        else:
            with xp.Trace(tag):
                return fn(*args, **kwargs)
    return wrapped


# -------- KV cache builders --------
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, DynamicCache
def create_llama_past_key_values(config, kv_len, device):
    num_layers = config.num_hidden_layers
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    dtype = torch.bfloat16 if getattr(config, 'torch_dtype', None) in (torch.bfloat16, None) else config.torch_dtype

    key_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)
    value_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)

    rope = LlamaRotaryEmbedding(config=config, device=device)

    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)
    cos, sin = rope(dummy_x, position_ids)

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(key_states, value_states, layer_idx, {
            "cos": cos, "sin": sin, "cache_position": position_ids
        })
    return cache

def create_phimoe_past_key_values(config, kv_len, device):
    from transformers.models.llama.modeling_llama import DynamicCache
    num_layers = config.num_hidden_layers
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    dtype = torch.bfloat16 if getattr(config, 'torch_dtype', None) in (torch.bfloat16, None) else config.torch_dtype

    key_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)
    value_states = torch.zeros((1, num_kv, kv_len, head_dim), device=device, dtype=dtype)

    rope = PhimoeRotaryEmbedding(config=config)

    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    cos, sin = rope(dummy_x, kv_len)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(key_states, value_states, layer_idx, {
            "cos": cos, "sin": sin, "cache_position": position_ids
        })
    return cache

# -------- Arch-specific patching --------
def patch_llama_decoder_layer(layer, use_xla=False):
    # print(layer)
    sa, mlp = layer.self_attn, layer.mlp
    # --- Attention projections (children) ---
    sa.q_proj.forward = xla_timed_wrapper("self_attn/q_proj", sa.q_proj.forward, use_xla)
    sa.k_proj.forward = xla_timed_wrapper("self_attn/k_proj", sa.k_proj.forward, use_xla)
    sa.v_proj.forward = xla_timed_wrapper("self_attn/v_proj", sa.v_proj.forward, use_xla)
    sa.o_proj.forward = xla_timed_wrapper("self_attn/o_proj", sa.o_proj.forward, use_xla)

    # --- Attention block (parent) ---
    # exclusive("self_attn") == RoPE + QK^T + softmax + Attn·V
    sa.forward = xla_timed_wrapper("self_attn", sa.forward, use_xla)

    # --- MLP projections (children) ---
    mlp.gate_proj.forward = xla_timed_wrapper("mlp/gate_proj", mlp.gate_proj.forward, use_xla)
    mlp.up_proj.forward   = xla_timed_wrapper("mlp/up_proj",   mlp.up_proj.forward,   use_xla)
    mlp.down_proj.forward = xla_timed_wrapper("mlp/down_proj", mlp.down_proj.forward, use_xla)

    # --- MLP block (parent) ---
    # Activation is NOT separately tagged -> included in exclusive("mlp")
    mlp.forward = xla_timed_wrapper("mlp", mlp.forward, use_xla)

    # --- LayerNorm / RMSNorm ---
    # Keep these simple, per your request:
    layer.input_layernorm.forward = xla_timed_wrapper("input_layernorm", layer.input_layernorm.forward, use_xla)
    layer.post_attention_layernorm.forward = xla_timed_wrapper("post_layernorm",  layer.post_attention_layernorm.forward, use_xla)

    # NOTE: Do NOT tag SDPA or rope here if you want exclusive("self_attn") to equal the attention core.

def patch_opt_decoder_layer(layer, use_xla: bool = False):
    sa = layer.self_attn

    # --- Attention projections (children) ---
    sa.q_proj.forward   = xla_timed_wrapper("self_attn/q_proj", sa.q_proj.forward, use_xla)
    sa.k_proj.forward   = xla_timed_wrapper("self_attn/k_proj", sa.k_proj.forward, use_xla)
    sa.v_proj.forward   = xla_timed_wrapper("self_attn/v_proj", sa.v_proj.forward, use_xla)
    sa.out_proj.forward = xla_timed_wrapper("self_attn/o_proj", sa.out_proj.forward, use_xla)  # map OPT out_proj -> o_proj

    # --- Attention block (parent) ---
    # IMPORTANT: do NOT replace attention forward; just tag it so exclusive("self_attn")
    # becomes the attention core (matmul/softmax/matmul) since the projections above are children.
    # TODO: need to change matmul & softmax of OPT to attn in generate_trace
    sa.forward = xla_timed_wrapper("self_attn", sa.forward, use_xla)

    # --- MLP (OPT has no mlp module; tag parts individually) ---
    layer.fc1.forward  = xla_timed_wrapper("mlp/fc1", layer.fc1.forward,  use_xla)
    # activation_fn is a callable, not a Module.forward; it can be wrapped the same way:
    layer.activation_fn = xla_timed_wrapper("mlp/act_fn", layer.activation_fn, use_xla)
    layer.fc2.forward = xla_timed_wrapper("mlp/fc2", layer.fc2.forward,  use_xla)

    # --- LayerNorms ---
    layer.self_attn_layer_norm.forward = xla_timed_wrapper("input_layernorm", layer.self_attn_layer_norm.forward, use_xla)
    layer.final_layer_norm.forward = xla_timed_wrapper("post_layernorm",  layer.final_layer_norm.forward,     use_xla)

def patch_phimoe_decoder_layer(layer, use_xla=False):
     # ----- Attention -----
    sa = layer.self_attn
    sa.q_proj.forward = xla_timed_wrapper("self_attn/q_proj", sa.q_proj.forward, use_xla)
    sa.k_proj.forward = xla_timed_wrapper("self_attn/k_proj", sa.k_proj.forward, use_xla)
    sa.v_proj.forward = xla_timed_wrapper("self_attn/v_proj", sa.v_proj.forward, use_xla)
    sa.o_proj.forward = xla_timed_wrapper("self_attn/o_proj", sa.o_proj.forward, use_xla)
    # Parent: attention block → exclusive(self_attn) == QK^T + softmax + Attn·V
    sa.forward = xla_timed_wrapper("self_attn", sa.forward, use_xla)

    # (Intentionally NOT tagging rope/SDPA to avoid subtracting from self_attn exclusive)

    # ----- MoE / MLP -----
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is not None:
      # Parent: whole MoE block
      moe.forward = xla_timed_wrapper("mlp", moe.forward, use_xla)

      # Gate
      if hasattr(moe, "gate"):
          moe.gate.forward = xla_timed_wrapper("mlp/gate", moe.gate.forward, use_xla)

      # Sparse mixer (often defined in helper modules)
      try:
          pm, _ = _phimoe_modules()
      except NameError:
          pm = None
      if pm is not None and hasattr(pm, "sparsemixer"):
          pm.sparsemixer = xla_timed_wrapper("mlp/sparsemixer", pm.sparsemixer, use_xla)

      # Experts (w1=up, w3=gate, w2=down)
      experts = getattr(moe, "experts", None)
      if experts is not None:
          for expert in experts:
              if hasattr(expert, "w1"):
                  expert.w1.forward = xla_timed_wrapper("mlp/expert.w1",   expert.w1.forward, use_xla)
              if hasattr(expert, "w3"):
                  expert.w3.forward = xla_timed_wrapper("mlp/expert.w2", expert.w3.forward, use_xla)
              if hasattr(expert, "w2"):
                  expert.w2.forward = xla_timed_wrapper("mlp/expert.w3", expert.w2.forward, use_xla)
              # Do NOT wrap expert.act_fn → stays inside exclusive('mlp')

    # ----- LayerNorms -----
    layer.input_layernorm.forward = xla_timed_wrapper("input_layernorm", layer.input_layernorm.forward, use_xla)
    layer.post_attention_layernorm.forward = xla_timed_wrapper("post_layernorm",  layer.post_attention_layernorm.forward, use_xla)

def patch_model(model, config, use_xla=False):
    archs = [a.lower() for a in getattr(config, "architectures", [])]
    arch = archs[0] if archs else ""
    if "llama" in arch:
        for lyr in model.model.layers:
            patch_llama_decoder_layer(lyr, use_xla=use_xla)
        model.model.embed_tokens.forward = xla_timed_wrapper("embedding", model.model.embed_tokens.forward, use_xla)
        model.model.norm.forward         = xla_timed_wrapper("final_layernorm", model.model.norm.forward, use_xla)
    elif "opt" in arch:
        for lyr in model.model.decoder.layers:
            patch_opt_decoder_layer(lyr, use_xla=use_xla)
        model.model.decoder.embed_tokens.forward     = xla_timed_wrapper("embedding", model.model.decoder.embed_tokens.forward, use_xla)
        model.model.decoder.final_layer_norm.forward = xla_timed_wrapper("final_layernorm", model.model.decoder.final_layer_norm.forward, use_xla)
    elif "phimoe" in arch:
        for lyr in model.model.layers:
            patch_phimoe_decoder_layer(lyr, use_xla=use_xla)
        model.model.embed_tokens.forward = xla_timed_wrapper("embedding", model.model.embed_tokens.forward, use_xla)
        model.model.norm.forward         = xla_timed_wrapper("final_layernorm", model.model.norm.forward, use_xla)
    else:
        raise NotImplementedError(f"Unsupported arch: {archs}")

    model.lm_head.forward = xla_timed_wrapper("lm_head", model.lm_head.forward, use_xla)

def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")

def resolve_device(device_flag: str):
    device_flag = (device_flag or "").lower()
    if device_flag in ("xla", "tpu") and _XLA_AVAILABLE:
        try:
            dev = torch_xla.device()
            try:
                from torch_xla import runtime as _xr
                print(f"[XLA] runtime device_type={_xr.device_type()}")
            except Exception:
                pass
            return "xla", dev
        except Exception as e:
            print(f"[warn] Could not initialize XLA device ({e}); falling back.")
    if device_flag in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")

def run_profile(
    hardware="TPU-v6e-1",
    model_name="meta-llama/Llama3.1-8B",
    num_layers=1,
    input_lengths=(128, 256),
    kv_cache_lengths=(0, 128),
    device_flag="xla",
    warmup=5,
    repeat=20,
    csv_append=True,
    verbose=True,
    out_dir="/content/drive/MyDrive/tpu_profile",
    hf_token="",
    progress=True,
    flush_every=100,   # flush Drive every N configs
):

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # 0) path & header
    out_dir_hw = os.path.join(out_dir, hardware)
    os.makedirs(out_dir_hw, exist_ok=True)
    out_path = os.path.join(out_dir_hw, f"{_sanitize_model_name(model_name)}.csv")
    fieldnames = ["hardware","model","layer_name","input","kv_cache","latency(ns)"]

    def _ensure_header(path, overwrite=False):
        if overwrite:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            return
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    _ensure_header(out_path, overwrite=(not csv_append))
    if csv_append and (not os.path.exists(out_path) or os.path.getsize(out_path) == 0):
        _ensure_header(out_path, overwrite=False)

    existing_attn = set()

    if not out_path.startswith("/content/drive/"):
        print(f"[warn] out_path is not in Drive: {out_path}")

    # 1) device/model
    dev_kind, device = resolve_device(device_flag)
    if verbose:
        print(f"[Device] kind={dev_kind}, device={device}")

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
    model_val = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=hf_token or None).eval()
    model_val.to(device)
    use_xla_timers = (dev_kind == "xla")
    patch_model(model, model.config, use_xla=use_xla_timers)

    # 2) (input, kv) pairs
    pairs = []
    max_pos = getattr(model.config, "max_position_embeddings", None)
    for il in input_lengths:
        if il <=0:
            il = 1
        pairs.append((il, 0))
    for kl in kv_cache_lengths:
        if kl <=0:
            kl = 1
        pairs.append((1, kl))

    outer = tqdm(pairs, disable=not progress, desc="Profiling configs", unit="cfg")
    buffered_rows = []
    since_last_flush = 0

    # ---------- Chrome trace + xplane helpers ----------
    _TRACE_EXTS = (".trace.json.gz", ".json", ".xplane.pb")

    def _find_trace_files(logdir: str) -> List[str]:
        hits = []
        for root, _, files in os.walk(logdir):
            for fn in files:
                if fn.endswith(_TRACE_EXTS):
                    hits.append(os.path.join(root, fn))
        return hits

    def _latest_trace_files(logdir: str) -> List[str]:
        # profile run dirs look like: <logdir>/plugins/profile/2025_10_01_11_31_21
        run_dirs = []
        for root, dirs, _ in os.walk(os.path.join(logdir, "plugins", "profile")):
            for d in dirs:
                run_dirs.append(os.path.join(root, d))
            break  # only direct children under .../profile/
        if not run_dirs:
            return _find_trace_files(logdir)  # fallback: any files directly under logdir

        # choose latest by mtime (robust across envs)
        latest = max(run_dirs, key=lambda p: os.path.getmtime(p))
        hits = []
        json_files = []
        xplane_files = []
        for root, _, files in os.walk(latest):
            for fn in files:
                full = os.path.join(root, fn)
                if fn.endswith((".trace.json.gz", ".json")):
                    json_files.append(full)
                elif fn.endswith(".xplane.pb"):
                    xplane_files.append(full)
        # torch_xla >= 2.8 writes both JSON and xplane side-by-side; the JSON
        # carries post-processed host↔device correlation and is what the
        # notebook's parser was designed for. Prefer it; fall back to xplane.
        return json_files if json_files else xplane_files

    # --- minimal xplane.pb reader (raw protobuf wire-format, no TF dep) ---
    def _xp_vd(b, p):
        r = 0; s = 0
        while True:
            x = b[p]; p += 1
            r |= (x & 0x7F) << s
            if not (x & 0x80):
                return r, p
            s += 7

    def _xp_iter(b, start=0, end=None):
        if end is None:
            end = len(b)
        p = start
        while p < end:
            tag, p = _xp_vd(b, p)
            fn = tag >> 3; wt = tag & 7
            if wt == 0:
                v, p = _xp_vd(b, p); yield fn, 0, v
            elif wt == 1:
                yield fn, 1, b[p:p+8]; p += 8
            elif wt == 2:
                L, p = _xp_vd(b, p); yield fn, 2, b[p:p+L]; p += L
            elif wt == 5:
                yield fn, 5, b[p:p+4]; p += 4
            else:
                return

    def _xp_load_xplane(path: str):
        """Return list of {name,pid,tid,ts,dur} from one xplane.pb.

        Durations converted ps → µs; line.id used as tid; plane index as pid.
        """
        with open(path, "rb") as f:
            data = f.read()
        out = []
        plane_idx = 0
        # XSpace.planes field = 1
        for fn, wt, val in _xp_iter(data):
            if fn != 1 or wt != 2:
                continue
            plane_idx += 1
            plane_name = ""
            event_md = {}  # id -> name
            lines = []     # list of (line_id, [events])
            for pfn, pwt, pval in _xp_iter(val):
                if pfn == 2 and pwt == 2:
                    plane_name = pval.decode("utf-8", "ignore")
                elif pfn == 4 and pwt == 2:
                    key = None; md_name = ""
                    for mfn, mwt, mval in _xp_iter(pval):
                        if mfn == 1 and mwt == 0:
                            key = mval
                        elif mfn == 2 and mwt == 2:
                            for xfn, xwt, xval in _xp_iter(mval):
                                if xfn == 2 and xwt == 2:
                                    md_name = xval.decode("utf-8", "ignore")
                    if key is not None:
                        event_md[key] = md_name
                elif pfn == 3 and pwt == 2:
                    line_id = 0
                    evs = []
                    for lfn, lwt, lval in _xp_iter(pval):
                        if lfn == 1 and lwt == 0:
                            line_id = lval
                        elif lfn == 4 and lwt == 2:
                            mid = off = dur = 0
                            for efn, ewt, eval_ in _xp_iter(lval):
                                if efn == 1 and ewt == 0:
                                    mid = eval_
                                elif efn == 2 and ewt == 0:
                                    off = eval_
                                elif efn == 3 and ewt == 0:
                                    dur = eval_
                            evs.append((mid, off, dur))
                    lines.append((line_id, evs))
            # Only emit host plane events — device plane events don't carry our tags.
            # But give the caller both; _exclusive_total filters by name stack.
            for line_id, evs in lines:
                for mid, off_ps, dur_ps in evs:
                    out.append({
                        "name": event_md.get(mid, f"?id{mid}"),
                        "pid": plane_idx,
                        "tid": line_id,
                        "ts": off_ps / 1e6,   # ps → µs
                        "dur": dur_ps / 1e6,  # ps → µs
                    })
        return out

    def _load_events(logdir_or_file: str):
        files = []
        if os.path.isdir(logdir_or_file):
            files = _latest_trace_files(logdir_or_file)
        else:
            if logdir_or_file.endswith(_TRACE_EXTS):
                files = [logdir_or_file]

        evs = []
        for fp in files:
            try:
                if fp.endswith(".xplane.pb"):
                    evs.extend(_xp_load_xplane(fp))
                    continue
                if fp.endswith(".gz"):
                    with gzip.open(fp, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(fp, "r") as f:
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
                        "ts":  ev.get("ts", 0),   # µs
                        "dur": ev.get("dur", 0),  # µs
                    })
        return evs

    def _exclusive_total(events):
        """Compute EXCLUSIVE time per tag across the whole trace (in microseconds)."""
        Marker = namedtuple("Marker", ["t","type","name"])
        by_thread=defaultdict(list)
        for ev in events:
            s=ev["ts"]; e=ev["ts"]+ev["dur"]
            by_thread[(ev["pid"],ev["tid"])].append(Marker(s,"start",ev["name"]))
            by_thread[(ev["pid"],ev["tid"])].append(Marker(e,"end",  ev["name"]))
        exclusive=defaultdict(float)
        for key, markers in by_thread.items():
            if not markers: continue
            markers.sort(key=lambda m:(m.t, 0 if m.type=="end" else 1))
            stack=[]; prev=markers[0].t
            for m in markers:
                if stack:
                    exclusive[stack[-1]] += (m.t - prev)
                prev=m.t
                if m.type=="start":
                    stack.append(m.name)
                else:
                    if stack and stack[-1]==m.name:
                        stack.pop()
                    else:
                        if m.name in stack:
                            while stack and stack[-1]!=m.name:
                                stack.pop()
                            if stack and stack[-1]==m.name:
                                stack.pop()
        return exclusive  # microseconds

    def _map_tags_to_results(exclusive_us: dict, arch: str):
        """
        Map our tags to your CSV naming convention.
        Values are expected in microseconds. Return dict[name_in_csv] = us.
        """
        ex = exclusive_us
        out = {}

        # Common names present from patch_model:
        if "embedding" in ex:          out["embedding"] = ex["embedding"]
        if "final_layernorm" in ex:    out["final_layernorm"] = ex["final_layernorm"]
        if "lm_head" in ex:            out["lm_head"] = ex["lm_head"]
        if "input_layernorm" in ex:    out["input_layernorm"] = ex["input_layernorm"]
        if "post_layernorm" in ex:     out["post_layernorm"] = ex["post_layernorm"]

        # Attention projections
        if "self_attn/q_proj" in ex: out["q_proj"] = ex["self_attn/q_proj"]
        if "self_attn/k_proj" in ex: out["k_proj"] = ex["self_attn/k_proj"]
        if "self_attn/v_proj" in ex: out["v_proj"] = ex["self_attn/v_proj"]
        if "self_attn/o_proj" in ex: out["o_proj"] = ex["self_attn/o_proj"]
        # Attention Core (remainder)
        if "self_attn" in ex:
            out["attn"] = ex["self_attn"]

        # MLP / MoE
        if "llama" in arch:
            # projections
            if "mlp/gate_proj" in ex: out["gate_proj"] = ex["mlp/gate_proj"]
            if "mlp/up_proj"   in ex: out["up_proj"]   = ex["mlp/up_proj"]
            if "mlp/down_proj" in ex: out["down_proj"] = ex["mlp/down_proj"]
            # Activation (remainder)
            if "mlp"   in ex: out["act_fn"]    = ex["mlp"]

        elif "opt" in arch:
            # OPT: no 'mlp' parent; keep original names
            if "mlp/fc1"     in ex: out["fc1"]     = ex["mlp/fc1"]
            if "mlp/act_fn"  in ex: out["act_fn"]  = ex["mlp/act_fn"]
            if "mlp/fc2"     in ex: out["fc2"]     = ex["mlp/fc2"]
            # qk_matmul/softmax/sv_matmul only exist if you add custom tags; otherwise they are skipped.

        elif "phi" in arch:
            # Router / mixer
            if "mlp/gate"      in ex: out["gate"]        = ex["mlp/gate"]
            if "mlp/sparsemixer" in ex: out["sparsemixer"] = ex["mlp/sparsemixer"]
            # Experts (map to expert.w*)
            if "mlp/expert.w1"   in ex: out["expert.w1"]   = ex["mlp/expert.w1"]
            if "mlp/expert.w2"   in ex: out["expert.w2"]   = ex["mlp/expert.w2"]
            if "mlp/expert.w3"   in ex: out["expert.w3"]   = ex["mlp/expert.w3"]
            # Activation (remainder)
            if "mlp"   in ex: out["act_fn"]    = ex["mlp"]

        return out

    # 3) loop
    for input_len, kv_len in outer:
        if progress:
            outer.set_postfix_str(f"in={input_len}, kv={kv_len}")

        input_ids = torch.randint(0, tokenizer.vocab_size, (1, input_len), device=device)

        arch = (model.config.architectures[0].lower() if model.config.architectures else "")
        if "llama" in arch:
            past_key_values = create_llama_past_key_values(model.config, kv_len, device)
        elif "phi" in arch:
            past_key_values = create_phimoe_past_key_values(model.config, kv_len, device)
        else:
            kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            past_key_values = (
                torch.zeros((1, kv_heads, kv_len, head_dim), device=device, dtype=dtype),
                torch.zeros((1, kv_heads, kv_len, head_dim), device=device, dtype=dtype),
            )

        # warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_ids, past_key_values=past_key_values, use_cache=True)
            if use_xla_timers:
                torch_xla.sync()
            elif dev_kind == "cuda":
                torch.cuda.synchronize()

        # measure with XLA trace (device-time, exclusive)
        results_us = {}
        if use_xla_timers:
            trace_dir = "/tmp/xla_trace"
            os.makedirs(trace_dir, exist_ok=True)
            for fn in os.listdir(trace_dir):
                try: os.remove(os.path.join(trace_dir, fn))
                except: pass

            if kv_len == 0:
                pkv = None
            elif kv_len > 0:
                if "llama" in arch:
                    pkv = create_llama_past_key_values(model.config, kv_len, device)
                elif "phi" in arch:
                    pkv = create_phimoe_past_key_values(model.config, kv_len, device)
                else:
                    pkv = past_key_values

            # Original notebook flow (works on torch_xla 2.8 which re-exposes
            # start_trace/stop_trace). _load_events handles both the
            # Chrome-trace JSON output and the xplane.pb format.
            xp.start_trace(log_dir=trace_dir)
            for _ in range(repeat):
                if kv_len == 0:
                    pkv = None
                elif kv_len > 0:
                    if "llama" in arch:
                        pkv = create_llama_past_key_values(model.config, kv_len, device)
                    elif "phi" in arch:
                        pkv = create_phimoe_past_key_values(model.config, kv_len, device)
                    else:
                        pkv = past_key_values

                with torch.no_grad():
                    _ = model(input_ids, past_key_values=pkv if kv_len>0 else None, use_cache=True)
            torch_xla.sync()
            xp.stop_trace()

            events = _load_events("/tmp/xla_trace")
            exclusive_us = _exclusive_total(events)  # microseconds across the whole trace
            # average per iteration
            for k, v in exclusive_us.items():
                results_us[k] = float(v) / max(1, repeat) / num_layers

        else:
            # CUDA or CPU fallback: keep original sync, but no per-tag times available here.
            # (You can add a CUDA profiler path if needed.)
            pass

        # map to CSV naming convention
        results = _map_tags_to_results(results_us, arch=arch)


        # components list (same as original)
        if "llama" in arch:
            block_comps = ["input_layernorm","q_proj","k_proj","v_proj","rope","attn","o_proj",
                     "post_layernorm","gate_proj","up_proj","act_fn","down_proj"]
        elif "phi" in arch:
            block_comps = ["input_layernorm","q_proj","k_proj","v_proj","rope","attn","o_proj",
                     "post_layernorm","gate","sparsemixer","expert.w1","expert.w2","expert.w3"]
        else:  # OPT-like
            block_comps = ["input_layernorm","q_proj","k_proj","v_proj","qk_matmul","softmax","sv_matmul","o_proj",
                     "post_layernorm","fc1","act_fn","fc2"]

        common_comps = ["embedding","final_layernorm","lm_head"]
        estimated_latency = 0
        block_latecny = 0
        # buffer rows
        for comp in block_comps:
            if comp in results:
                block_latecny += int(max(results[comp], 0.0) * 1000.0)
        for comp in common_comps:
            if comp in results:
                estimated_latency += int(max(results[comp], 0.0) * 1000.0)
        estimated_latency += block_latecny * original_num_layers / num_layers
        if verbose:
            print(f"input_len: {input_len}, kv_len: {kv_len}, estimated:{estimated_latency}")

        for comp in block_comps:
            if comp in results:
                result_row = {
                    "hardware": hardware,
                    "model": model_name,
                    "layer_name": comp,
                    "input": input_len,
                    "kv_cache": kv_len,
                    "latency(ns)": int(max(results[comp], 0.0) * 1000.0),  # us -> ns
                }
                buffered_rows.append(result_row)

            if verbose:
                print(result_row)

        for comp in common_comps:
            if comp in results:
                result_row = {
                    "hardware": hardware,
                    "model": model_name,
                    "layer_name": comp,
                    "input": input_len,
                    "kv_cache": kv_len,
                    "latency(ns)": int(max(results[comp], 0.0) * 1000.0),  # us -> ns
                }
                buffered_rows.append(result_row)

            if verbose:
                print(result_row)

        since_last_flush += 1

        # batch flush
        if since_last_flush >= max(1, flush_every):
            with open(out_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerows(buffered_rows)
                f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
            try: os.sync()
            except Exception: pass
            if progress:
                outer.set_postfix_str(f"in={input_len}, kv={kv_len}, flushed {since_last_flush} cfgs")
            buffered_rows.clear()
            since_last_flush = 0

        if dev_kind == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # final flush
    if buffered_rows:
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerows(buffered_rows)
            f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass
        try: os.sync()
        except Exception: pass
        buffered_rows.clear()

    print("[OK] CSV stored: ", out_path)
    return out_path
# === Validation utilities (ported from validation.py, for TPU) ===
import os, csv, time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def _csv_path(out_dir, hardware, model_name):
    # Build CSV output path using hardware and model name
    return os.path.join(out_dir, hardware, f"{model_name.replace('/', '_').replace(':','-')}.csv")

def measure_generation_latency(
    model,
    input_length=10,
    output_length=5,
    warmup=3,
    repeat=5,
    verbose=False
):
    device = next(model.parameters()).device
    # Dummy input (fixed length)
    input_ids = torch.randint(0, model.config.vocab_size, (1, input_length), device=device)

    # ---- Warmup (compile/cache): prefill + decode loop ----
    for _ in range(warmup):
        with torch.no_grad():
            # Prefill
            out = model(input_ids=input_ids, use_cache=True)
        torch_xla.sync()
        pkv = out.past_key_values
        last = input_ids[:, -1:]  # last token as next decode input
        # Decode warmup for full output_length (compile shapes for growing KV)
        for _ in range(max(0, output_length)):
            with torch.no_grad():
                out = model(input_ids=last, past_key_values=pkv, use_cache=True)
            torch_xla.sync()
            pkv = out.past_key_values
            # Greedy next token (stay on device; no .item())
            logits = out.logits[:, -1, :]
            last = torch.argmax(logits, dim=-1, keepdim=True)

    # ---- Measurement: sum of prefill + each decode step ----
    total_ns = []

    # Prefill timing
    for _ in range(repeat):
        iter_ns = 0
        torch_xla.sync()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=None, use_cache=True)
        torch_xla.sync()  # wait device to finish this prefill
        iter_ns += (time.perf_counter_ns() - t0)

        # Decode timing (output_length steps)
        pkv = out.past_key_values
        last = input_ids[:, -1:]
        for _ in range(1, max(1, output_length)):
            torch_xla.sync()
            t1 = time.perf_counter_ns()
            with torch.no_grad():
                out = model(input_ids=last, past_key_values=pkv, use_cache=True)
            torch_xla.sync()   # wait device to finish this decode step
            iter_ns += (time.perf_counter_ns() - t1)

            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            last = torch.argmax(logits, dim=-1, keepdim=True)
        total_ns.append(iter_ns)
    m_total_ns = statistics.median(total_ns)
    dt_ms = m_total_ns / 1e6

    if verbose:
        print(f"[measure(step)] input={input_length}, output={output_length} -> {dt_ms:.2f} ms")
    return dt_ms

# 2) Estimate total latency from per-op CSV
def estimate_total_latency(
    out_dir,
    hardware,
    model_name="meta-llama/Llama-3.1-8B",
    num_layers=32,
    input_length=10,
    output_length=5,
    csv_path=None,
    verbose=False
):
    config = AutoConfig.from_pretrained(model_name)

    # Load CSV: (input, kv) -> {block_name -> latency_ns}
    latency_db = defaultdict(dict)
    csv_path = csv_path or _csv_path(out_dir, hardware, f"{model_name}")
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["input"]), int(row["kv_cache"]))
            latency_db[key][row["layer_name"]] = int(row["latency(ns)"])

    # Component list per architecture
    arch = model_name.lower()
    if "llama" in arch:
        comps = ["input_layernorm","q_proj","k_proj","v_proj","rope","attn","o_proj",
                 "post_layernorm","gate_proj","up_proj","act_fn","down_proj"]
    elif "phi" in arch:
        comps = ["input_layernorm","q_proj","k_proj","v_proj","rope","attn","o_proj",
                 "post_layernorm","gate","sparsemixer"]
    else:  # OPT-like
        comps = ["input_layernorm","q_proj","k_proj","v_proj","qk_matmul","softmax","sv_matmul",
                 "o_proj","post_layernorm","fc1","act_fn","fc2"]

    total_ns = 0


    # Prefill: (input_len, kv=0)
    pre_key = (input_length, 0)
    if pre_key not in latency_db:
        raise ValueError(f"Missing latency for input={input_length}, kv=0 in {csv_path}")
    pre = latency_db[pre_key]
    total_ns += num_layers * sum(pre.get(c, 0) for c in comps)
    # One-time parts (embedding / final LN / LM head)
    for c in ["embedding","final_layernorm","lm_head"]:
        total_ns += pre.get(c, 0)

    # Phi-MoE expert weighting (keep original logic)
    if "phi" in arch:
        n_experts = getattr(config, "num_local_experts", 1)
        moe_comps = ["expert.w1","expert.w2","act_fn","expert.w3"]
        total_ns += num_layers * n_experts * sum(pre.get(c, 0) for c in moe_comps)

    # Decode: (input=1, kv = input_length + i)
    for i in range(max(0, output_length - 1)):
        kv = input_length + i
        dec_key = (1, kv)
        if dec_key not in latency_db:
            raise ValueError(f"Missing latency for input=1, kv={kv} in {csv_path}")
        dec = latency_db[dec_key]
        total_ns += num_layers * sum(dec.get(c, 0) for c in comps)
        # Per-token head/others (keep original logic)
        total_ns += dec.get("lm_head", 0)
        if "phi" in arch:
            total_ns += num_layers * n_experts * sum(dec.get(c, 0) for c in moe_comps)
        for c in ["embedding","final_layernorm","lm_head"]:
            total_ns += dec.get(c, 0)
        total_ns += 10e6 * i # 10ms of compile time at each decode

    if verbose:
        print(f"[estimate] {total_ns/1e6:.3f} ms (from CSV)")
    return total_ns / 1e6  # ms

# 3) Validation loop: measured vs. estimated
def validate_latency_estimation(
    out_dir,
    hardware,
    model_name="meta-llama/Llama-3.1-8B",
    num_layers=None,
    input_lengths=(10, 20),
    output_lengths=(2, 4),
    warmup=3,
    repeat=10,
    verbose=False,
    csv_path=None
):

    # Load model/tokenizer (auto-select device/precision)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).eval()

    # Optionally truncate number of layers
    if num_layers is not None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = model.model.layers[:num_layers]
        elif hasattr(model, "model") and hasattr(model.model, "decoder"):
            model.model.decoder.layers = model.model.decoder.layers[:num_layers]
    model.to(torch.device("xla"))

    rows = []
    for il in input_lengths:
        for ol in output_lengths:
            il = 1 if il <= 0 else il
            ol = 1 if ol <= 0 else ol
            m_meas = measure_generation_latency(model, il, ol, warmup=warmup, repeat=repeat, verbose=False)
            m_est = estimate_total_latency(out_dir, hardware, model_name, num_layers or model.config.num_hidden_layers,
                                                   il, ol, csv_path=csv_path, verbose=False)
            err = (m_est - m_meas) / max(1e-9, m_meas) * 100.0
            rows.append((il, ol, m_meas, m_est, err))
            if verbose:
                print(f"[validate] in={il}, out={ol}  measured={m_meas:.2f} ms  estimated={m_est:.2f} ms  Δ={err:+.1f}%")
    return rows

# 4) Apply a global scale factor to the CSV (optional)
def scale_latency_csv(
    out_dir,
    hardware,
    model_name,
    scaling_factor=1.0,
    overwrite=False
):
    src = _csv_path(out_dir, hardware, model_name)
    dst = src if overwrite else src.replace(".csv", f".scaled_{scaling_factor:.2f}.csv")

    with open(src, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames.copy()
        rows = []
        if not overwrite and "scaled_latency(ns)" not in fieldnames:
            fieldnames.append("scaled_latency(ns)")
        for row in reader:
            val = int(row["latency(ns)"])
            if overwrite:
                row["latency(ns)"] = int(val * scaling_factor)
            else:
                row["scaled_latency(ns)"] = int(val * scaling_factor)
            rows.append(row)

    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote: {dst}")
    return dst

# === Patch: validate → fit scaling → write scaled CSV ===
from statistics import median

def _compute_scaling_factor(rows, method="median"):
    ratios = [meas / est for (_, _, meas, est, _) in rows if est and est > 0]
    if not ratios:
        return 1.0
    if method == "mean":
        return float(sum(ratios) / len(ratios))
    return float(median(ratios))  # default: median is more robust to outliers


def _compute_error_stats(rows):
    errs = [err for (*_, err) in rows if err is not None]
    if not errs:
        return {
            "n": 0,
            "mean_signed_pct_err": 0.0,
            "mean_abs_pct_err": 0.0,
            "median_abs_pct_err": 0.0,
        }
    abs_errs = [abs(e) for e in errs]
    return {
        "n": len(errs),
        "mean_signed_pct_err": float(sum(errs) / len(errs)),
        "mean_abs_pct_err": float(sum(abs_errs) / len(abs_errs)),
        "median_abs_pct_err": float(median(abs_errs)),
    }


def validate_and_scale(
    out_dir,
    hardware,
    model_name,
    num_layers=None,
    input_lengths=(1,),
    output_lengths=(1, 2),
    warmup=5,
    repeat=3,
    verbose=True,
    method="median",
    overwrite=False,      # If True, overwrite original 'latency(ns)'; else add 'scaled_latency(ns)'
    csv_path=None,
    return_error_stats=False,  # If True, return a 4th value: error stats dict
):
    # 1) Validate: measured vs. estimated
    rows = validate_latency_estimation(
        out_dir=out_dir,
        hardware=hardware,
        model_name=model_name,
        num_layers=num_layers,
        input_lengths=input_lengths,
        output_lengths=output_lengths,
        warmup=warmup,
        repeat=repeat,
        verbose=verbose,
        csv_path=csv_path,
    )

    # 2) Fit scaling factor (measured / estimated)
    sf = _compute_scaling_factor(rows, method=method)
    if verbose:
        print(f"[scale] fitted scaling_factor = {sf:.4f} (method={method})")

    # --- Error stats (before scaling) ---
    err_stats = _compute_error_stats(rows)
    if verbose:
        print(
            "[error] n={n}  mean_abs={ma:.2f}%  median_abs={md:.2f}%  mean_signed={ms:+.2f}%"
            .format(
                n=err_stats["n"],
                ma=err_stats["mean_abs_pct_err"],
                md=err_stats["median_abs_pct_err"],
                ms=err_stats["mean_signed_pct_err"],
            )
        )

    # 3) Apply to CSV
    scaled_path = scale_latency_csv(
        out_dir=out_dir,
        hardware=hardware,
        model_name=model_name,
        scaling_factor=sf,
        overwrite=overwrite,
    )
    if verbose:
        print(f"[scale] wrote scaled CSV: {scaled_path}")

    if return_error_stats:
        return sf, scaled_path, rows, err_stats
    return sf, scaled_path, rows
