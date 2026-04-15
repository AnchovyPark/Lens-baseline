"""flash_attn stub — SDPA-based drop-in replacement on TPU / non-CUDA backends.

The authors' attention_profiler.py hard-imports:
    from flash_attn import flash_attn_varlen_func

On TPU we install this stub package ahead of the real flash_attn on sys.path so
the import succeeds. The single function we expose reshapes the varlen inputs
(which, in the authors' usage, always have equal per-sample lengths) back to a
dense batched form and routes them through `torch.nn.functional.scaled_dot_product_attention`.

This follows the CAL 2025 paper's stated methodology ("replacing CUDA APIs with
XLA APIs for TPU compatibility"). flash_attn is fundamentally a CUDA kernel
package — SDPA is the PyTorch-native equivalent that XLA compiles through.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    q,                      # [total_q, num_heads, head_dim]
    k,                      # [total_k, num_kv_heads, head_dim]
    v,                      # [total_k, num_kv_heads, head_dim]
    cu_seqlens_q,           # [B+1] int32
    cu_seqlens_k,           # [B+1] int32
    max_seqlen_q,           # int
    max_seqlen_k,           # int
    dropout_p=0.0,
    causal=False,
    *args,
    **kwargs,
):
    """Drop-in for flash_attn.flash_attn_varlen_func using PyTorch SDPA.

    The authors' _build_varlen_qkv in attention_profiler.py:45-55 distributes
    q_len and kv_len EQUALLY across batch_size samples, so after the cu_seqlens
    packing the inputs are effectively a regular batched tensor. We simply
    reshape back to [B, H, L, D] and call SDPA.
    """
    batch_size = int(cu_seqlens_q.numel() - 1)
    Lq = int(max_seqlen_q)
    Lk = int(max_seqlen_k)

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]

    # Repack: [total, H, D] -> [B, L, H, D] -> [B, H, L, D]
    q_b = q.reshape(batch_size, Lq, num_heads, head_dim).transpose(1, 2).contiguous()
    k_b = k.reshape(batch_size, Lk, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    v_b = v.reshape(batch_size, Lk, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    # GQA expansion: scaled_dot_product_attention supports enable_gqa in recent
    # PyTorch, but to be compatible with older builds we expand K/V explicitly.
    if num_heads != num_kv_heads:
        repeat = num_heads // num_kv_heads
        k_b = k_b.repeat_interleave(repeat, dim=1)
        v_b = v_b.repeat_interleave(repeat, dim=1)

    # For decode (Lq < Lk), `is_causal=True` with SDPA will apply the mask only
    # over the trailing Lq rows — which matches FA2 semantics.
    out_b = F.scaled_dot_product_attention(
        q_b, k_b, v_b,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=bool(causal),
    )

    # Unpack: [B, H, Lq, D] -> [B, Lq, H, D] -> [total_q, H, D]
    out = out_b.transpose(1, 2).contiguous().reshape(batch_size * Lq, num_heads, head_dim)
    return out


__all__ = ["flash_attn_varlen_func"]
__version__ = "xla-stub"
