"""flash-attn-compatible wrappers around vLLM's bundled FA2/FA3 kernels."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import torch

from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    is_fa_version_supported,
)


def _cumulative_lengths(lengths: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(1, dtype=torch.int32, device=lengths.device)
    return torch.cat([zero, lengths.to(dtype=torch.int32).cumsum(dim=0)])


def _fixed_cumulative_lengths(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        0,
        (batch_size + 1) * sequence_length,
        sequence_length,
        dtype=torch.int32,
        device=device,
    )


def _flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    fa_version: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    **_: Any,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("vLLM FlashAttention expects [batch, tokens, heads, dim]")
    if dropout_p:
        raise ValueError("vLLM FlashAttention compatibility layer is inference-only")
    if return_attn_probs:
        raise ValueError("return_attn_probs is unsupported during inference")
    del deterministic

    batch_size, q_length = q.shape[:2]
    k_length = k.shape[1]
    cu_q = _fixed_cumulative_lengths(batch_size, q_length, q.device)
    cu_k = _fixed_cumulative_lengths(batch_size, k_length, q.device)
    output = flash_attn_varlen_func(
        q.reshape(-1, *q.shape[2:]),
        k.reshape(-1, *k.shape[2:]),
        v.reshape(-1, *v.shape[2:]),
        max_seqlen_q=q_length,
        cu_seqlens_q=cu_q,
        max_seqlen_k=k_length,
        cu_seqlens_k=cu_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=list(window_size),
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        fa_version=fa_version,
    )
    return output.reshape(batch_size, q_length, *output.shape[1:])


def _flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    fa_version: int,
    cache_seqlens: int | torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    **_: Any,
) -> torch.Tensor:
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("vLLM FlashDecoding expects [batch, tokens, heads, dim]")
    batch_size, q_length = q.shape[:2]
    cu_q = _fixed_cumulative_lengths(batch_size, q_length, q.device)
    if cache_seqlens is None or isinstance(cache_seqlens, int):
        cache_length = (
            k_cache.shape[1]
            if cache_seqlens is None
            else cache_seqlens
        )
        if cache_length > k_cache.shape[1]:
            raise ValueError("cache_seqlens exceeds the allocated KV cache")
        packed_k = k_cache[:, :cache_length].reshape(
            -1, *k_cache.shape[2:]
        )
        packed_v = v_cache[:, :cache_length].reshape(
            -1, *v_cache.shape[2:]
        )
        cu_k = _fixed_cumulative_lengths(
            batch_size, cache_length, q.device
        )
        max_k_length = cache_length
    else:
        lengths = cache_seqlens.to(device=q.device, dtype=torch.int32)
        if lengths.numel() != batch_size:
            raise ValueError("cache_seqlens must have one entry per batch item")
        if bool((lengths > k_cache.shape[1]).any()):
            raise ValueError("cache_seqlens exceeds the allocated KV cache")
        packed_k = torch.cat(
            [
                k_cache[index, : int(length)]
                for index, length in enumerate(lengths)
            ]
        )
        packed_v = torch.cat(
            [
                v_cache[index, : int(length)]
                for index, length in enumerate(lengths)
            ]
        )
        cu_k = _cumulative_lengths(lengths)
        max_k_length = int(lengths.max())
    output = flash_attn_varlen_func(
        q.reshape(-1, *q.shape[2:]),
        packed_k,
        packed_v,
        max_seqlen_q=q_length,
        cu_seqlens_q=cu_q,
        max_seqlen_k=max_k_length,
        cu_seqlens_k=cu_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=list(window_size),
        softcap=softcap,
        fa_version=fa_version,
    )
    return output.reshape(batch_size, q_length, *output.shape[1:])


@lru_cache(maxsize=2)
def vllm_flash_attention_module(fa_version: int) -> SimpleNamespace:
    if fa_version not in {2, 3}:
        raise ValueError(f"Unsupported vLLM FlashAttention version: {fa_version}")
    if not is_fa_version_supported(fa_version):
        raise RuntimeError(f"vLLM FlashAttention {fa_version} is unavailable")

    def flash_attn_func(q, k, v, **kwargs):
        return _flash_attn_func(q, k, v, fa_version=fa_version, **kwargs)

    def flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs):
        return _flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            fa_version=fa_version,
            **kwargs,
        )

    return SimpleNamespace(
        flash_attn_func=flash_attn_func,
        flash_attn_with_kvcache=flash_attn_with_kvcache,
    )
