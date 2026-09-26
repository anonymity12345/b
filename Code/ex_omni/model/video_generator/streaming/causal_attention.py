from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from ex_omni.model.attention import (
    FLASH_ATTENTION_3,
    FLASH_ATTENTION_2,
    SDPA,
    flash_attention_function,
    resolve_attn_implementation,
)

from ..models.attention_ops import video_flash_attention
from .chunk_policy import full_cache_mask
from .state import DiTStreamingConfig, LayerKVCache


def average_pool_complex_frequencies(
    frequencies: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """Pool a 1-D complex RoPE table for FAR compressed tokens."""
    if factor == 1:
        return frequencies
    if factor <= 0:
        raise ValueError("RoPE pooling factor must be positive.")
    real = F.avg_pool1d(
        frequencies.real.transpose(0, 1).unsqueeze(0),
        kernel_size=factor,
        stride=factor,
    ).squeeze(0).transpose(0, 1)
    imag = F.avg_pool1d(
        frequencies.imag.transpose(0, 1).unsqueeze(0),
        kernel_size=factor,
        stride=factor,
    ).squeeze(0).transpose(0, 1)
    magnitude = torch.sqrt(real.square() + imag.square()).clamp_min(1e-12)
    return torch.complex(real / magnitude, imag / magnitude)


def make_3d_positions(
    frame_offset: int,
    grid_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    frames, height, width = grid_size
    frame_ids = torch.arange(
        frame_offset,
        frame_offset + frames,
        device=device,
        dtype=torch.long,
    )
    height_ids = torch.arange(height, device=device, dtype=torch.long)
    width_ids = torch.arange(width, device=device, dtype=torch.long)
    return torch.stack(
        torch.meshgrid(frame_ids, height_ids, width_ids, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)


def move_reference_sink_rope(
    frame_positions: torch.Tensor,
    current_frame,
    config: DiTStreamingConfig,
) -> torch.Tensor:
    """Move only the reference sink; keep ordinary history at absolute RoPE positions."""
    if frame_positions.numel() == 0:
        return frame_positions
    current_frame = torch.as_tensor(
        current_frame, device=frame_positions.device, dtype=frame_positions.dtype,
    )
    return torch.where(
        frame_positions == 0,
        (current_frame - config.sink_rope_distance).clamp_min(0),
        frame_positions,
    )


def build_3d_rope_freqs(
    base_freqs: Sequence[torch.Tensor],
    positions: torch.Tensor,
    current_frame,
    config: DiTStreamingConfig,
    *,
    compressed: bool = False,
) -> torch.Tensor:
    device_freqs = tuple(
        frequencies.to(device=positions.device) for frequencies in base_freqs
    )
    if compressed:
        scale = config.compression_scale
        device_freqs = (
            average_pool_complex_frequencies(device_freqs[0], scale[0]),
            average_pool_complex_frequencies(device_freqs[1], scale[1]),
            average_pool_complex_frequencies(device_freqs[2], scale[2]),
        )
    temporal = move_reference_sink_rope(positions[:, 0], current_frame, config)
    height = positions[:, 1]
    width = positions[:, 2]
    if not torch.compiler.is_compiling() and positions.numel():
        for axis, indices in ((1, height), (2, width)):
            index = int(indices.max())
            if index >= device_freqs[axis].shape[0]:
                raise ValueError(
                    f"RoPE axis {axis} index {index} exceeds table size "
                    f"{device_freqs[axis].shape[0]}."
                )
    temporal_freqs = device_freqs[0]
    # Evaluate only requested long positions; retain the exact table values
    # for in-range rows even when the same batch also contains long rows.
    dim = temporal_freqs.shape[-1] * 2
    inverse = 1.0 / (10000.0 ** (
        torch.arange(0, dim, 2, device=positions.device).float() / dim
    ))
    phase = temporal.double()[:, None] * inverse.double()[None, :]
    extended = torch.polar(torch.ones_like(phase), phase).to(temporal_freqs.dtype)
    in_range = (temporal >= 0) & (temporal < temporal_freqs.shape[0])
    selected_temporal = torch.where(
        in_range[:, None],
        temporal_freqs[temporal.clamp(min=0, max=temporal_freqs.shape[0] - 1)],
        extended,
    )
    return torch.cat(
        [selected_temporal, device_freqs[1][height], device_freqs[2][width]],
        dim=-1,
    ).unsqueeze(1)


def _append_kept_suffix(previous, current, start):
    """Append only retained rows, avoiding concat followed by index_select."""
    if previous is None or start >= previous.shape[1]:
        offset = start - (0 if previous is None else previous.shape[1])
        return current[:, offset:].contiguous()
    return torch.cat((previous[:, start:].to(current), current), dim=1)


@torch.compiler.disable
def _capture_reference_sink(key, value, positions):
    """Capture frame zero once, outside compiled cache-update graphs."""
    indices = (positions[:, 0] == 0).nonzero().flatten()
    if not indices.numel():
        return None, None, None
    return (
        key.index_select(1, indices).detach(),
        value.index_select(1, indices).detach(),
        positions.index_select(0, indices).detach(),
    )


def update_full_cache(
    cache: Optional[LayerKVCache],
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    config: DiTStreamingConfig,
    selection=None,
) -> LayerKVCache:
    sink_key = cache.sink_key if cache is not None else None
    sink_value = cache.sink_value if cache is not None else None
    sink_positions = cache.sink_positions if cache is not None else None
    if sink_key is None:
        sink_key, sink_value, sink_positions = _capture_reference_sink(key, value, positions)
        if sink_key is None:
            raise ValueError("Reference sink is missing; prefill the reference before generating chunks.")
    if selection is None:
        selection = prepare_cache_update_selection(cache, positions, config)
    # Selection always indexes the original previous + current rows. Keeping
    # frame zero in that index space preserves the contiguous-suffix fast path.
    keep, kept_positions = selection[:2]
    if len(selection) == 3 and selection[2] is not None:
        start = selection[2]
        key = _append_kept_suffix(cache.key if cache is not None else None, key, start)
        value = _append_kept_suffix(cache.value if cache is not None else None, value, start)
    else:
        if cache is not None:
            key = torch.cat([cache.key, key], dim=1)
            value = torch.cat([cache.value, value], dim=1)
        key = key.index_select(1, keep)
        value = value.index_select(1, keep)
    return LayerKVCache(
        key=key.detach(),
        value=value.detach(),
        positions=kept_positions.detach(),
        compressed_key=cache.compressed_key if cache is not None else None,
        compressed_value=cache.compressed_value if cache is not None else None,
        compressed_positions=cache.compressed_positions if cache is not None else None,
        sink_key=sink_key,
        sink_value=sink_value,
        sink_positions=sink_positions,
    )


def update_compressed_cache(
    cache: Optional[LayerKVCache],
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    config: DiTStreamingConfig,
    selection=None,
) -> LayerKVCache:
    """Append compressed history without duplicating or modifying the sink."""
    if selection is None:
        selection = prepare_cache_update_selection(cache, positions, config, compressed=True)
    bounded, kept_positions = selection[:2]
    if len(selection) == 3 and selection[2] is not None:
        start = selection[2]
        compressed_key = _append_kept_suffix(
            cache.compressed_key if cache is not None else None, key, start,
        )
        compressed_value = _append_kept_suffix(
            cache.compressed_value if cache is not None else None, value, start,
        )
    else:
        compressed_key, compressed_value = key, value
        if cache is not None and cache.compressed_key is not None:
            compressed_key = torch.cat([cache.compressed_key.to(key), key], dim=1)
            compressed_value = torch.cat([cache.compressed_value.to(value), value], dim=1)
        compressed_key = compressed_key.index_select(1, bounded)
        compressed_value = compressed_value.index_select(1, bounded)
    return LayerKVCache(
        key=cache.key if cache is not None else key[:, :0],
        value=cache.value if cache is not None else value[:, :0],
        positions=cache.positions if cache is not None else positions[:0],
        compressed_key=compressed_key.detach(),
        compressed_value=compressed_value.detach(),
        compressed_positions=kept_positions.detach(),
        sink_key=cache.sink_key if cache is not None else None,
        sink_value=cache.sink_value if cache is not None else None,
        sink_positions=cache.sink_positions if cache is not None else None,
    )


@torch.compiler.disable
def visible_compressed_cache(
    cache: Optional[LayerKVCache],
    current_frame: int,
    config: DiTStreamingConfig,
):
    """Return compressed frames only after their full-resolution copy expires."""
    if (
        cache is None
        or cache.compressed_key is None
        or cache.compressed_positions is None
        or not cache.compressed_positions.numel()
    ):
        return None
    visible = ~full_cache_mask(
        cache.compressed_positions[:, 0], current_frame, config
    )
    visible &= cache.compressed_positions[:, 0] != 0
    indices = visible.nonzero().flatten()
    if not indices.numel():
        return None
    return (
        cache.compressed_key.index_select(1, indices),
        cache.compressed_value.index_select(1, indices),
        cache.compressed_positions.index_select(0, indices),
    )


@torch.compiler.disable
def prepare_cache_update_selection(cache, positions, config, *, compressed=False, contiguous_suffix=False):
    previous = None if cache is None else (
        cache.compressed_positions if compressed else cache.positions)
    combined = torch.cat((previous.to(positions), positions), dim=0) if previous is not None else positions
    newest = combined[:, 0].max() if combined.shape[0] else 0
    mask = (combined[:, 0] >= newest - config.max_compressed_latent_frames + 1
            if compressed else full_cache_mask(combined[:, 0], newest, config))
    mask &= combined[:, 0] != 0
    indices = mask.nonzero().flatten()
    kept_positions = combined.index_select(0, indices)
    if contiguous_suffix:
        start = int(indices[0]) if indices.numel() else combined.shape[0]
        # nonzero emits sorted unique indices. This count proves there are no
        # holes in the suffix; arbitrary/nonmonotone positions safely fall back.
        if indices.numel() != combined.shape[0] - start:
            start = None
        return indices, kept_positions, start
    return indices, kept_positions


@torch.compiler.disable
def prepare_streaming_attention_metadata(
    cache, query_positions, current_positions, base_freqs, config, *, compressed=False,
    memo=None, memo_key=None,
):
    """Position math is identical across DiT layers; compute it once per forward.

    Integer indices also avoid the device synchronization of boolean indexing
    for every layer's K/V. No model activations or cross-request state is cached.
    Keep this once-per-forward, data-dependent cache selection outside compiled
    graphs. Its nonzero outputs and aliased sequence-parallel position views
    trigger unstable symbolic guards; the expensive DiT layers remain compiled.
    """
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    current_frame = current_positions[:, 0].amax()
    query_freqs = build_3d_rope_freqs(
        base_freqs, query_positions, current_frame, config, compressed=compressed,
    )
    current_freqs = build_3d_rope_freqs(
        base_freqs, current_positions, current_frame, config, compressed=compressed,
    ) if query_positions is not current_positions else query_freqs
    full_indices = compressed_indices = None
    frequencies = []
    if (config.reference_sink_attention and cache is not None and cache.sink_key is not None
            and not bool((current_positions[:, 0] == 0).any())):
        frequencies.append(build_3d_rope_freqs(
            base_freqs, cache.sink_positions.to(current_positions), current_frame, config,
        ))
    if cache is not None and cache.compressed_positions is not None:
        compressed_visible = ~full_cache_mask(
            cache.compressed_positions[:, 0], current_frame, config,
        )
        compressed_visible &= cache.compressed_positions[:, 0] != 0
        compressed_indices = compressed_visible.nonzero().flatten()
        if compressed_indices.numel():
            frequencies.append(build_3d_rope_freqs(
                base_freqs, cache.compressed_positions.index_select(0, compressed_indices).to(current_positions),
                current_frame, config, compressed=True,
            ))
    if cache is not None and cache.positions.numel():
        full_visible = full_cache_mask(cache.positions[:, 0], current_frame, config)
        full_visible &= cache.positions[:, 0] != 0
        full_indices = full_visible.nonzero().flatten()
        if full_indices.numel():
            frequencies.append(build_3d_rope_freqs(
                base_freqs, cache.positions.index_select(0, full_indices).to(current_positions),
                current_frame, config,
            ))
    frequencies.append(current_freqs)
    result = query_freqs, torch.cat(frequencies, dim=0), full_indices, compressed_indices
    if memo is not None:
        memo.clear()
        memo[memo_key] = result
    return result


@torch.compiler.disable
def streaming_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    attn_implementation: str = FLASH_ATTENTION_2,
    cache_seqlens=None,
) -> torch.Tensor:
    backend = (
        attn_implementation
        if attn_implementation
        in {FLASH_ATTENTION_3, FLASH_ATTENTION_2, SDPA, "eager"}
        else resolve_attn_implementation(
            attn_implementation, warn_on_fallback=False
        )
    )
    if (
        backend in {FLASH_ATTENTION_3, FLASH_ATTENTION_2}
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
    ):
        query = rearrange(query, "b s (n d) -> b s n d", n=num_heads)
        key = rearrange(key, "b s (n d) -> b s n d", n=num_heads)
        value = rearrange(value, "b s (n d) -> b s n d", n=num_heads)
        if cache_seqlens is not None:
            if backend != FLASH_ATTENTION_3:
                raise ValueError("Persistent graph KV buffers require FA3")
            from ex_omni.model.attention import flash_attention_module
            output = flash_attention_module(backend).flash_attn_with_kvcache(
                query, key, value, cache_seqlens=cache_seqlens,
                causal=False, num_splits=1)
        else:
            output = video_flash_attention(query, key, value, backend)
        return rearrange(output, "b s n d -> b s (n d)", n=num_heads)
    query = rearrange(query, "b s (n d) -> b n s d", n=num_heads)
    key = rearrange(key, "b s (n d) -> b n s d", n=num_heads)
    value = rearrange(value, "b s (n d) -> b n s d", n=num_heads)
    output = F.scaled_dot_product_attention(query, key, value)
    return rearrange(output, "b n s d -> b s (n d)", n=num_heads)


def block_causal_mask(
    num_frames: int,
    tokens_per_frame: int,
    chunk_frames: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a boolean block-causal mask with bidirectional attention per chunk."""
    if num_frames <= 0 or tokens_per_frame <= 0 or chunk_frames <= 0:
        raise ValueError("Mask dimensions and chunk_frames must be positive.")
    frame_ids = torch.arange(num_frames, device=device).repeat_interleave(
        tokens_per_frame
    )
    chunk_ids = torch.div(frame_ids, chunk_frames, rounding_mode="floor")
    return chunk_ids[:, None] >= chunk_ids[None, :]
