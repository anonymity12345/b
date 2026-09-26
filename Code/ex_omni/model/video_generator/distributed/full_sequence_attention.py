"""Full-sequence self-attention with native WORLD-group Ulysses exchange."""

from __future__ import annotations

import torch

from ex_omni.distributed.sequence_parallel import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..models.wan_video_dit import flash_attention, rope_apply


def full_sequence_attention_forward(self, x: torch.Tensor, freqs: torch.Tensor, **kwargs):
    """Exchange sequence shards for whole attention heads, then restore shards."""
    if kwargs.get("streaming_config") is not None:
        raise ValueError("full-sequence attention cannot consume streaming KV caches")
    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    local_tokens = x.shape[1]
    if freqs.shape[0] != local_tokens * world_size:
        raise ValueError(
            "full-sequence sequence parallelism requires equal token shards; "
            f"got {freqs.shape[0]} positions across {world_size} ranks"
        )
    local_freqs = freqs[rank * local_tokens : (rank + 1) * local_tokens]
    q = rope_apply(self.norm_q(self.q(x)), local_freqs, self.num_heads)
    k = rope_apply(self.norm_k(self.k(x)), local_freqs, self.num_heads)
    v = self.v(x)
    group = get_sp_group()
    qkv = group.sequence_to_heads(torch.stack((q, k, v), dim=2), num_heads=self.num_heads)
    q, k, v = qkv.unbind(dim=2)
    local_heads = q.shape[-1] // self.head_dim
    attended = flash_attention(q, k, v, num_heads=local_heads)
    restored = group.heads_to_sequence(
        attended, num_heads=self.num_heads, head_dim=self.head_dim
    )
    return self.o(restored)
