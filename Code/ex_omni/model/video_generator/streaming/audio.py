from typing import Optional, Tuple

import torch
import torch.nn as nn

from .state import AudioStreamingState


def pack_streaming_audio(
    frame_embeddings: torch.Tensor,
    audio_proj: nn.Module,
    audio_cond_projs: nn.ModuleList,
    state: AudioStreamingState,
) -> Tuple[Optional[torch.Tensor], AudioStreamingState]:
    """Incrementally apply OmniAvatar AudioPack without resetting chunk phase.

    `frame_embeddings` is `[B, T, C]`. For image-to-video sessions the first
    call should include one null/reference frame before the first speech frame.
    """
    if frame_embeddings.ndim != 3:
        raise ValueError(
            "Streaming audio embeddings must have shape [B, T, C], "
            f"got {tuple(frame_embeddings.shape)}."
        )
    input_frames = frame_embeddings.shape[1]
    if not state.initialized:
        if input_frames == 0:
            return None, state
        prefix = frame_embeddings[:, :1].repeat(1, 3, 1)
        frame_embeddings = torch.cat([prefix, frame_embeddings], dim=1)
        state.initialized = True
    if state.pending_frames is not None:
        frame_embeddings = torch.cat(
            [state.pending_frames.to(frame_embeddings), frame_embeddings],
            dim=1,
        )

    complete_frames = frame_embeddings.shape[1] // 4 * 4
    state.pending_frames = frame_embeddings[:, complete_frames:].detach()
    state.consumed_frames += input_frames
    if complete_frames == 0:
        return None, state

    packed_input = frame_embeddings[:, :complete_frames]
    packed_input = packed_input.permute(0, 2, 1)[:, :, :, None, None]
    packed = audio_proj(packed_input)
    conditioned = torch.stack(
        [projection(packed) for projection in audio_cond_projs],
        dim=1,
    )
    return conditioned, state

