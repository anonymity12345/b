import torch

from .state import DiTStreamingConfig


def chunk_id_for_frame(frame: int, config: DiTStreamingConfig) -> int:
    if frame < config.first_chunk_latent_frames:
        return 0
    return 1 + (
        frame - config.first_chunk_latent_frames
    ) // config.chunk_latent_frames


def full_cache_mask(
    frame_positions: torch.Tensor,
    newest_frame,
    config: DiTStreamingConfig,
) -> torch.Tensor:
    """Select the newest FAR chunks at full patch resolution."""
    first = config.first_chunk_latent_frames
    newest_frame = torch.as_tensor(
        newest_frame, device=frame_positions.device, dtype=frame_positions.dtype
    )
    newest_chunk = torch.where(
        newest_frame < first,
        torch.zeros_like(newest_frame),
        1 + torch.div(
            newest_frame - first,
            config.chunk_latent_frames,
            rounding_mode="floor",
        ),
    )
    oldest_chunk = torch.clamp(
        newest_chunk - config.full_chunk_limit + 1, min=0
    )
    chunk_ids = torch.where(
        frame_positions < first,
        torch.zeros_like(frame_positions),
        1 + torch.div(
            frame_positions - first,
            config.chunk_latent_frames,
            rounding_mode="floor",
        ),
    )
    return chunk_ids >= oldest_chunk
