from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class LayerKVCache:
    """Unrotated full- and compressed-resolution K/V for one DiT layer."""

    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor
    compressed_key: Optional[torch.Tensor] = None
    compressed_value: Optional[torch.Tensor] = None
    compressed_positions: Optional[torch.Tensor] = None
    prepared_attention: object = field(default=None, repr=False)
    current_key_output: object = field(default=None, repr=False)
    sink_key: Optional[torch.Tensor] = None
    sink_value: Optional[torch.Tensor] = None
    sink_positions: Optional[torch.Tensor] = None


@dataclass
class DiTStreamingConfig:
    use_cuda_graphs: bool = False
    reference_sink_attention: bool = True
    chunk_latent_frames: int = 3
    first_chunk_latent_frames: int = 1
    full_chunk_limit: int = 3
    full_patch_size: Tuple[int, int, int] = (1, 2, 2)
    compressed_patch_size: Tuple[int, int, int] = (1, 4, 4)
    max_compressed_latent_frames: int = 96
    sink_rope_distance: int = 30

    def validate(self):
        if type(self.use_cuda_graphs) is not bool:
            raise ValueError("use_cuda_graphs must be a boolean.")
        if type(self.reference_sink_attention) is not bool:
            raise ValueError("reference_sink_attention must be a boolean.")
        if self.sink_rope_distance <= 0:
            raise ValueError("sink_rope_distance must be positive.")
        if self.chunk_latent_frames != 3:
            raise ValueError("Streaming streaming requires three latent frames per chunk.")
        if self.first_chunk_latent_frames != 1:
            raise ValueError("Streaming streaming requires one reference-only latent frame.")
        if self.full_chunk_limit <= 0:
            raise ValueError("full_chunk_limit must be positive.")
        if len(self.full_patch_size) != 3 or len(self.compressed_patch_size) != 3:
            raise ValueError("Streaming patch sizes must contain three dimensions.")
        if any(size <= 0 for size in self.full_patch_size):
            raise ValueError("full_patch_size must be positive.")
        if any(size <= 0 for size in self.compressed_patch_size):
            raise ValueError("compressed_patch_size must be positive.")
        if self.compressed_patch_size[0] != self.full_patch_size[0]:
            raise ValueError("FAR compression supports spatial compression only.")
        if any(
            compressed % full
            for full, compressed in zip(
                self.full_patch_size, self.compressed_patch_size
            )
        ):
            raise ValueError(
                "compressed_patch_size must be divisible by full_patch_size."
            )
        if self.max_compressed_latent_frames <= 0:
            raise ValueError("max_compressed_latent_frames must be positive.")

    @property
    def compression_scale(self) -> Tuple[int, int, int]:
        return tuple(
            compressed // full
            for full, compressed in zip(
                self.full_patch_size, self.compressed_patch_size
            )
        )


@dataclass
class DiTStreamingState:
    layer_caches: List[Optional[LayerKVCache]] = field(default_factory=list)
    global_frame_offset: int = 0
    attention_metadata_cache: dict = field(default_factory=dict, repr=False)
    text_context_cache: object = field(default=None, repr=False)

    def ensure_layers(self, num_layers: int):
        if not self.layer_caches:
            self.layer_caches = [None] * num_layers
        elif len(self.layer_caches) != num_layers:
            raise ValueError(
                f"Expected {num_layers} layer caches, got {len(self.layer_caches)}."
            )

    def reset(self):
        self.layer_caches = [None] * len(self.layer_caches)
        self.global_frame_offset = 0
        self.attention_metadata_cache.clear()
        self.text_context_cache = None


@dataclass
class AudioStreamingState:
    pending_frames: Optional[torch.Tensor] = None
    initialized: bool = False
    consumed_frames: int = 0

    def reset(self):
        self.pending_frames = None
        self.initialized = False
        self.consumed_frames = 0


@dataclass
class VAEStreamingState:
    feature_cache: List[Optional[torch.Tensor]]
    decoded_latent_frames: int = 0
