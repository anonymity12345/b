"""Distributed entry points for one-world TP + FSDP inference."""

from .context import DistributedContext, current_context, initialize_distributed
from .tp import (
    WorldMPU,
    wrap_dialogue_model_deepspeed,
    wrap_dialogue_model_native_tp,
    wrap_dialogue_model_tensor_parallel,
)

__all__ = [
    "DistributedContext",
    "WorldMPU",
    "current_context",
    "initialize_distributed",
    "wrap_dialogue_model_deepspeed",
    "wrap_dialogue_model_native_tp",
    "wrap_dialogue_model_tensor_parallel",
]
