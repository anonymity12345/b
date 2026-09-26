"""Inference-only compile boundary for the standalone Hopper FA3 extension."""
import os
from pathlib import Path
import torch

from ex_omni.model.attention import FLASH_ATTENTION_3, flash_attention_function


@torch.library.custom_op('ex_omni::flash_attention_3', mutates_args=())
def _flash_attention_3(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return flash_attention_function(FLASH_ATTENTION_3)(query, key, value, causal=False)


@_flash_attention_3.register_fake
def _flash_attention_3_fake(query, key, value):
    return torch.empty_like(query)


_captured_shapes = set()


def video_flash_attention(query, key, value, implementation):
    # Optional tensor capture for attention-kernel diagnostics.
    capture = os.environ.get("EX_OMNI_CAPTURE_ATTENTION")
    if capture and int(os.environ.get("RANK", "0")) == 0:
        shape = (query.shape[1], key.shape[1])
        if shape not in _captured_shapes and len(_captured_shapes) < 5:
            _captured_shapes.add(shape)
            directory = Path(capture)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(dict(q=query.detach().cpu(), k=key.detach().cpu(), v=value.detach().cpu()),
                       directory / f"q{shape[0]}-kv{shape[1]}.pt")
    if (implementation == FLASH_ATTENTION_3 and not torch.is_grad_enabled()
            and torch.compiler.is_compiling()):
        return _flash_attention_3(query, key, value)
    return flash_attention_function(implementation)(query, key, value, causal=False)
