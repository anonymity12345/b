"""Compatibility runtime namespace for the video model.

The original module parsed CLI arguments. The inference package instead sets
``args`` explicitly from
:class:`ex_omni.model.video_generator.native_runtime.NativeWanRuntime`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping
import sys


args: SimpleNamespace | None = None


def set_runtime_config(config: Mapping[str, Any]) -> SimpleNamespace:
    global args
    args = SimpleNamespace(**dict(config))
    dit_module = sys.modules.get(
        "ex_omni.model.video_generator.models.wan_video_dit"
    )
    if dit_module is not None:
        dit_module._attention_backend.cache_clear()
        backend = dit_module._attention_backend()
        from ex_omni.model.attention import (
            FLASH_ATTENTION_2,
            FLASH_ATTENTION_3,
            flash_attention_function,
        )

        if backend in {FLASH_ATTENTION_2, FLASH_ATTENTION_3}:
            flash_attention_function(backend)
    return args


def convert_namespace_to_dict(namespace: SimpleNamespace) -> dict[str, Any]:
    return dict(vars(namespace))
