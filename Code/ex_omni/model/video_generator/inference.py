"""Explicit-config video inference backend."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import random
from typing import Any, Mapping

from ex_omni.config import normalize_video_config


def _import_object(path: str):
    module_name, separator, object_name = path.rpartition(".")
    if not separator:
        raise ValueError("runtime_class must be a fully-qualified name")
    return getattr(importlib.import_module(module_name), object_name)


def _initialize_distributed(config: Mapping[str, Any]) -> None:
    sequence_parallel = int(config.get("sequence_parallel", config.get("sp_size", 1)))
    explicit = bool(config.get("distributed", False))
    if sequence_parallel <= 1 and not explicit:
        return
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("distributed OmniAvatar inference requires CUDA")
    if not dist.is_initialized():
        dist.init_process_group(
            backend=str(config.get("distributed_backend", "nccl")),
            init_method=str(config.get("distributed_init_method", "env://")),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", config.get("local_rank", 0)))
    torch.cuda.set_device(local_rank)


class WanInferenceBackend:
    """Configurable video inference backend.

    ``video.runtime_class`` may specify a compatible runtime class.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = normalize_video_config(config)
        self._runtime = None

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def load(self):
        if self._runtime is None:
            _initialize_distributed(self.config)
            runtime_path = self.config.get(
                "runtime_class",
                "ex_omni.model.video_generator.native_runtime.NativeWanRuntime",
            )
            runtime_class = _import_object(str(runtime_path))
            self._runtime = runtime_class(self.config)
        return self._runtime

    def generate(self, request):
        mode = request.mode
        if self._runtime is not None and mode != self.config.get("mode", "full_sequence"):
            raise ValueError("a loaded video backend cannot switch full-sequence/streaming mode")
        self.config = normalize_video_config(
            self.config,
            mode=mode,
            overrides=request.overrides,
        )
        if mode == "streaming":
            checkpoint = self.config.get("lora_checkpoint")
            if not checkpoint:
                raise ValueError("video.lora_checkpoint is required for Streaming mode")
        runtime = self.load()
        if mode == "streaming":
            method = getattr(runtime, "generate_streaming", None)
            if method is None:
                raise RuntimeError("configured runtime does not support Streaming causal inference")
        else:
            method = getattr(runtime, "generate_full_sequence", None)
            if method is None:
                method = getattr(runtime, "generate", None)
        if method is None:
            raise TypeError("runtime must implement generate_full_sequence/generate_streaming")
        seed = int(request.seed if request.seed is not None else self.config.get("seed", 42))
        random.seed(seed)
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return method(request)

    def start_stream(
        self,
        *,
        prompt: str,
        ref_image,
        seed: int | None = None,
        overrides: Mapping[str, Any] | None = None,
    ):
        if str(self.config.get("mode", "full_sequence")) != "streaming":
            raise ValueError("incremental video streaming requires Streaming mode")
        checkpoint = self.config.get("lora_checkpoint")
        if not checkpoint:
            raise ValueError("video.lora_checkpoint is required for Streaming mode")
        runtime = self.load()
        method = getattr(runtime, "start_streaming_stream", None)
        if method is None:
            raise RuntimeError("configured runtime does not support Streaming streaming")
        seed = int(seed if seed is not None else self.config.get("seed", 42))
        random.seed(seed)
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return method(
            prompt=prompt,
            ref_image=ref_image,
            seed=seed,
            overrides=dict(overrides or {}),
        )
