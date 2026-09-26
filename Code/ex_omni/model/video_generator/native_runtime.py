"""Native full-sequence and streaming video runtimes."""

from __future__ import annotations

from pathlib import Path
import time
import os
from typing import Any, Mapping

from ex_omni.config import normalize_video_config


def _load_state_dict(path: str):
    import torch

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _checkpoint_metadata(path: str) -> dict[str, str]:
    if not str(path).endswith(".safetensors"):
        return {}
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _resolve_streaming_checkpoint_config(
    config: Mapping[str, Any], metadata: Mapping[str, str]
) -> dict[str, Any]:
    """Require a persistent-sink checkpoint for streaming inference."""
    required = {
        "persistent_sink": "true",
        "first_chunk_latent_frames": "1",
        "chunk_latent_frames": "3",
        "far_enabled": "true",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Streaming checkpoint requires {key}={expected}; got {metadata.get(key)!r}. "
                "Use a checkpoint trained with the persistent reference sink."
            )
    metadata_distance = metadata.get("sink_rope_distance")
    if metadata_distance is not None and (
        not metadata_distance.isdecimal() or int(metadata_distance) <= 0
    ):
        raise ValueError("Streaming checkpoint sink_rope_distance must be a positive integer")
    result = dict(config)
    result.setdefault("stream_sink_rope_distance", 30)
    return normalize_video_config(result, mode="streaming")


def _checkpoint_load_summary(state, incompatible) -> dict[str, int]:
    unexpected = set(incompatible.unexpected_keys)
    keys = set(state)
    matched = keys - unexpected
    lora_keys = {
        key for key in keys if ".lora_A." in key or ".lora_B." in key
    }
    audio_keys = {
        key
        for key in keys
        if key.startswith(
            ("audio_proj.", "audio_cond_projs.", "speech_token_adapter.")
        )
    }
    return {
        "total": len(keys),
        "matched": len(matched),
        "lora_total": len(lora_keys),
        "lora_matched": len(lora_keys - unexpected),
        "audio_total": len(audio_keys),
        "audio_matched": len(audio_keys - unexpected),
        "missing": len(incompatible.missing_keys),
        "unexpected": len(unexpected),
    }


def _match_size(sizes, height: int, width: int):
    return min(
        sizes,
        key=lambda size: (
            abs(size[0] / size[1] - height / width),
            abs(max(size) - max(width, height)),
        ),
    )


def _dynamic_full_sequence_window_frames(
    max_tokens: int,
    selected_size: tuple[int, int] | list[int],
) -> int:
    if int(max_tokens) <= 0:
        raise ValueError("video.max_tokens must be positive")
    height, width = (int(selected_size[0]), int(selected_size[1]))
    length = int(int(max_tokens) * 16 * 16 * 4 / height / width)
    return length // 4 * 4 + 1 if length % 4 else length - 3


def _resolve_full_sequence_window_frames(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
    selected_size: tuple[int, int] | list[int],
) -> int:
    window_frames = overrides.get("window_frames", config.get("window_frames"))
    if window_frames is None:
        return _dynamic_full_sequence_window_frames(
            int(overrides.get("max_tokens", config.get("max_tokens", 30000))),
            selected_size,
        )
    if type(window_frames) is not int or window_frames < 5:
        raise ValueError("video.window_frames must be an integer >= 5 or null")
    if (window_frames - 1) % 4 != 0:
        raise ValueError(
            "video.window_frames must satisfy (frames - 1) % 4 == 0"
        )
    return window_frames


class _AsyncVaeDecoder:
    """Serialize stateful VAE decode on a dedicated CUDA device."""

    def __init__(self, vae, *, device) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.vae = vae
        if os.environ.get("EX_OMNI_VAE_CHANNELS_LAST") == "1":
            import torch
            # Keep reference-image encoding unchanged: its latents condition
            # every diffusion chunk, so even small rounding changes can drift.
            torch.nn.utils.convert_conv3d_weight_memory_format(
                self.vae.model.decoder, torch.channels_last_3d
            )
            print("[video] VAE decoder-only Conv3d weights use channels_last_3d", flush=True)
        self.device = device
        self.state = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ex-omni-vae",
        )

    def submit(self, latents, *, first_chunk: bool, expected_frames: int):
        return self._executor.submit(
            self._decode,
            latents,
            first_chunk=first_chunk,
            expected_frames=expected_frames,
        )

    def _decode(self, latents, *, first_chunk: bool, expected_frames: int):
        import torch

        torch.cuda.set_device(self.device)
        started = time.perf_counter()
        vae_started = torch.cuda.Event(enable_timing=True)
        vae_finished = torch.cuda.Event(enable_timing=True)
        vae_started.record()
        local_latents = latents.to(self.device, non_blocking=True)
        frames, self.state = self.vae.decode_streaming(
            local_latents,
            state=self.state,
            device=self.device,
        )
        vae_finished.record()
        vae_finished.synchronize()
        postprocess_started = time.perf_counter()
        frames = (
            (frames.permute(0, 2, 1, 3, 4).float() + 1.0)
            .div(2.0)
            .clamp(0, 1)
        )
        if first_chunk:
            frames = frames[:, 1:]
        frames = frames[0, :expected_frames]
        result = frames.permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
        finished = time.perf_counter()
        return result, {
            "vae_decode_seconds": vae_started.elapsed_time(vae_finished) / 1000.0,
            "frame_postprocess_seconds": finished - postprocess_started,
            "vae_pipeline_seconds": finished - started,
        }

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


class StreamingVideoSession:
    """Persistent 6-unit FIFO driving the causal Streaming session."""

    units_per_chunk = 6
    frames_per_unit = 2

    def __init__(
        self,
        runtime,
        *,
        prompt: str,
        ref_image: Path,
        seed: int | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        import numpy as np

        self.runtime = runtime
        self.prompt = prompt
        self.ref_image = Path(ref_image)
        self.seed = seed
        self.overrides = dict(overrides or {})
        self._fifo = np.empty((0, 16), dtype=np.int64)
        self._closed = False
        self._start_frame = 0
        self._chunk_id = 0
        self._state = runtime._start_streaming_stream(
            prompt=prompt,
            ref_image=self.ref_image,
            seed=seed,
            overrides=self.overrides,
        )
        self.setup_timing_seconds = dict(
            self._state.get("setup_timing_seconds", {})
        )

    @staticmethod
    def _normalize(units):
        import numpy as np

        values = np.asarray(units, dtype=np.int64)
        if values.ndim == 2 and values.shape[0] == 16 and values.shape[1] != 16:
            values = values.T
        if values.ndim != 2 or values.shape[1] != 16:
            raise ValueError(f"stream units must have shape [N,16], got {values.shape}")
        return np.ascontiguousarray(values)

    def _payload(self, frames, timing_seconds, metadata):
        expected = int(metadata["valid_units"]) * self.frames_per_unit
        frames = frames[:expected]
        return {
            "frames": frames,
            "start_frame": int(metadata["start_frame"]),
            "valid_units": int(metadata["valid_units"]),
            "padded_units": self.units_per_chunk
            - int(metadata["valid_units"]),
            "final": bool(metadata["final"]),
            "timing_seconds": timing_seconds,
        }

    def _collect_pending(self, *, block: bool, chunk_callback=None) -> list[dict[str, Any]]:
        outputs = []
        pending = self._state.setdefault("pending_vae_outputs", [])
        while pending:
            item = pending[0]
            future = item["future"]
            if not block and not future.done():
                break
            frames, vae_timing = future.result()
            pending.pop(0)
            timing = dict(item["timing_seconds"])
            timing.update(vae_timing)
            payload = self._payload(frames, timing, item)
            if block and not pending:
                payload["final"] = True
            outputs.append(payload)
            if chunk_callback is not None:
                chunk_callback(payload)
        return outputs

    def _run(self, units, *, valid_units: int, final: bool):
        result = self.runtime._process_streaming_units(
            self._state,
            units,
            chunk_id=self._chunk_id,
            valid_units=valid_units,
        )
        expected = valid_units * self.frames_per_unit
        metadata = {
            "start_frame": self._start_frame,
            "valid_units": valid_units,
            "final": final,
        }
        self._start_frame += expected
        self._chunk_id += 1
        if isinstance(result, Mapping) and "decode_future" in result:
            self._state.setdefault("pending_vae_outputs", []).append(
                {
                    **metadata,
                    "future": result["decode_future"],
                    "timing_seconds": dict(result.get("timing_seconds", {})),
                }
            )
            return None
        if isinstance(result, tuple):
            frames, timing_seconds = result
        else:
            frames, timing_seconds = result, {}
        return self._payload(frames, timing_seconds, metadata)

    def push_units(self, units) -> list[dict[str, Any]]:
        import numpy as np

        if self._closed:
            raise RuntimeError("streaming video session is already finished")
        values = self._normalize(units)
        if not len(values):
            return []
        self._fifo = np.concatenate([self._fifo, values], axis=0)
        chunks = []
        while len(self._fifo) >= self.units_per_chunk:
            current, self._fifo = (
                self._fifo[: self.units_per_chunk],
                self._fifo[self.units_per_chunk :],
            )
            payload = self._run(
                current,
                valid_units=self.units_per_chunk,
                final=False,
            )
            if payload is not None:
                chunks.append(payload)
        chunks.extend(self._collect_pending(block=False))
        return chunks

    def finish(self, chunk_callback=None) -> list[dict[str, Any]]:
        import numpy as np

        if self._closed:
            return []
        self._closed = True
        chunks = []
        if len(self._fifo):
            valid = len(self._fifo)
            padded = np.concatenate(
                [
                    self._fifo,
                    np.zeros((self.units_per_chunk - valid, 16), dtype=np.int64),
                ],
                axis=0,
            )
            self._fifo = np.empty((0, 16), dtype=np.int64)
            payload = self._run(padded, valid_units=valid, final=True)
            if payload is not None:
                chunks.append(payload)
                if chunk_callback is not None:
                    chunk_callback(payload)
        chunks.extend(self._collect_pending(block=True, chunk_callback=chunk_callback))
        if chunks:
            chunks[-1]["final"] = True
        decoder = self._state.get("async_vae_decoder")
        if decoder is not None:
            decoder.shutdown(wait=True)
        return chunks

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        decoder = self._state.get("async_vae_decoder")
        if decoder is not None:
            decoder.shutdown(wait=False)


class NativeWanRuntime:
    """Full-sequence video runtime with lazy model initialization."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import torch
        import torch.nn as nn

        self.config = normalize_video_config(config)
        from ex_omni.distributed.context import current_context

        self.distributed_context = current_context()
        self.config.setdefault(
            "sp_size", int(self.config.get("sequence_parallel", 1))
        )
        if (self.config.get("mode") == "full_sequence"
                and self.distributed_context.is_distributed
                and int(self.config.get("sequence_parallel", 1)) > 1
                and int(self.config["sequence_parallel"]) != self.distributed_context.world_size):
            raise ValueError("full-sequence sequence_parallel must equal the video worker world size")
        if self.distributed_context.is_distributed:
            self.device = torch.device("cuda", self.distributed_context.local_rank)
        else:
            self.device = torch.device(str(self.config.get("device", "cuda:0")))
            if self.device.type == "cuda" and self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        dtype_name = str(self.config.get("dtype", "bf16"))
        self.dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }.get(dtype_name, torch.float32)
        use_fsdp = bool(
            self.config.get("use_fsdp", self.config.get("distributed", False))
        )
        auto_vae_offload = (
            str(self.config.get("mode", "full_sequence")) == "streaming"
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
            and not self.distributed_context.is_distributed
            and int(self.config.get("sequence_parallel", 1)) == 1
            and not use_fsdp
        )
        configured_vae_device = self.config.get("vae_device")
        if configured_vae_device is not None:
            if self.config["mode"] != "streaming" or use_fsdp:
                raise ValueError("video.vae_device requires a Streaming runtime without FSDP")
            if not self.config.get("resident_models", False) and not self.config.get("compile", {}).get("enabled", False):
                raise ValueError("video.vae_device requires resident models or compiled resident DiT")
            self.vae_device = torch.device(str(configured_vae_device))
            if self.vae_device.type != "cuda" or self.vae_device.index is None:
                raise ValueError("video.vae_device must be an explicit cuda:N device")
            if self.vae_device.index >= torch.cuda.device_count():
                raise ValueError("video.vae_device is not visible")
            if self.distributed_context.is_distributed and self.vae_device.index < self.distributed_context.world_size:
                raise ValueError("the dedicated VAE GPU must be outside the Video worker group")
            self.vae_pipeline_enabled = self.vae_device != self.device
        elif auto_vae_offload:
            main_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            vae_index = next(
                index
                for index in range(torch.cuda.device_count())
                if index != main_index
            )
            self.vae_device = torch.device("cuda", vae_index)
            self.vae_pipeline_enabled = True
        else:
            self.vae_device = self.device
            self.vae_pipeline_enabled = False

        # The video model reads architecture knobs through this compatibility
        # namespace. It is populated explicitly from the constructor config;
        # no argparse parsing occurs.
        from .utils import args_config

        args_config.set_runtime_config(self.config)
        from .models.model_manager import ModelManager
        from .wan_video import WanVideoPipeline
        from .models.speech_tokens import QwenSpeechTokenEmbedding

        initialization_error = None
        try:
            self.streaming_checkpoint_metadata = {}
            if self.config["mode"] == "streaming":
                self.streaming_checkpoint_metadata = _checkpoint_metadata(
                    str(self.config["lora_checkpoint"])
                )
                self.config = _resolve_streaming_checkpoint_config(
                    self.config, self.streaming_checkpoint_metadata
                )
                args_config.set_runtime_config(self.config)
                self._log_checkpoint(
                    "Streaming checkpoint policy: "
                    f"steps={self.config['steps']}, "
                    "persistent_sink=true, "
                    f"sink_rope_distance={self.config['stream_sink_rope_distance']}."
                )
            manager = ModelManager(device="cpu", infer=True)
            model_paths = [str(self.config["dit_path"]).split(",")]
            # T5 runs on rank 0 and its context tensor is broadcast to all ranks.
            if self.distributed_context.is_rank0:
                model_paths.append(str(self.config["text_encoder_path"]))
            rank0_vae = (
                self.config["mode"] == "streaming"
                and self.distributed_context.is_distributed
                and bool(self.config.get("rank0_vae", True))
            )
            if self.distributed_context.is_rank0 or not rank0_vae:
                model_paths.append(str(self.config["vae_path"]))
            manager.load_models(
                model_paths,
                torch_dtype=self.dtype,
                device="cpu",
            )
            sequence_parallel = int(
                self.config.get("sequence_parallel", self.config.get("sp_size", 1))
            )
            streaming_sequence_parallel = (
                self.config["mode"] == "streaming" and sequence_parallel > 1
            )
            self.pipe = WanVideoPipeline.from_model_manager(
                manager,
                torch_dtype=self.dtype,
                device=str(self.device),
                use_usp=sequence_parallel > 1
                and not streaming_sequence_parallel,
                infer=True,
                tokenizer_path=self.config.get("text_tokenizer_path"),
            )
            dit = self.pipe.denoising_model()
            if self.config["mode"] == "streaming":
                setup_flowmap = getattr(dit, "setup_flowmap_conditioning", None)
                if setup_flowmap is None:
                    raise RuntimeError(
                        "configured DiT does not support AnyFlow dual-time conditioning"
                    )
                setup_flowmap(
                    gate=float(self.config.get("flowmap_gate", 0.25)),
                    deltatime_type=str(self.config.get("deltatime_type", "r")),
                )
                setup_far = getattr(dit, "setup_far_conditioning", None)
                if setup_far is None:
                    raise RuntimeError(
                        "configured DiT does not support FAR compressed history"
                    )
                setup_far(
                    compressed_patch_size=tuple(
                        self.config.get(
                            "stream_compressed_patch_size", (1, 4, 4)
                        )
                    )
                )
            if not hasattr(dit, "speech_token_adapter"):
                input_dim = int(self.config.get("speech_token_embedding_dim", 256))
                output_dim = int(self.config.get("speech_token_adapter_dim", 10752))
                dit.speech_token_adapter = nn.Sequential(
                    nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)
                ).to(device=self.device, dtype=self.dtype)
            self.speech_embedding = QwenSpeechTokenEmbedding(
                str(self.config["speech_tokenizer_path"]),
                num_codebooks=int(self.config.get("speech_token_num_codebooks", 16)),
                codebook_source=str(
                    self.config.get("speech_token_codebook_source", "decoder")
                ),
            ).to(self.device)
            self.speech_embedding.eval().requires_grad_(False)
            self._load_checkpoint(dit)
            if bool(self.config.get("merge_lora", False)):
                self._merge_lora_for_inference(dit)
        except Exception as exc:
            initialization_error = exc

        if self.distributed_context.is_distributed:
            import torch.distributed as dist

            local_error = (
                f"{type(initialization_error).__name__}: {initialization_error}"
                if initialization_error is not None
                else None
            )
            errors = [None] * self.distributed_context.world_size
            dist.all_gather_object(
                errors,
                local_error,
                group=self.distributed_context.process_group,
            )
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            ]
            if failures:
                raise RuntimeError(
                    "video initialization failed before FSDP: " + "; ".join(failures)
                )
        elif initialization_error is not None:
            raise initialization_error

        if streaming_sequence_parallel:
            from ex_omni.distributed.sequence_parallel import (
                get_sequence_parallel_world_size,
                get_sp_group,
            )

            if get_sequence_parallel_world_size() != sequence_parallel:
                raise RuntimeError(
                    "Streaming SP requires sequence_parallel to equal the "
                    "torch.distributed WORLD size."
                )
            self.pipe.sp_size = get_sequence_parallel_world_size()
            self.pipe.sp_group = get_sp_group()
            self.pipe.streaming_sequence_parallel = True

        dit.requires_grad_(False)
        compile_config = dict(self.config.get("compile", {}))
        if use_fsdp:
            if not self.distributed_context.is_distributed:
                raise ValueError("video.use_fsdp requires a distributed process group")
            from .distributed.fsdp import shard_model

            dit = dit.to(device=self.device, dtype=self.dtype)
            ignored_modules = [
                module
                for module in (
                    getattr(dit, "speech_token_adapter", None),
                    getattr(dit, "audio_proj", None),
                    getattr(dit, "audio_cond_projs", None),
                )
                if module is not None
            ]
            self.pipe.dit = shard_model(
                dit,
                device_id=self.device,
                param_dtype=self.dtype,
                process_group=self.distributed_context.process_group,
                ignored_modules=ignored_modules,
                # Root FSDP input casting recreates dataclass containers. The
                # streaming state is mutable, so updates would otherwise land
                # in a temporary copy instead of the persistent session state.
                cast_root_forward_inputs=False,
                use_orig_params=bool(compile_config.get("enabled", False)),
            )
            # Streaming VAE/T5 run on rank 0; both remain outside the
            # FSDP-sharded DiT backbone.
            migration_error = None
            try:
                if self.pipe.vae is not None:
                    self.pipe.vae = self.pipe.vae.to(
                        device=self.device, dtype=self.dtype
                    )
                if self.pipe.text_encoder is not None:
                    self.pipe.text_encoder = self.pipe.text_encoder.to(
                        device=self.device, dtype=self.dtype
                    )
            except Exception as exc:
                migration_error = f"{type(exc).__name__}: {exc}"
            import torch.distributed as dist

            migration_errors = [None] * self.distributed_context.world_size
            dist.all_gather_object(
                migration_errors,
                migration_error,
                group=self.distributed_context.process_group,
            )
            migration_failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(migration_errors)
                if error is not None
            ]
            if migration_failures:
                raise RuntimeError(
                    "video auxiliary-model migration failed after FSDP: "
                    + "; ".join(migration_failures)
                )
        elif bool(compile_config.get("enabled", False)) or bool(
            self.config.get("resident_models", False)
        ):
            # ModelManager materializes inference modules on CPU. Eager mode
            # installs per-layer offload hooks below, but compiled graphs need
            # stable CUDA parameter addresses for their entire lifetime.
            self.pipe.dit = dit.to(device=self.device, dtype=self.dtype)
            if self.pipe.vae is not None:
                self.pipe.vae = self.pipe.vae.to(
                    device=self.vae_device, dtype=self.dtype
                )
            if self.pipe.text_encoder is not None:
                self.pipe.text_encoder = self.pipe.text_encoder.to(
                    device=self.device, dtype=self.dtype
                )
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        compile_enabled = bool(compile_config.get("enabled", False))
        keep_model_on_gpu = bool(
            compile_config.get("keep_model_on_gpu", False)
        )
        if not use_fsdp and compile_enabled and not keep_model_on_gpu:
            raise ValueError(
                "single-GPU compile requires a resident video model because VRAM "
                "offloading invalidates compiled graphs"
            )
        self._apply_compile_optimizations(use_fsdp=use_fsdp)
        if (
            not use_fsdp
            and not compile_enabled
            and not bool(self.config.get("resident_models", False))
        ):
            self.pipe.enable_vram_management(
                num_persistent_param_in_dit=self.config.get("num_persistent_param_in_dit")
            )

    def _checkpoint_paths(self) -> list[str]:
        paths = [
            self.config.get("base_checkpoint"),
            self.config.get("lora_checkpoint"),
        ]
        return [str(path) for path in paths if path]

    def _log_checkpoint(self, message: str) -> None:
        if self.distributed_context.is_rank0:
            print(message, flush=True)

    def _apply_compile_optimizations(self, *, use_fsdp: bool) -> None:
        options = dict(self.config.get("compile", {}))
        if not bool(options.get("enabled", False)):
            return
        import torch

        if not hasattr(torch, "compile"):
            if bool(options.get("fallback_on_error", True)):
                self._log_checkpoint(
                    "torch.compile unavailable; continuing in eager mode."
                )
                return
            raise RuntimeError("video.compile.enabled requires torch.compile")
        import torch._dynamo

        torch._dynamo.config.suppress_errors = bool(
            options.get("fallback_on_error", True)
        )
        torch._dynamo.config.recompile_limit = int(
            options.get("recompile_limit", 8)
        )
        mode = str(options.get("mode", "default"))
        cuda_graphs = bool(options.get("cuda_graphs", False))
        streaming_graphs = (
            bool(self.config.get("graph", False))
            and self.config.get("mode") == "streaming"
        )
        effective_mode = "reduce-overhead" if cuda_graphs else mode
        self._log_checkpoint(
            "Compiling video DiT "
            f"(backend={options.get('backend', 'inductor')}, "
            f"mode={effective_mode}, inductor_graphs={cuda_graphs}, "
            f"streaming_graphs={streaming_graphs}, "
            f"recompile_limit={options.get('recompile_limit', 8)}, "
            f"fsdp={use_fsdp}). First chunks warm up."
        )
        try:
            frequencies = getattr(self.pipe.dit, "freqs", None)
            if frequencies is not None:
                self.pipe.dit.freqs = tuple(
                    frequency.to(device=self.device)
                    for frequency in frequencies
                )
            self.pipe.dit = torch.compile(
                self.pipe.dit,
                backend=str(options.get("backend", "inductor")),
                mode=effective_mode,
                fullgraph=bool(options.get("fullgraph", False)),
                dynamic=bool(options.get("dynamic", False)),
            )
            if cuda_graphs:
                def mark_cudagraph_step(_module, _args):
                    torch.compiler.cudagraph_mark_step_begin()

                self.pipe.dit.register_forward_pre_hook(
                    mark_cudagraph_step,
                    prepend=True,
                )
                self.pipe.cudagraphs_enabled = True
        except Exception as exc:
            if not bool(options.get("fallback_on_error", True)):
                raise
            self._log_checkpoint(
                "Video torch.compile setup failed; continuing in eager mode: "
                f"{type(exc).__name__}: {exc}"
            )

    def _merge_lora_for_inference(self, dit) -> None:
        from peft.tuners.tuners_utils import BaseTunerLayer

        layers = [
            module
            for module in dit.modules()
            if isinstance(module, BaseTunerLayer) and not module.merged
        ]
        if not layers:
            self._log_checkpoint("No unmerged Video LoRA layers found.")
            return
        started = time.perf_counter()
        for module in layers:
            module.merge(safe_merge=False)
        unloaded = 0

        def unload_merged_adapters(parent) -> None:
            nonlocal unloaded
            for name, child in list(parent.named_children()):
                if isinstance(child, BaseTunerLayer):
                    setattr(parent, name, child.get_base_layer())
                    unloaded += 1
                else:
                    unload_merged_adapters(child)

        unload_merged_adapters(dit)
        self._log_checkpoint(
            f"Merged and unloaded {len(layers)} Video LoRA layers "
            f"({unloaded} wrappers removed) for inference in "
            f"{time.perf_counter() - started:.2f}s."
        )

    def _load_checkpoint(self, dit) -> None:
        architecture = str(
            self.config.get(
                "architecture", self.config.get("train_architecture", "lora")
            )
        )
        paths = self._checkpoint_paths()
        if not paths:
            raise ValueError("video inference checkpoint is required")
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(path)
        if architecture == "lora":
            from peft import LoraConfig, inject_adapter_in_model

            init = self.config.get("init_lora_weights", "kaiming")
            targets = str(
                self.config.get(
                    "lora_target_modules", "q,k,v,o,ffn.0,ffn.2"
                )
            ).split(",")
            lora = LoraConfig(
                r=int(self.config.get("lora_rank", 128)),
                lora_alpha=float(self.config.get("lora_alpha", 64.0)),
                init_lora_weights=True if init == "kaiming" else init,
                target_modules=targets,
            )
            self._log_checkpoint(
                "Injecting video LoRA adapter: "
                f"rank={lora.r}, alpha={lora.lora_alpha}, "
                f"targets={','.join(targets)}."
            )
            inject_adapter_in_model(lora, dit)
            configured_lora = str(self.config.get("lora_checkpoint"))
            for index, path in enumerate(paths, start=1):
                layer = (
                    "lora_checkpoint"
                    if path == configured_lora
                    else "base_checkpoint"
                )
                state = _load_state_dict(path)
                self._log_checkpoint(
                    f"Loading video {layer} [{index}/{len(paths)}] "
                    f"from {path} ({len(state)} tensors)."
                )
                incompatible = dit.load_state_dict(state, strict=False)
                summary = _checkpoint_load_summary(state, incompatible)
                self._log_checkpoint(
                    f"Loaded video {layer}: "
                    f"matched={summary['matched']}/{summary['total']}, "
                    f"LoRA={summary['lora_matched']}/{summary['lora_total']}, "
                    f"audio={summary['audio_matched']}/{summary['audio_total']}, "
                    f"missing_model_keys={summary['missing']}, "
                    f"unexpected_checkpoint_keys={summary['unexpected']}."
                )
                if summary["unexpected"]:
                    preview = ", ".join(incompatible.unexpected_keys[:8])
                    self._log_checkpoint(
                        f"Unexpected keys in video {layer}: {preview}"
                    )
                if (
                    self.config.get("mode") != "streaming"
                    and layer == "lora_checkpoint"
                    and summary["lora_total"] > 0
                    and summary["lora_matched"] == 0
                ):
                    raise RuntimeError(
                        "video LoRA checkpoint did not match any injected "
                        f"adapter parameters: {path}"
                    )
        else:
            path = paths[-1]
            state = _load_state_dict(path)
            self._log_checkpoint(
                f"Loading strict video checkpoint from {path} "
                f"({len(state)} tensors)."
            )
            dit.load_state_dict(state, strict=True)
            self._log_checkpoint(
                f"Loaded strict video checkpoint: matched={len(state)}/{len(state)}."
            )

    def _prepare_image(
        self,
        path: Path,
        overrides: Mapping[str, Any] | None = None,
    ):
        import torch.nn.functional as functional
        from PIL import Image
        import torchvision.transforms.functional as vision

        image = vision.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        _, _, height, width = image.shape
        options = {**self.config, **dict(overrides or {})}
        sizes = options.get(
            f"image_sizes_{options.get('max_hw', 720)}",
            [[400, 720], [720, 720], [720, 400]],
        )
        target = _match_size(sizes, height, width)
        scale = max(target[0] / height, target[1] / width)
        resized = vision.resize(image, [int(height * scale), int(width * scale)])
        pad_height = target[0] - resized.shape[-2]
        pad_width = target[1] - resized.shape[-1]
        resized = functional.pad(
            resized,
            (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
        )
        return resized.mul(2).sub(1).unsqueeze(2), target

    def _prepare_audio(self, token_path: Path, video_length: int, first_fixed: int):
        import torch
        from .models.speech_tokens import (
            load_speech_tokens,
            normalize_speech_tokens,
            pad_features_with_zeros,
            repeat_tokens_to_frames,
        )

        tokens = normalize_speech_tokens(
            load_speech_tokens(token_path),
            int(self.config.get("speech_token_num_codebooks", 16)),
        )
        repeat = max(
            int(
                round(
                    float(self.config.get("fps", 25))
                    / float(self.config.get("speech_token_rate", 12.5))
                )
            ),
            1,
        )
        original_length = tokens.shape[0] * repeat
        audio_length = original_length
        fixed = int(self.config.get("overlap_frame", 1))
        if audio_length < video_length - first_fixed:
            audio_length += (video_length - first_fixed) - audio_length % (video_length - first_fixed)
        elif (audio_length - (video_length - first_fixed)) % (video_length - fixed):
            audio_length += (video_length - fixed) - (
                audio_length - (video_length - first_fixed)
            ) % (video_length - fixed)
        with torch.no_grad():
            features = repeat_tokens_to_frames(
                self.speech_embedding(tokens.to(self.device)), original_length, repeat
            )
        self.pipe.load_models_to_device(["dit"])
        embedded = self.pipe.dit.speech_token_adapter(features.to(self.device, dtype=self.dtype))
        return pad_features_with_zeros(embedded, audio_length), original_length

    def generate_full_sequence(self, request):
        import torch

        if not request.ref_image.exists():
            raise FileNotFoundError(request.ref_image)
        token_path = Path(request.speech_tokens)
        if not token_path.exists():
            raise FileNotFoundError(token_path)
        image, selected_size = self._prepare_image(
            request.ref_image,
            request.overrides,
        )
        length = _resolve_full_sequence_window_frames(
            self.config,
            request.overrides,
            selected_size,
        )
        latent_length = (length + 3) // 4
        fixed = int(self.config.get("overlap_frame", 1))
        first_fixed = 1
        prefix_latent = (3 + fixed) // 4
        audio, original_length = self._prepare_audio(token_path, length, first_fixed)
        times = (audio.shape[0] - length + first_fixed) // (length - fixed) + 1
        if times * (length - fixed) + fixed < audio.shape[0]:
            times += 1
        progress_callback = request.overrides.get("progress_callback")
        preview_callback = request.overrides.get("preview_callback")

        cfg_scale = float(
            request.overrides.get(
                "cfg",
                self.config.get("cfg", self.config.get("guidance_scale", 3.5)),
            )
        )
        audio_cfg_scale = float(
            request.overrides.get(
                "audio_cfg",
                self.config.get(
                    "audio_cfg", self.config.get("audio_scale", 3.5)
                ),
            )
        )
        negative_prompt = str(self.config.get("negative_prompt", ""))
        positive_context, negative_context = self._prepare_prompt_contexts(
            request.prompt,
            negative_prompt=negative_prompt,
            include_negative=cfg_scale != 1.0,
        )
        prompt_emb_posi = {"context": positive_context}
        prompt_emb_nega = (
            {"context": negative_context}
            if negative_context is not None
            else None
        )

        self.pipe.full_sequence_cfg_parallel_device = self.config.get("full_sequence_cfg_parallel_device")
        self.pipe.load_models_to_device(["vae"])
        image_latent = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
        mask = torch.zeros_like(image_latent.repeat(1, 1, latent_length, 1, 1)[:, :1])
        image_cat = image_latent.repeat(1, 1, latent_length, 1, 1)
        mask[:, :, 1:] = 1
        image_condition = {"y": torch.cat([image_cat, mask], dim=1)}
        video = []
        audio_prefix = torch.zeros_like(audio[:first_fixed])
        current_image = image
        current_latent = image_latent
        for clip in range(times):
            def report_clip_progress(step, total_steps, clip_index=clip):
                if callable(progress_callback):
                    progress_callback(
                        step,
                        total_steps,
                        clip_index + 1,
                        times,
                    )

            overlap = first_fixed if clip == 0 else fixed
            if clip:
                image_condition["y"][:, -1:, :prefix_latent] = 0
            start = 0 if clip == 0 else length - first_fixed + (clip - 1) * (length - overlap)
            chunk = audio[start : min(start + length - overlap, audio.shape[0])]
            chunk = torch.cat([audio_prefix, chunk])
            audio_prefix = chunk[-fixed:]
            audio_condition = {"audio_emb": chunk.unsqueeze(0).to(self.device, self.dtype)}
            if current_latent is None:
                self.pipe.load_models_to_device(["vae"])
                current_latent = self.pipe.encode_video(current_image.to(self.dtype)).to(self.device)
            current_latent = torch.cat(
                [
                    current_latent,
                    torch.zeros_like(
                        current_latent[:, :, :1].repeat(
                            1, 1, latent_length - (3 + overlap) // 4, 1, 1
                        )
                    ),
                ],
                dim=2,
            )
            frames, _, _ = self.pipe.log_video(
                current_latent,
                request.prompt,
                (3 + overlap) // 4,
                image_condition,
                audio_condition,
                negative_prompt,
                num_inference_steps=int(
                    request.overrides.get("steps", self.config.get("steps", self.config.get("num_steps", 50)))
                ),
                cfg_scale=cfg_scale,
                audio_cfg_scale=audio_cfg_scale,
                sigma_shift=float(self.config.get("sigma_shift", 5.0)),
                return_latent=True,
                tea_cache_l1_thresh=float(self.config.get("tea_cache_l1_thresh", 0)),
                tea_cache_model_id="Wan2.1-T2V-1.3B",
                progress_callback=report_clip_progress,
                prompt_emb_posi=prompt_emb_posi,
                prompt_emb_nega=prompt_emb_nega,
                decode_reconstruction=False,
            )
            current_latent = None
            current_image = (
                frames[:, -fixed:].clip(0, 1).mul(2).sub(1).permute(0, 2, 1, 3, 4)
            )
            video.append(frames if clip == 0 else frames[:, overlap:])
            if (
                callable(preview_callback)
                and self.distributed_context.is_rank0
            ):
                preview = torch.cat(video, dim=1)[:, : original_length + 1]
                preview_frames = (
                    preview[0]
                    .permute(0, 2, 3, 1)
                    .float()
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .cpu()
                    .numpy()
                )
                preview_callback(
                    preview_frames,
                    float(self.config.get("fps", 25)),
                    clip + 1,
                    times,
                )
        result = torch.cat(video, dim=1)[:, : original_length + 1]
        if self.distributed_context.is_rank0:
            self._save_video(result, request.output_path)
        return {
            "output_path": request.output_path,
            "mode": "full_sequence",
            "frames": int(result.shape[1]),
            "fps": float(self.config.get("fps", 25)),
        }

    def _broadcast_rank0_tensor(self, value):
        """Broadcast a variable-shaped tensor from rank 0 on the shared world."""
        import torch

        context = self.distributed_context
        if not context.is_distributed:
            return value
        import torch.distributed as dist

        metadata = [
            (
                tuple(value.shape),
                str(value.dtype).replace("torch.", ""),
            )
            if context.is_rank0
            else None
        ]
        dist.broadcast_object_list(metadata, src=0, group=context.process_group)
        shape, dtype_name = metadata[0]
        if not context.is_rank0:
            dtype = getattr(torch, dtype_name)
            value = torch.empty(shape, device=self.device, dtype=dtype)
        dist.broadcast(value, src=0, group=context.process_group)
        return value

    def _prepare_prompt_contexts(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        include_negative: bool = False,
    ):
        """Encode T5 prompts on rank 0 and broadcast them to every FSDP rank."""
        positive_context = None
        negative_context = None
        rank0_error = None
        if self.distributed_context.is_rank0:
            try:
                if self.pipe.text_encoder is None:
                    raise RuntimeError("rank 0 must load the T5 text encoder")
                self.pipe.load_models_to_device(["text_encoder"])
                positive_context = self.pipe.encode_prompt(
                    prompt, positive=True
                )["context"].to(self.device, dtype=self.dtype)
                if include_negative:
                    negative_context = self.pipe.encode_prompt(
                        negative_prompt, positive=False
                    )["context"].to(self.device, dtype=self.dtype)
            except Exception as exc:
                rank0_error = f"{type(exc).__name__}: {exc}"
        if self.distributed_context.is_distributed:
            import torch.distributed as dist

            status = [rank0_error]
            dist.broadcast_object_list(
                status,
                src=0,
                group=self.distributed_context.process_group,
            )
            rank0_error = status[0]
        if rank0_error is not None:
            raise RuntimeError(f"rank0 prompt encoding failed: {rank0_error}")
        positive_context = self._broadcast_rank0_tensor(positive_context)
        if include_negative:
            negative_context = self._broadcast_rank0_tensor(negative_context)
        return positive_context, negative_context

    def _start_streaming_stream(
        self,
        *,
        prompt: str,
        ref_image: Path,
        seed: int | None,
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        import torch

        from ex_omni.config import normalize_video_config

        from .schedulers.contracts import FlowMapSchedule
        from .streaming import (
            AudioStreamingState,
            DiTStreamingConfig,
            StreamingInferenceSession,
        )

        request_config = normalize_video_config(
            self.config,
            mode="streaming",
            overrides=overrides,
        )
        if getattr(self, "streaming_checkpoint_metadata", None):
            request_config = _resolve_streaming_checkpoint_config(
                request_config, self.streaming_checkpoint_metadata
            )
        if float(request_config.get("tea_cache_l1_thresh", 0)) > 0:
            raise ValueError("Streaming causal inference does not support TeaCache")
        if not ref_image.exists():
            raise FileNotFoundError(ref_image)

        context = None
        reference_latent = None
        rank0_error = None
        setup_started = time.perf_counter()
        prompt_finished = setup_started
        image_finished = setup_started
        vae_finished = setup_started
        if self.distributed_context.is_rank0:
            try:
                if self.pipe.text_encoder is None:
                    raise RuntimeError("rank 0 must load the T5 text encoder")
                self.pipe.load_models_to_device(["text_encoder"])
                context = self.pipe.encode_prompt(prompt, positive=True)["context"].to(
                    self.device, dtype=self.dtype
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                prompt_finished = time.perf_counter()
                image, _ = self._prepare_image(ref_image, overrides)
                image_finished = time.perf_counter()
                self.pipe.load_models_to_device(["vae"])
                if self.vae_pipeline_enabled:
                    reference_latent = self.pipe.vae.encode(
                        image.to(dtype=self.dtype),
                        device=self.vae_device,
                        tiled=True,
                    ).to(self.device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.vae_device)
                        torch.cuda.synchronize(self.device)
                else:
                    reference_latent = self.pipe.encode_video(
                        image.to(dtype=self.dtype)
                    ).to(self.device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device)
                vae_finished = time.perf_counter()
            except Exception as exc:
                rank0_error = f"{type(exc).__name__}: {exc}"
        if self.distributed_context.is_distributed:
            import torch.distributed as dist

            status = [rank0_error]
            dist.broadcast_object_list(
                status, src=0, group=self.distributed_context.process_group
            )
            rank0_error = status[0]
        if rank0_error is not None:
            raise RuntimeError(f"rank0 context/reference preparation failed: {rank0_error}")
        context = self._broadcast_rank0_tensor(context)
        reference_latent = self._broadcast_rank0_tensor(reference_latent)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        setup_finished = time.perf_counter()
        setup_timing_seconds = {
            "video_prompt_encode_seconds": prompt_finished - setup_started,
            "reference_image_prepare_seconds": image_finished - prompt_finished,
            "reference_vae_encode_seconds": vae_finished - image_finished,
            "reference_broadcast_and_session_seconds": setup_finished - vae_finished,
        }

        text_cfg = float(
            request_config.get(
                "cfg", request_config.get("guidance_scale", 1.0)
            )
        )
        audio_cfg = float(
            request_config.get(
                "audio_cfg",
                request_config.get("audio_scale", 1.0),
            )
        )
        if text_cfg != 1.0 or audio_cfg != 1.0:
            raise ValueError("Streaming streaming requires cfg=audio_cfg=1.0")
        reference_conditioning = request_config.get(
            "stream_reference_conditioning", "persistent_sink"
        )
        if reference_conditioning not in (
            "persistent_sink", "previous_chunk_latent_with_sink",
            "previous_chunk_latent"
        ):
            raise ValueError(
                "stream_reference_conditioning must be persistent_sink, "
                "previous_chunk_latent_with_sink, or previous_chunk_latent"
            )
        stream_config = DiTStreamingConfig(
            use_cuda_graphs=bool(request_config.get("graph", False)),
            chunk_latent_frames=int(
                request_config.get("stream_chunk_latent_frames", 3)
            ),
            first_chunk_latent_frames=int(
                request_config.get("stream_first_chunk_latent_frames", 1)
            ),
            full_chunk_limit=int(
                request_config.get("stream_full_chunk_limit", 3)
            ),
            sink_rope_distance=int(request_config["stream_sink_rope_distance"]),
            full_patch_size=tuple(
                request_config.get("stream_full_patch_size", (1, 2, 2))
            ),
            compressed_patch_size=tuple(
                request_config.get("stream_compressed_patch_size", (1, 4, 4))
            ),
            max_compressed_latent_frames=int(
                request_config.get("stream_max_compressed_latent_frames", 96)
            ),
        )
        if stream_config.chunk_latent_frames != 3:
            raise ValueError("6-unit streaming requires stream_chunk_latent_frames=3")
        if not bool(request_config.get("use_kv_cache", True)):
            raise ValueError("FAR streaming requires use_kv_cache=true")
        schedule = FlowMapSchedule.shifted(
            int(request_config.get("steps", request_config.get("num_steps", 8))),
            num_train_timesteps=int(
                request_config.get("training_timesteps", 1000)
            ),
            shift=float(request_config.get("sigma_shift", 5.0)),
        )
        schedule_values = schedule.source_timesteps
        session = StreamingInferenceSession(
            self.pipe,
            config=stream_config,
            num_inference_steps=len(schedule_values),
            sigma_shift=float(request_config.get("sigma_shift", 5.0)),
            text_cfg_scale=text_cfg,
            audio_cfg_scale=audio_cfg,
            flowmap_schedule=schedule,
            decode_output=(
                self.distributed_context.is_rank0
                and not self.vae_pipeline_enabled
            ),
            use_kv_cache=bool(request_config.get("use_kv_cache", True)),
        )
        async_vae_decoder = (
            _AsyncVaeDecoder(self.pipe.vae, device=self.vae_device)
            if self.vae_pipeline_enabled and self.distributed_context.is_rank0
            else None
        )
        audio_state = AudioStreamingState()
        dit = self.pipe.dit
        dit_module = getattr(dit, "module", dit)
        reference_audio = reference_latent.new_zeros(
            1,
            1,
            int(request_config.get("speech_token_adapter_dim", 10752)),
        )
        from .streaming import pack_streaming_audio

        prepared_reference_audio, audio_state = pack_streaming_audio(
            reference_audio,
            dit_module.audio_proj,
            dit_module.audio_cond_projs,
            audio_state,
        )
        if prepared_reference_audio is None:
            raise RuntimeError("reference audio did not produce an AudioPack group")
        reference_condition = torch.cat(
            [reference_latent, torch.zeros_like(reference_latent[:, :1])],
            dim=1,
        )
        session.prefill_reference(
            reference_latent,
            context,
            reference_condition,
            prepared_audio_condition=prepared_reference_audio,
        )
        if async_vae_decoder is not None:
            async_vae_decoder.submit(
                reference_latent.detach(),
                first_chunk=True,
                expected_frames=0,
            ).result()
        noise_seed = int(
            seed if seed is not None else request_config.get("seed", 42)
        )
        noise_generator = (
            torch.Generator(device=self.device).manual_seed(noise_seed)
            if self.distributed_context.is_rank0
            else None
        )
        return {
            "session": session,
            "audio_state": audio_state,
            "context": context,
            "reference_latent": reference_latent,
            "reference_conditioning": reference_conditioning,
            "previous_generated_latent": None,
            "stream_config": stream_config,
            "seed": noise_seed,
            "noise_generator": noise_generator,
            "setup_timing_seconds": setup_timing_seconds,
            "async_vae_decoder": async_vae_decoder,
            "pending_vae_outputs": [],
            "progress_callback": overrides.get("progress_callback"),
        }

    def _process_streaming_units(
        self,
        state: dict[str, Any],
        units,
        *,
        chunk_id: int,
        valid_units: int,
    ):
        import numpy as np
        import torch

        from .models.speech_tokens import repeat_tokens_to_frames
        from .streaming import pack_streaming_audio

        cuda_timing = bool(
            torch.cuda.is_available() and str(self.device).startswith("cuda")
        )
        audio_started_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        audio_finished_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        latent_prepared_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        audio_wall_started = time.perf_counter()
        if audio_started_event is not None:
            audio_started_event.record()
        tokens = torch.as_tensor(units, dtype=torch.long)
        frame_count = int(tokens.shape[0]) * 2
        features = repeat_tokens_to_frames(
            self.speech_embedding(tokens.to(self.device)), frame_count, 2
        )
        self.pipe.load_models_to_device(["dit"])
        dit = self.pipe.dit
        dit_module = getattr(dit, "module", dit)
        audio_embeddings = dit_module.speech_token_adapter(
            features.to(self.device, dtype=self.dtype)
        )
        valid_audio_frames = int(valid_units) * 2
        if valid_audio_frames < audio_embeddings.shape[0]:
            audio_embeddings[valid_audio_frames:] = 0
        audio_embeddings = audio_embeddings.unsqueeze(0)
        prepared_audio, state["audio_state"] = pack_streaming_audio(
            audio_embeddings,
            dit_module.audio_proj,
            dit_module.audio_cond_projs,
            state["audio_state"],
        )
        if prepared_audio is None:
            raise RuntimeError("audio chunk did not produce a complete AudioPack group")
        audio_wall_finished = time.perf_counter()
        if audio_finished_event is not None:
            audio_finished_event.record()

        reference = state["reference_latent"]
        latent_frames = (
            state["stream_config"].chunk_latent_frames
        )
        shape = (
            reference.shape[0],
            reference.shape[1],
            latent_frames,
            reference.shape[3],
            reference.shape[4],
        )
        if self.distributed_context.is_rank0:
            initial_latents = torch.randn(
                shape,
                generator=state["noise_generator"],
                device=self.device,
                dtype=self.dtype,
            )
        else:
            initial_latents = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.distributed_context.broadcast_tensor(initial_latents, src=0)

        reference_conditioning = state["reference_conditioning"]
        if chunk_id > 0 and reference_conditioning == "previous_chunk_latent":
            state["stream_config"].reference_sink_attention = False
        image_source = reference
        if chunk_id > 0 and reference_conditioning in (
            "previous_chunk_latent_with_sink", "previous_chunk_latent"
        ):
            image_source = state["previous_generated_latent"]
            if image_source is None:
                raise RuntimeError("previous generated latent is unavailable")
        image_cat = image_source.repeat(1, 1, latent_frames, 1, 1)
        mask = torch.ones_like(image_cat[:, :1])
        latent_prepare_wall_finished = time.perf_counter()
        if latent_prepared_event is not None:
            latent_prepared_event.record()
        progress_callback = state.get("progress_callback")

        def report_diffusion_progress(timesteps):
            total_steps = len(timesteps)
            for step_index, timestep in enumerate(timesteps):
                yield timestep
                if callable(progress_callback):
                    progress_callback(
                        step_index + 1,
                        total_steps,
                        chunk_id + 1,
                        None,
                    )

        output = state["session"].denoise_chunk(
            initial_latents,
            state["context"],
            torch.cat([image_cat, mask], dim=1),
            prepared_audio_condition=prepared_audio,
            step_progress_bar=(
                report_diffusion_progress
                if callable(progress_callback)
                else None
            ),
        )
        if reference_conditioning in (
            "previous_chunk_latent_with_sink", "previous_chunk_latent"
        ):
            state["previous_generated_latent"] = output.latents[:, :, -1:].detach().clone()
        expected_offset = (
            state["stream_config"].first_chunk_latent_frames
            + (chunk_id + 1) * state["stream_config"].chunk_latent_frames
        )
        actual_offset = state["session"].dit_state.global_frame_offset
        if actual_offset != expected_offset:
            raise RuntimeError(
                "persistent DiT state did not survive the forward call: "
                f"expected global_frame_offset={expected_offset}, "
                f"got {actual_offset}. Check FSDP root input casting."
            )
        cancelled, failed = self.distributed_context.boundary_status()
        if cancelled:
            raise InterruptedError("video stream cancelled at chunk boundary")
        if failed:
            raise RuntimeError("a distributed rank failed at video chunk boundary")
        timing_seconds = dict(output.timing_seconds)
        if audio_started_event is not None:
            timing_seconds.update(
                {
                    "speech_conditioning_seconds": (
                        audio_started_event.elapsed_time(audio_finished_event)
                        / 1000.0
                    ),
                    "latent_prepare_seconds": (
                        audio_finished_event.elapsed_time(latent_prepared_event)
                        / 1000.0
                    ),
                }
            )
        else:
            timing_seconds.update(
                {
                    "speech_conditioning_seconds": (
                        audio_wall_finished - audio_wall_started
                    ),
                    "latent_prepare_seconds": (
                        latent_prepare_wall_finished - audio_wall_finished
                    ),
                }
            )
        async_vae_decoder = state.get("async_vae_decoder")
        if async_vae_decoder is not None:
            timing_seconds.pop("vae_decode_seconds", None)
            expected = int(valid_units) * 2
            return {
                "decode_future": async_vae_decoder.submit(
                    output.latents.detach(),
                    first_chunk=False,
                    expected_frames=expected,
                ),
                "timing_seconds": timing_seconds,
            }
        if not self.distributed_context.is_rank0:
            timing_seconds["frame_postprocess_seconds"] = 0.0
            return np.empty((0, 0, 0, 3), dtype=np.uint8), timing_seconds
        frame_postprocess_started = time.perf_counter()
        frames = (
            (output.frames.permute(0, 2, 1, 3, 4).float() + 1.0)
            .div(2.0)
            .clamp(0, 1)
        )
        expected = int(valid_units) * 2
        frames = frames[0, :expected]
        result = frames.permute(0, 2, 3, 1).mul(255).byte().cpu().numpy()
        timing_seconds["frame_postprocess_seconds"] = (
            time.perf_counter() - frame_postprocess_started
        )
        return result, timing_seconds

    def start_streaming_stream(
        self,
        *,
        prompt: str,
        ref_image: str | Path,
        seed: int | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> StreamingVideoSession:
        return StreamingVideoSession(
            self,
            prompt=prompt,
            ref_image=Path(ref_image),
            seed=seed,
            overrides=overrides,
        )

    def generate_streaming(self, request):
        """Run the causal streaming chunk pipeline."""
        import imageio.v2 as imageio
        import numpy as np

        from .models.speech_tokens import load_speech_tokens, normalize_speech_tokens

        token_path = Path(request.speech_tokens)
        if not token_path.exists():
            raise FileNotFoundError(token_path)
        tokens = normalize_speech_tokens(
            load_speech_tokens(token_path),
            int(self.config.get("speech_token_num_codebooks", 16)),
        ).numpy()
        stream = self.start_streaming_stream(
            prompt=request.prompt,
            ref_image=request.ref_image,
            seed=request.seed,
            overrides=request.overrides,
        )
        chunks = stream.push_units(tokens)
        chunks.extend(stream.finish())
        if self.distributed_context.is_rank0:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(
                request.output_path, fps=float(self.config.get("fps", 25))
            ) as writer:
                for chunk in chunks:
                    for frame in chunk["frames"]:
                        writer.append_data(np.asarray(frame, dtype=np.uint8))
        return {
            "output_path": request.output_path,
            "mode": "streaming",
            "frames": int(tokens.shape[0] * 2),
            "fps": float(self.config.get("fps", 25)),
        }

    def _save_video(self, video, output_path: Path) -> None:
        import imageio.v2 as imageio
        import numpy as np

        # FSDP gathers identical frames on every rank; only rank 0 writes so the
        # eight ranks do not race on the same output file.
        if self.distributed_context.is_distributed and not self.distributed_context.is_rank0:
            return
        print(f"[video] encoding silent MP4: {output_path}", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = (
            video[0]
            .permute(0, 2, 3, 1)
            .float()
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )
        with imageio.get_writer(output_path, fps=float(self.config.get("fps", 25))) as writer:
            for array in arrays:
                writer.append_data(np.asarray(array, dtype=np.uint8))
        print(f"[video] silent MP4 encoding complete: {output_path}", flush=True)
