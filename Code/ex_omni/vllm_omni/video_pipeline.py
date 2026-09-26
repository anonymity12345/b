"""vLLM-Omni custom diffusion pipeline for Ex-Omni's streaming Wan2.1 model."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn


MODEL_ARCH = "ExOmniWan21StreamingPipeline"
MODEL_TYPE = "ex_omni_wan21_streaming"


def get_ex_omni_wan21_post_process_func(od_config):
    del od_config

    def post_process_func(result, sampling_params=None):
        output_type = getattr(sampling_params, "output_type", None) or "pt"
        if isinstance(result, dict):
            frames = result["frames"]
            ex_omni_metadata = {
                "operation": result.get("operation", "generate"),
                "payloads": list(result.get("payloads", [])),
                "setup_timing_seconds": dict(
                    result.get("setup_timing_seconds", {})
                ),
            }
        else:
            frames = result
            ex_omni_metadata = {}
        if output_type == "np" and isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        metadata = {"video": {"fps": 25.0}}
        if ex_omni_metadata:
            metadata["ex_omni"] = ex_omni_metadata
        return {
            "payload": {"video": frames},
            "metadata": metadata,
        }

    return post_process_func


def register_vllm_omni_pipeline() -> None:
    from vllm_omni.config import register_pipeline
    from vllm_omni.config.stage_config import (
        PipelineConfig,
        StageExecutionType,
        StagePipelineConfig,
    )
    from vllm_omni.diffusion.registry import register_diffusion_model

    register_pipeline(
        PipelineConfig(
            model_type=MODEL_TYPE,
            model_arch=MODEL_ARCH,
            diffusers_class_name=MODEL_ARCH,
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="dit",
                    execution_type=StageExecutionType.DIFFUSION,
                    input_sources=(),
                    final_output=True,
                    final_output_type="video",
                    model_arch=MODEL_ARCH,
                ),
            ),
        )
    )
    register_diffusion_model(
        model_arch=MODEL_ARCH,
        module_name="ex_omni.vllm_omni.video_pipeline",
        class_name="ExOmniWan21StreamingPipeline",
        post_process_func_name="get_ex_omni_wan21_post_process_func",
    )


class ExOmniWan21StreamingPipeline(nn.Module):
    """Schedule the validated causal Wan2.1 runtime through DiffusionEngine.

    This adapter deliberately retains Ex-Omni's model loader and streaming
    kernels. The checkpoint is a custom Wan2.1 1.3B audio-conditioned streaming model,
    not a Wan2.2-S2V checkpoint and not a diffusers pipeline.
    """

    supports_request_batch: ClassVar[bool] = False

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        custom_args = dict(getattr(od_config, "custom_pipeline_args", {}) or {})
        config_path = custom_args.get("ex_omni_config")
        if not config_path:
            raise ValueError(
                "custom_pipeline_args.ex_omni_config is required for "
                "ExOmniWan21StreamingPipeline"
            )

        from ex_omni.config import load_config
        from ex_omni.model.video_generator.inference import WanInferenceBackend

        config = load_config(Path(config_path).expanduser())
        video_config = dict(config["video"])
        # vLLM-Omni owns the worker process. Keep the validated low-latency
        # topology inside that worker: one DiT GPU plus an asynchronous VAE
        # on the second visible GPU, without creating a torchrun process group.
        topology = {
            "distributed": False,
            "use_fsdp": False,
            "sequence_parallel": 1,
            "rank0_vae": False,
        }
        topology.update(dict(custom_args.get("video_overrides") or {}))
        video_config.update(topology)
        if str(video_config.get("mode")) != "streaming":
            raise ValueError("vLLM-Omni Wan2.1 adapter requires video.mode=streaming")
        self.backend = WanInferenceBackend(video_config)
        self._config_path = str(Path(config_path).expanduser().resolve())
        self._sessions: dict[str, Any] = {}

    def load_weights(self, weights) -> set[str]:
        # The backend owns checkpoint loading from the Ex-Omni YAML. The custom
        # module intentionally declares no vLLM weight sources.
        unexpected = next(iter(weights), None)
        if unexpected is not None:
            raise RuntimeError(
                "vLLM-Omni unexpectedly supplied stock diffusion weights "
                "to the custom Wan2.1 runtime"
            )
        return set()

    @staticmethod
    def _prompt_payload(req) -> dict[str, Any]:
        if len(req.prompts) != 1:
            raise ValueError("Ex-Omni Wan2.1 supports one request per worker")
        prompt = req.prompts[0]
        if isinstance(prompt, str):
            return {"prompt": prompt}
        return dict(prompt)

    @staticmethod
    def _speech_units(payload: dict[str, Any]) -> np.ndarray:
        units = payload.get("speech_units")
        if units is None:
            units = payload.get("multi_modal_data", {}).get("speech_units")
        values = np.asarray(units, dtype=np.int64)
        if values.ndim == 2 and values.shape[0] == 16 and values.shape[1] != 16:
            values = values.T
        if values.ndim != 2 or values.shape[1] != 16:
            raise ValueError(
                f"speech_units must have shape [N,16], got {values.shape}"
            )
        return np.ascontiguousarray(values)

    @staticmethod
    def _session_id(payload: dict[str, Any], sampling) -> str:
        session_id = payload.get("session_id")
        if session_id is None:
            extra_args = dict(getattr(sampling, "extra_args", {}) or {})
            session_id = extra_args.get("session_id")
        if session_id is None:
            raise ValueError("session_id is required for incremental Video requests")
        return str(session_id)

    @staticmethod
    def _request_overrides(payload: dict[str, Any], sampling) -> dict[str, Any]:
        overrides = dict(payload.get("video_overrides") or {})
        if sampling.height is not None and sampling.width is not None:
            height, width = int(sampling.height), int(sampling.width)
            overrides["image_sizes_720"] = [[height, width]]
        return overrides

    @staticmethod
    def _empty_frames() -> np.ndarray:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    @classmethod
    def _format_chunks(
        cls,
        chunks,
        *,
        operation: str,
        started: float,
        setup_timing_seconds: dict[str, float] | None = None,
    ):
        from vllm_omni.diffusion.data import DiffusionOutput

        normalized = []
        frame_arrays = []
        stage_durations: dict[str, float] = {
            "total_seconds": time.perf_counter() - started,
        }
        for chunk in chunks:
            frames = np.ascontiguousarray(chunk["frames"])
            frame_arrays.append(frames)
            metadata = {
                key: value
                for key, value in dict(chunk).items()
                if key != "frames"
            }
            metadata["timing_seconds"] = {
                key: float(value)
                for key, value in dict(
                    metadata.get("timing_seconds", {})
                ).items()
            }
            normalized.append(metadata)
            for name, value in metadata["timing_seconds"].items():
                stage_durations[name] = stage_durations.get(name, 0.0) + value
        frames = (
            np.concatenate(frame_arrays, axis=0)
            if frame_arrays
            else cls._empty_frames()
        )
        return DiffusionOutput(
            output={
                "frames": torch.from_numpy(np.ascontiguousarray(frames)),
                "operation": operation,
                "payloads": normalized,
                "setup_timing_seconds": dict(setup_timing_seconds or {}),
            },
            stage_durations=stage_durations,
        )

    def _start_session(self, payload: dict[str, Any], sampling):
        session_id = self._session_id(payload, sampling)
        if session_id in self._sessions:
            raise RuntimeError(f"Video session {session_id!r} is already active")
        prompt = str(payload.get("prompt") or "")
        if not prompt:
            raise ValueError("VTP prompt is required")
        ref_image = payload.get("ref_image")
        if ref_image is None:
            ref_image = payload.get("multi_modal_data", {}).get("image")
        if not isinstance(ref_image, (str, Path)):
            raise TypeError(
                "ref_image must be a local path so the existing crop/pad "
                "and reference-VAE path remains unchanged"
            )
        seed = int(sampling.seed if sampling.seed is not None else 42)
        session = self.backend.start_stream(
            prompt=prompt,
            ref_image=Path(ref_image),
            seed=seed,
            overrides=self._request_overrides(payload, sampling),
        )
        self._sessions[session_id] = session
        return session

    def forward(self, req):
        from vllm_omni.diffusion.data import DiffusionOutput

        if req.is_dummy_run():
            return DiffusionOutput(
                output=torch.zeros((1, 1, 1, 3), dtype=torch.uint8)
            )

        payload = self._prompt_payload(req)
        sampling = req.sampling_params
        operation = str(payload.get("operation") or "generate").lower()
        started = time.perf_counter()

        if operation == "start":
            session = self._start_session(payload, sampling)
            return self._format_chunks(
                [],
                operation=operation,
                started=started,
                setup_timing_seconds=dict(
                    getattr(session, "setup_timing_seconds", {})
                ),
            )

        if operation in {"push", "finish", "abort"}:
            session_id = self._session_id(payload, sampling)
            session = self._sessions.get(session_id)
            if session is None:
                raise RuntimeError(f"Video session {session_id!r} is not active")
            if operation == "push":
                chunks = session.push_units(self._speech_units(payload))
            elif operation == "finish":
                self._sessions.pop(session_id, None)
                chunks = session.finish()
            else:
                self._sessions.pop(session_id, None)
                session.abort()
                chunks = []
            return self._format_chunks(
                chunks,
                operation=operation,
                started=started,
            )

        if operation != "generate":
            raise ValueError(f"unknown Video operation: {operation!r}")

        prompt = str(payload.get("prompt") or "")
        if not prompt:
            raise ValueError("VTP prompt is required")
        ref_image = payload.get("ref_image")
        if ref_image is None:
            ref_image = payload.get("multi_modal_data", {}).get("image")
        if not isinstance(ref_image, (str, Path)):
            raise TypeError(
                "ref_image must be a local path so the existing crop/pad "
                "and reference-VAE path remains unchanged"
            )
        seed = int(sampling.seed if sampling.seed is not None else 42)
        session = self.backend.start_stream(
            prompt=prompt,
            ref_image=Path(ref_image),
            seed=seed,
            overrides=self._request_overrides(payload, sampling),
        )
        chunks = session.push_units(self._speech_units(payload))
        chunks.extend(session.finish())
        if not chunks:
            raise RuntimeError("Wan2.1 Streaming produced no video chunks")
        return self._format_chunks(
            chunks,
            operation=operation,
            started=started,
            setup_timing_seconds=dict(
                getattr(session, "setup_timing_seconds", {})
            ),
        )
