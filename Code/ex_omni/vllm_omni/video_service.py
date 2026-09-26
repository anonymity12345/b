"""vLLM-Omni Video service with Ex-Omni's incremental IPC contract."""

from __future__ import annotations

from multiprocessing.connection import Listener
import queue
from pathlib import Path
import time
import traceback
import threading
from typing import Any
import uuid

import numpy as np

from ex_omni.video_service import _authkey, _send_response


class VllmOmniVideoRuntime:
    """Own one resident Omni engine and one active Wan2.1 stream."""

    def __init__(self, config_path: str | Path, model_path: str | Path) -> None:
        from ex_omni.config import load_config
        from ex_omni.vllm_omni.video_pipeline import (
            register_vllm_omni_pipeline,
        )

        self.config_path = Path(config_path).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.config = load_config(self.config_path)
        self.video_config = dict(self.config["video"])
        register_vllm_omni_pipeline()

        from vllm_omni.entrypoints.omni import Omni

        started = time.perf_counter()
        self.omni = Omni(
            model=str(self.model_path),
            diffusion_load_format="dummy",
            custom_pipeline_args={
                "pipeline_class": (
                    "ex_omni.vllm_omni.video_pipeline."
                    "ExOmniWan21StreamingPipeline"
                ),
                "ex_omni_config": str(self.config_path),
            },
            enforce_eager=True,
        )
        self.model_load_seconds = time.perf_counter() - started
        self.active: dict[str, Any] | None = None

    @staticmethod
    def _condition(kwargs: dict[str, Any]) -> str:
        video_prompt = kwargs.pop("video_prompt", None)
        vtp = kwargs.pop("vtp", None)
        strict_vtp = bool(kwargs.pop("strict_vtp", True))
        if (video_prompt is None) == (vtp is None):
            raise ValueError("provide exactly one of video_prompt or vtp")
        if video_prompt is not None:
            return str(video_prompt)
        from ex_omni.vtp import vtp_to_video_prompt

        return vtp_to_video_prompt(
            str(vtp),
            strict=strict_vtp,
        )

    def _sampling(self, context: dict[str, Any]):
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        kwargs: dict[str, Any] = {
            "seed": int(context["seed"]),
            "num_inference_steps": int(
                self.video_config.get("steps", 8)
            ),
            "output_type": "pt",
        }
        if context.get("height") is not None:
            kwargs["height"] = int(context["height"])
            kwargs["width"] = int(context["width"])
        return OmniDiffusionSamplingParams(**kwargs)

    @staticmethod
    def _resolution(overrides: dict[str, Any]) -> tuple[int, int] | None:
        sizes = overrides.get("image_sizes_720")
        if (
            isinstance(sizes, (list, tuple))
            and sizes
            and isinstance(sizes[0], (list, tuple))
            and len(sizes[0]) == 2
        ):
            return int(sizes[0][0]), int(sizes[0][1])
        return None

    def _request(self, payload: dict[str, Any], context: dict[str, Any]):
        outputs = self.omni.generate(
            payload,
            self._sampling(context),
            use_tqdm=False,
        )
        if len(outputs) != 1:
            raise RuntimeError(
                f"vLLM-Omni returned {len(outputs)} Video outputs"
            )
        output = outputs[0]
        if output.error:
            raise RuntimeError(output.error)
        metadata = dict(output.multimodal_output.get("metadata", {}))
        ex_metadata = dict(metadata.get("ex_omni", {}))
        chunk_metadata = list(ex_metadata.get("payloads", []))
        frames = (
            output.images[0]
            if output.images
            else np.empty((0, 0, 0, 3), dtype=np.uint8)
        )
        if hasattr(frames, "detach"):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        cursor = 0
        payloads = []
        for item in chunk_metadata:
            item = dict(item)
            frame_count = int(item["valid_units"]) * 2
            item["frames"] = np.ascontiguousarray(
                frames[cursor : cursor + frame_count]
            )
            cursor += frame_count
            payloads.append(item)
        if cursor != len(frames):
            raise RuntimeError(
                "vLLM-Omni Video frame metadata mismatch: "
                f"described={cursor}, returned={len(frames)}"
            )
        return {
            "payloads": payloads,
            "setup_timing_seconds": dict(
                ex_metadata.get("setup_timing_seconds", {})
            ),
            "stage_durations": dict(output.stage_durations),
        }

    def start(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.active is not None:
            raise RuntimeError("a Video session is already active")
        request = dict(kwargs)
        prompt = self._condition(request)
        ref_image = Path(request.pop("ref_image")).expanduser().resolve()
        seed_value = request.pop("seed", None)
        seed = int(
            self.video_config.get("seed", 42)
            if seed_value is None
            else seed_value
        )
        resolution = self._resolution(request)
        session_id = uuid.uuid4().hex
        context = {
            "session_id": session_id,
            "seed": seed,
            "height": resolution[0] if resolution else None,
            "width": resolution[1] if resolution else None,
        }
        result = self._request(
            {
                "operation": "start",
                "session_id": session_id,
                "prompt": prompt,
                "ref_image": str(ref_image),
                "video_overrides": request,
            },
            context,
        )
        self.active = context
        return result

    def push(self, units) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("no active Video session")
        return self._request(
            {
                "operation": "push",
                "session_id": self.active["session_id"],
                "speech_units": np.asarray(units, dtype=np.int64),
            },
            self.active,
        )

    def finish(self) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("no active Video session")
        context, self.active = self.active, None
        return self._request(
            {
                "operation": "finish",
                "session_id": context["session_id"],
            },
            context,
        )

    def abort(self) -> dict[str, Any]:
        if self.active is None:
            return {"payloads": [], "setup_timing_seconds": {}}
        context, self.active = self.active, None
        return self._request(
            {
                "operation": "abort",
                "session_id": context["session_id"],
            },
            context,
        )

    def close(self) -> None:
        if self.active is not None:
            try:
                self.abort()
            except Exception:
                traceback.print_exc()
        self.omni.close()


def serve_vllm_omni_video(
    config_path: str | Path,
    *,
    model_path: str | Path,
    address: str | Path,
    authkey: str | bytes | None = None,
) -> int:
    """Serve the existing RemoteVideoGenerator protocol through vLLM-Omni."""
    import torch

    runtime = VllmOmniVideoRuntime(config_path, model_path)
    socket_path = Path(address).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    listener = Listener(
        str(socket_path),
        family="AF_UNIX",
        authkey=_authkey(authkey),
    )
    print(
        f"[vllm-omni-video] ready address={socket_path} "
        f"load_seconds={runtime.model_load_seconds:.3f}",
        flush=True,
    )
    connection = None
    response_queue = None
    response_sender = None
    response_sender_errors = []
    try:
        connection = listener.accept()
        response_queue = queue.Queue(maxsize=2)

        def send_responses() -> None:
            try:
                while True:
                    response = response_queue.get()
                    if response is None:
                        return
                    _send_response(connection, response)
            except Exception as exc:
                response_sender_errors.append(exc)

        response_sender = threading.Thread(
            target=send_responses,
            name="vllm-video-response-sender",
            daemon=True,
        )
        response_sender.start()
        while True:
            command = connection.recv()
            operation = str(command.get("operation", ""))
            if operation == "disconnect":
                response_queue.put({"ok": True})
                break
            if operation == "ping":
                physical_gpu_count = torch.cuda.device_count()
                response_queue.put(
                    {
                        "ok": True,
                        "world_size": 1,
                        "physical_gpu_count": physical_gpu_count,
                        "vae_offload_enabled": physical_gpu_count > 1,
                        "vae_device": (
                            "cuda:1" if physical_gpu_count > 1 else "cuda:0"
                        ),
                        "model_load_seconds": runtime.model_load_seconds,
                        "compute_seconds": 0.0,
                    }
                )
                continue

            started = time.perf_counter()
            try:
                if operation == "start":
                    result = runtime.start(dict(command["kwargs"]))
                elif operation == "push":
                    result = runtime.push(command["units"])
                elif operation == "finish":
                    result = runtime.finish()
                elif operation == "abort":
                    result = runtime.abort()
                else:
                    raise ValueError(
                        f"unknown Video operation: {operation!r}"
                    )
                torch.cuda.synchronize()
                response = {
                    "ok": True,
                    "error": None,
                    "payloads": result.get("payloads", []),
                    "compute_seconds": time.perf_counter() - started,
                    "session_setup_timing_seconds": result.get(
                        "setup_timing_seconds", {}
                    ),
                    "stage_durations": result.get("stage_durations", {}),
                }
            except Exception as exc:
                traceback.print_exc()
                response = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "payloads": [],
                    "compute_seconds": time.perf_counter() - started,
                    "session_setup_timing_seconds": {},
                }
            response_queue.put(response)
            if response_sender_errors:
                raise response_sender_errors[0]
    except EOFError:
        return 0
    finally:
        if response_queue is not None:
            response_queue.put(None)
        if response_sender is not None:
            response_sender.join()
        if connection is not None:
            connection.close()
        listener.close()
        runtime.close()
        socket_path.unlink(missing_ok=True)
    return 0
