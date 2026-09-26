"""Local IPC boundary between Dialogue Model/Speech and video workers."""

from __future__ import annotations

from multiprocessing.connection import Client, Listener
import queue
import os
from pathlib import Path
import time
import threading
from typing import Any, Mapping


DEFAULT_AUTHKEY_ENV = "EX_OMNI_VIDEO_AUTHKEY"
_RAW_FRAME_KEY = "__ex_omni_raw_frame__"


def _authkey(value: str | bytes | None) -> bytes:
    if isinstance(value, bytes):
        return value
    resolved = value or os.environ.get(DEFAULT_AUTHKEY_ENV, "ex-omni-local")
    return resolved.encode("utf-8")


def _apply_video_service_environment(
    config: dict[str, Any],
    environment: Mapping[str, str] | None = None,
) -> None:
    env = os.environ if environment is None else environment
    if env.get("EX_OMNI_VAE_DEVICE"):
        config.setdefault("video", {})["vae_device"] = env["EX_OMNI_VAE_DEVICE"]


def _poll_video_command(connection):
    if not connection.poll(1.0):
        return None
    try:
        return connection.recv()
    except EOFError:
        return {"operation": "disconnect"}


def _send_response(connection, response: Mapping[str, Any]) -> None:
    """Send frame arrays outside the control pickle on a sender thread."""
    wire_response = dict(response)
    raw_frames = []
    payloads = response.get("payloads")
    if payloads:
        import numpy as np

        wire_payloads = []
        for payload in payloads:
            wire_payload = dict(payload)
            frames = wire_payload.get("frames")
            if isinstance(frames, np.ndarray):
                frames = np.ascontiguousarray(frames)
                wire_payload["frames"] = {
                    "shape": list(frames.shape),
                    "dtype": frames.dtype.str,
                    "nbytes": int(frames.nbytes),
                    _RAW_FRAME_KEY: True,
                }
                raw_frames.append(frames)
            wire_payloads.append(wire_payload)
        wire_response["payloads"] = wire_payloads

    connection.send(wire_response)
    for frames in raw_frames:
        connection.send_bytes(memoryview(frames).cast("B"))


def _recv_response(connection) -> Any:
    response = connection.recv()
    if not isinstance(response, dict):
        return response
    payloads = response.get("payloads")
    if not payloads:
        return response

    import numpy as np

    for payload in payloads:
        descriptor = payload.get("frames")
        if not (
            isinstance(descriptor, dict)
            and descriptor.get(_RAW_FRAME_KEY) is True
        ):
            continue
        shape = tuple(int(value) for value in descriptor["shape"])
        dtype = np.dtype(descriptor["dtype"])
        frames = np.empty(shape, dtype=dtype)
        expected = int(descriptor["nbytes"])
        if frames.nbytes != expected:
            raise RuntimeError(
                "video service returned inconsistent frame metadata: "
                f"shape/dtype={frames.nbytes} bytes, declared={expected} bytes"
            )
        received = connection.recv_bytes_into(memoryview(frames).cast("B"))
        if received != expected:
            raise RuntimeError(
                "video service returned a truncated frame buffer: "
                f"received={received}, expected={expected}"
            )
        payload["frames"] = frames
    return response


class RemoteVideoStreamSession:
    """Synchronous session proxy; callers may place it behind a worker queue."""

    def __init__(
        self,
        generator: "RemoteVideoGenerator",
        progress_callback=None,
    ) -> None:
        self._generator = generator
        self._progress_callback = progress_callback
        self.last_compute_seconds = 0.0
        self.setup_timing_seconds: dict[str, float] = {}
        self._finished = False
        self._pending_responses = 0

    def push_units(self, units):
        if self._finished:
            raise RuntimeError("remote video stream is already finished")
        response = self._generator._request(
            {"operation": "push", "units": units},
            progress_callback=self._progress_callback,
        )
        self.last_compute_seconds = float(response["compute_seconds"])
        return response["payloads"]

    def submit_units(self, units) -> None:
        if self._finished:
            raise RuntimeError("remote video stream is already finished")
        self._generator._send_request(
            {"operation": "push", "units": units}
        )
        self._pending_responses += 1

    def submit_finish(self) -> None:
        if self._finished:
            raise RuntimeError("remote video stream is already finished")
        self._generator._send_request({"operation": "finish"})
        self._pending_responses += 1
        self._finished = True

    def submit_abort(self) -> None:
        if self._finished:
            return
        self._generator._send_request({"operation": "abort"})
        self._pending_responses += 1
        self._finished = True

    def receive_response(self):
        if self._pending_responses <= 0:
            raise RuntimeError("remote video stream has no pending response")
        response = self._generator._receive_response(
            progress_callback=self._progress_callback,
        )
        self._pending_responses -= 1
        self.last_compute_seconds = float(response["compute_seconds"])
        return response["payloads"], self.last_compute_seconds

    def finish(self, chunk_callback=None):
        if self._finished:
            return []
        response = self._generator._request(
            {"operation": "finish", "stream_payloads": callable(chunk_callback)},
            progress_callback=self._progress_callback,
            payload_callback=chunk_callback,
        )
        self.last_compute_seconds = float(response["compute_seconds"])
        self._finished = True
        return response["payloads"]

    def abort(self) -> None:
        if self._finished:
            return
        self._generator._request({"operation": "abort"})
        self._finished = True


class RemoteVideoGenerator:
    """Connect a Dialogue Model process to an independent video service."""

    is_remote = True

    def __init__(
        self,
        address: str | Path,
        *,
        authkey: str | bytes | None = None,
        connect_timeout_seconds: float = 600.0,
    ) -> None:
        self.address = str(Path(address).expanduser().resolve())
        self.authkey = _authkey(authkey)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self._connection = None
        self.world_size: int | None = None
        self.model_load_seconds: float | None = None
        self.vae_offload_enabled: bool | None = None
        self.vae_device: str | None = None
        self.physical_gpu_count: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._connection is not None

    def ensure_loaded(self):
        if self._connection is not None:
            return self
        deadline = time.monotonic() + self.connect_timeout_seconds
        last_error = None
        while time.monotonic() < deadline:
            try:
                self._connection = Client(
                    self.address,
                    family="AF_UNIX",
                    authkey=self.authkey,
                )
                response = self._request({"operation": "ping"})
                self.world_size = int(response["world_size"])
                self.model_load_seconds = float(response["model_load_seconds"])
                self.vae_offload_enabled = bool(response["vae_offload_enabled"])
                self.vae_device = str(response["vae_device"])
                self.physical_gpu_count = int(response["physical_gpu_count"])
                return self
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_error = exc
                self._connection = None
                time.sleep(0.25)
        raise TimeoutError(
            f"video service was not reachable at {self.address}: {last_error}"
        )

    def _send_request(self, payload: Mapping[str, Any]) -> None:
        self.ensure_loaded()
        self._connection.send(dict(payload))

    def _receive_response(self, progress_callback=None, payload_callback=None) -> dict[str, Any]:
        response = _recv_response(self._connection)
        while (
            isinstance(response, dict)
            and response.get("type") in ("diffusion_progress", "video_payload")
        ):
            if response.get("type") == "video_payload":
                if not callable(payload_callback):
                    raise RuntimeError("Unexpected incremental video payload")
                for payload in response["payloads"]:
                    payload_callback(payload)
            elif callable(progress_callback):
                progress_callback(
                    int(response["step"]),
                    int(response["total_steps"]),
                    int(response.get("chunk", 1)),
                    response.get("total_chunks"),
                )
            response = _recv_response(self._connection)
        if not isinstance(response, dict):
            raise RuntimeError("video service returned a malformed response")
        if not response.get("ok", False):
            raise RuntimeError(
                "video service request failed: "
                + str(response.get("error", "unknown error"))
            )
        return response

    def _request(
        self,
        payload: Mapping[str, Any],
        progress_callback=None,
        payload_callback=None,
    ) -> dict[str, Any]:
        self._send_request(payload)
        return self._receive_response(progress_callback=progress_callback, payload_callback=payload_callback)

    def start_stream(self, **kwargs):
        progress_callback = kwargs.pop("progress_callback", None)
        kwargs["_report_diffusion_progress"] = callable(progress_callback)
        response = self._request({"operation": "start", "kwargs": kwargs})
        session = RemoteVideoStreamSession(
            self,
            progress_callback=progress_callback,
        )
        session.last_compute_seconds = float(response["compute_seconds"])
        session.setup_timing_seconds = dict(
            response.get("session_setup_timing_seconds", {})
        )
        return session

    def generate(self, **kwargs):
        """Run a full-sequence request in the independent video worker."""
        from .schemas import SpeechTokens, VideoResult

        import tempfile

        preview_callback = kwargs.pop("preview_callback", None)
        progress_callback = kwargs.pop("progress_callback", None)

        def on_preview(payload):
            if callable(preview_callback):
                preview_callback(
                    payload["frames"], float(payload["fps"]),
                    int(payload["chunk"]), int(payload["total_chunks"]),
                )

        with tempfile.TemporaryDirectory(prefix="ex-omni-full-sequence-") as directory:
            tokens = kwargs.get("speech_tokens")
            if isinstance(tokens, SpeechTokens):
                kwargs["speech_tokens"] = str(tokens.save(Path(directory) / "speech_tokens.npy"))
            kwargs["_report_preview"] = callable(preview_callback)
            kwargs["_report_diffusion_progress"] = callable(progress_callback)
            response = self._request(
                {"operation": "generate", "kwargs": kwargs},
                progress_callback=progress_callback,
                payload_callback=on_preview,
            )
        result = response["result"]
        return VideoResult(
            output_path=Path(result["output_path"]),
            mode=str(result["mode"]),
            frames=int(result["frames"]),
            fps=float(result["fps"]),
            metadata=dict(result.get("metadata", {})),
        )

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.send({"operation": "disconnect"})
            connection.recv()
        except (EOFError, OSError):
            pass
        finally:
            connection.close()


def _synchronize_cuda(context) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(context.local_rank)


def _critical_seconds(local_seconds: float, context) -> float:
    if not context.is_distributed:
        return float(local_seconds)
    import torch
    import torch.distributed as dist

    value = torch.tensor(
        [float(local_seconds)],
        dtype=torch.float64,
        device=torch.device("cuda", context.local_rank),
    )
    dist.all_reduce(
        value,
        op=dist.ReduceOp.MAX,
        group=context.process_group,
    )
    return float(value.item())


def serve_video(
    config_path: str | Path,
    *,
    address: str | Path,
    authkey: str | bytes | None = None,
) -> int:
    """Serve sequential video requests from an independent torchrun world."""
    from .config import load_config, normalize_video_config
    from .distributed.context import initialize_distributed
    from .execution import derive_request_topology_config
    from .video import OmniAvatarVideoGenerator

    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    source = load_config(config_path)
    _apply_video_service_environment(source)
    if (source.get("video", {}).get("compile", {}).get("enabled", False)
            or source.get("video", {}).get("sequence_parallel_mode") == "ulysses"
            or source.get("video", {}).get("cache_rotated_history", False)
            or source.get("graph", False)):
        # Inductor synchronizes the device when loading a new kernel. If ranks
        # advance past unfinished collectives, this can wait on a peer which
        # is itself compiling, causing deadlock. Complete each collective
        # before advancing; configure this before creating the NCCL group.
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    context = initialize_distributed("nccl")
    config = derive_request_topology_config(source, world_size=context.world_size)
    config["video"] = normalize_video_config(config["video"])
    generator = OmniAvatarVideoGenerator(config)
    load_started = time.perf_counter()
    backend = generator.ensure_loaded()
    runtime = getattr(backend, "_runtime", None)
    vae_offload_enabled = bool(getattr(runtime, "vae_pipeline_enabled", False))
    vae_device = str(getattr(runtime, "vae_device", "unknown"))
    physical_gpu_count = context.world_size + int(vae_offload_enabled)
    model_load_seconds = _critical_seconds(
        time.perf_counter() - load_started,
        context,
    )

    socket_path = Path(address).expanduser().resolve()
    listener = None
    connection = None
    session = None
    if context.is_rank0:
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        socket_path.unlink(missing_ok=True)
        listener = Listener(
            str(socket_path),
            family="AF_UNIX",
            authkey=_authkey(authkey),
        )
        print(
            f"[video-service] ready address={socket_path} "
            f"world_size={context.world_size}",
            flush=True,
        )
        connection = listener.accept()

    response_queue = None
    response_sender = None
    response_sender_errors = []
    if context.is_rank0:
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
            name="native-video-response-sender",
            daemon=True,
        )
        response_sender.start()

    try:
        while True:
            command = [_poll_video_command(connection) if context.is_rank0 else None]
            if context.is_distributed:
                import torch.distributed as dist

                dist.broadcast_object_list(
                    command,
                    src=0,
                    group=context.process_group,
                )
            command = command[0]
            if command is None:
                continue
            operation = str(command.get("operation", ""))
            if operation == "disconnect":
                if context.is_rank0:
                    response_queue.put({"ok": True})
                break
            if operation == "ping":
                if context.is_rank0:
                    response_queue.put(
                        {
                            "ok": True,
                            "world_size": context.world_size,
                            "physical_gpu_count": physical_gpu_count,
                            "vae_offload_enabled": vae_offload_enabled,
                            "vae_device": vae_device,
                            "model_load_seconds": model_load_seconds,
                            "compute_seconds": 0.0,
                        }
                    )
                continue

            local_error = None
            payloads = []
            result = None
            session_setup_timing_seconds = {}
            started = time.perf_counter()
            try:
                if operation == "generate":
                    generate_kwargs = dict(command["kwargs"])
                    report_preview = bool(generate_kwargs.pop("_report_preview", False))
                    report_progress = bool(generate_kwargs.pop("_report_diffusion_progress", False))
                    if report_preview and context.is_rank0:
                        def report_preview_window(frames, fps, chunk, total_chunks):
                            response_queue.put({
                                "type": "video_payload",
                                "payloads": [{
                                    "frames": frames,
                                    "fps": float(fps),
                                    "chunk": int(chunk),
                                    "total_chunks": int(total_chunks),
                                }],
                            })

                        generate_kwargs["preview_callback"] = report_preview_window
                    if report_progress and context.is_rank0:
                        def report_full_progress(step, total_steps, chunk=1, total_chunks=None):
                            response_queue.put({
                                "type": "diffusion_progress",
                                "step": int(step),
                                "total_steps": int(total_steps),
                                "chunk": int(chunk),
                                "total_chunks": int(total_chunks) if total_chunks is not None else None,
                            })

                        generate_kwargs["progress_callback"] = report_full_progress
                    result = generator.generate(**generate_kwargs)
                elif operation == "start":
                    if session is not None:
                        raise RuntimeError("a video session is already active")
                    start_kwargs = dict(command["kwargs"])
                    report_progress = bool(
                        start_kwargs.pop(
                            "_report_diffusion_progress",
                            False,
                        )
                    )
                    if report_progress and context.is_rank0:
                        def report_diffusion_progress(
                            step,
                            total_steps,
                            chunk=1,
                            total_chunks=None,
                        ):
                            response_queue.put(
                                {
                                    "type": "diffusion_progress",
                                    "step": int(step),
                                    "total_steps": int(total_steps),
                                    "chunk": int(chunk),
                                    "total_chunks": (
                                        int(total_chunks)
                                        if total_chunks is not None
                                        else None
                                    ),
                                }
                            )

                        start_kwargs["progress_callback"] = (
                            report_diffusion_progress
                        )
                    session = generator.start_stream(**start_kwargs)
                    session_setup_timing_seconds = dict(
                        getattr(session, "setup_timing_seconds", {})
                    )
                elif operation == "push":
                    if session is None:
                        raise RuntimeError("no active video session")
                    payloads = session.push_units(command["units"])
                elif operation == "finish":
                    if session is None:
                        raise RuntimeError("no active video session")
                    if command.get("stream_payloads"):
                        callback = None
                        if context.is_rank0:
                            def callback(payload):
                                # Later timing aggregation replaces timing_seconds.
                                # Freeze the envelope before the sender consumes it.
                                response_queue.put({
                                    "type": "video_payload",
                                    "payloads": [dict(payload)],
                                })
                        payloads = session.finish(chunk_callback=callback)
                    else:
                        payloads = session.finish()
                    session = None
                elif operation == "abort":
                    if session is not None:
                        abort = getattr(session, "abort", None)
                        if callable(abort):
                            abort()
                    session = None
                else:
                    raise ValueError(f"unknown video operation: {operation}")
                _synchronize_cuda(context)
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            local_seconds = time.perf_counter() - started

            errors = [local_error]
            if context.is_distributed:
                import torch.distributed as dist

                errors = [None] * context.world_size
                dist.all_gather_object(
                    errors,
                    local_error,
                    group=context.process_group,
                )
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            ]
            compute_seconds = _critical_seconds(local_seconds, context)
            # Every rank must enter the same collectives. With an asynchronous
            # rank-0 VAE, ranks can return different numbers of completed chunks.
            if context.is_distributed:
                gathered_payload_timings = [None] * context.world_size
                dist.all_gather_object(
                    gathered_payload_timings,
                    {int(item["start_frame"]): dict(item.get("timing_seconds", {}))
                     for item in payloads},
                    group=context.process_group,
                )
                if context.is_rank0:
                    for payload in payloads:
                        frame = int(payload["start_frame"])
                        keys = {
                            key
                            for rank_timings in gathered_payload_timings
                            for key in rank_timings.get(frame, {})
                        }
                        payload["timing_seconds"] = {
                            key: max(
                                float(rank_timings.get(frame, {}).get(key, 0.0))
                                for rank_timings in gathered_payload_timings
                            )
                            for key in keys
                        }
            if context.is_rank0:
                response_queue.put(
                    {
                        "ok": not failures,
                        "error": "; ".join(failures) if failures else None,
                        "result": (
                            {"output_path": str(result.output_path), "mode": result.mode,
                             "frames": result.frames, "fps": result.fps,
                             "metadata": dict(result.metadata)}
                            if operation == "generate" and not failures else None
                        ),
                        "payloads": payloads if not failures and not (operation == "finish" and command.get("stream_payloads")) else [],
                        "compute_seconds": compute_seconds,
                        "session_setup_timing_seconds": (
                            session_setup_timing_seconds
                        ),
                    }
                )
                if response_sender_errors:
                    raise response_sender_errors[0]
            if failures:
                session = None
    except EOFError:
        return 0
    finally:
        if response_queue is not None:
            response_queue.put(None)
        if response_sender is not None:
            response_sender.join()
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        if context.is_rank0:
            socket_path.unlink(missing_ok=True)
        if context.is_distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
    return 0
