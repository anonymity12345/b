"""End-to-end orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import queue
import threading
from typing import Any, Mapping
import uuid

from .dialogue_model import ExOmniDialogueModel
from .media import ffmpeg_executable
from .schemas import (
    Cancelled,
    Completed,
    DiffusionProgress,
    EndToEndResult,
    VTPDelta,
    VTPReady,
    SpeechAudioChunk,
    SpeechTokens,
    SpeechUnitsChunk,
    StreamError,
    TextDelta,
    VideoChunk,
    VideoResult,
)
from .video import OmniAvatarVideoGenerator
from .vtp import apply_vtp_overrides


class ExOmni2DPipeline:
    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        *,
        dialogue_model: ExOmniDialogueModel | Any | None = None,
        video_generator: OmniAvatarVideoGenerator | Any | None = None,
    ) -> None:
        if config is None and (dialogue_model is None or video_generator is None):
            raise ValueError("config is required unless both backends are injected")
        from .config import load_config

        payload = (
            load_config(config)
            if isinstance(config, (str, Path))
            else dict(config or {})
        )
        runtime = dict(payload.get("runtime", {}))
        self._ffmpeg_path = runtime.get("ffmpeg_path")
        self.dialogue_model = dialogue_model or ExOmniDialogueModel(config)  # type: ignore[arg-type]
        if video_generator is None:
            service_address = os.environ.get(
                "EX_OMNI_VIDEO_SERVICE_ADDRESS"
            ) or runtime.get("video_service_address")
            if service_address:
                from .video_service import RemoteVideoGenerator

                video_generator = RemoteVideoGenerator(
                    service_address,
                    authkey=runtime.get("video_service_authkey"),
                    connect_timeout_seconds=float(
                        runtime.get(
                            "video_service_connect_timeout_seconds",
                            600,
                        )
                    ),
                )
            else:
                video_generator = OmniAvatarVideoGenerator(config)  # type: ignore[arg-type]
        self.video_generator = video_generator
        self._session_role_inputs: dict[str, tuple[Path, Path, str | None]] = {}
        self._session_lock = threading.RLock()

    @staticmethod
    def _mux_generated_audio(
        video_path: Path,
        audio_path: Path,
        fps: float,
        frame_count: int,
        ffmpeg_path: str | Path | None = None,
    ) -> None:
        """Atomically add generated speech to an already-rendered MP4."""
        import os
        import subprocess

        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if fps <= 0 or frame_count <= 0:
            raise ValueError(
                f"invalid mux duration: frames={frame_count}, fps={fps}"
            )
        duration = frame_count / fps
        delay_ms = round(1000.0 / fps)
        temporary = video_path.with_name(
            f".{video_path.stem}.muxing-{os.getpid()}{video_path.suffix}"
        )
        try:
            print(
                f"[video] muxing generated speech into MP4: {video_path}",
                flush=True,
            )
            subprocess.run(
                [
                    ffmpeg_executable(ffmpeg_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-i",
                    str(audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-af",
                    (
                        f"adelay={delay_ms}:all=1,"
                        f"apad=whole_dur={duration:.6f},"
                        f"atrim=duration={duration:.6f}"
                    ),
                    "-t",
                    f"{duration:.6f}",
                    str(temporary),
                ],
                check=True,
                timeout=120,
            )
            os.replace(temporary, video_path)
            print(f"[video] audio mux complete: {video_path}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)

    def _resolve_session_role_inputs(
        self,
        session_id: str,
        *,
        ref_image: str | Path | None,
        ref_audio: str | Path | None,
        role_card: str | None,
    ) -> tuple[Path, Path, str | None]:
        requested_image = (
            Path(ref_image).expanduser().resolve()
            if ref_image is not None
            else None
        )
        requested_audio = (
            Path(ref_audio).expanduser().resolve()
            if ref_audio is not None
            else None
        )
        requested_role = str(role_card).strip() if role_card else None
        with self._session_lock:
            existing = self._session_role_inputs.get(session_id)
            if existing is None:
                if requested_image is None:
                    raise ValueError(
                        "ref_image is required on the first request of a session"
                    )
                if requested_audio is None:
                    raise ValueError(
                        "ref_audio is required on the first request of a session"
                    )
                existing = (requested_image, requested_audio, requested_role)
                self._session_role_inputs[session_id] = existing
                return existing

            image, audio, role = existing
            if requested_image is not None and requested_image != image:
                raise ValueError(
                    "a session cannot replace its ref_image; clear it first"
                )
            if requested_audio is not None and requested_audio != audio:
                raise ValueError(
                    "a session cannot replace its ref_audio; clear it first"
                )
            if requested_role is not None and requested_role != role:
                raise ValueError(
                    "a session cannot replace or add its role_card; clear it first"
                )
            return existing

    def generate_video(
        self,
        *,
        ref_image: str | Path,
        speech_tokens: SpeechTokens | str | Path,
        output_path: str | Path,
        video_prompt: str | None = None,
        vtp: str | None = None,
        **kwargs: Any,
    ) -> VideoResult:
        return self.video_generator.generate(
            video_prompt=video_prompt,
            vtp=vtp,
            ref_image=ref_image,
            speech_tokens=speech_tokens,
            output_path=output_path,
            **kwargs,
        )

    def chat_to_video(
        self,
        text: str,
        *,
        ref_image: str | Path | None = None,
        output_path: str | Path,
        session_id: str = "default",
        role_card: str | None = None,
        ref_audio: str | Path | None = None,
        dialogue_model_kwargs: Mapping[str, Any] | None = None,
        video_kwargs: Mapping[str, Any] | None = None,
        vtp_overrides: Mapping[str, str] | None = None,
    ) -> EndToEndResult:
        from .distributed.context import current_context

        context = current_context()
        ref_image, ref_audio, role_card = self._resolve_session_role_inputs(
            session_id,
            ref_image=ref_image,
            ref_audio=ref_audio,
            role_card=role_card,
        )
        # Full-sequence runs still shard across the world: build models and
        # create every dialogue_model TP / FSDP group on the main thread so collective
        # setup never happens on a background thread.
        if context.is_distributed:
            import torch

            torch.cuda.set_device(context.local_rank)
            self.dialogue_model.ensure_loaded()
            self.video_generator.ensure_loaded()
        dialogue_model_options = dict(dialogue_model_kwargs or {})
        dialogue_model_options.setdefault("role_card", role_card)
        dialogue_model_options.setdefault("ref_audio", ref_audio)
        dialogue_model_result = self.dialogue_model.generate(
            text,
            session_id=session_id,
            ref_image=ref_image,
            **dialogue_model_options,
        )
        if not dialogue_model_result.vtp:
            raise ValueError("dialogue_model returned an empty vtp")
        effective_vtp = apply_vtp_overrides(
            dialogue_model_result.vtp,
            **dict(vtp_overrides or {}),
        )
        if effective_vtp != dialogue_model_result.vtp:
            dialogue_model_result = replace(
                dialogue_model_result,
                vtp=effective_vtp,
            )
        video_options = dict(video_kwargs or {})
        dialogue_callback = video_options.pop("dialogue_callback", None)
        if callable(dialogue_callback):
            dialogue_callback(dialogue_model_result)
        preview_callback = video_options.get("preview_callback")
        if callable(preview_callback):
            speech_preview_path = dialogue_model_result.speech_audio_path

            def preview_with_audio(frames, fps, chunk, total_chunks):
                preview_callback(
                    frames,
                    fps,
                    chunk,
                    total_chunks,
                    speech_preview_path,
                )

            video_options["preview_callback"] = preview_with_audio
        video_result = self.video_generator.generate(
            vtp=dialogue_model_result.vtp,
            ref_image=dialogue_model_result.ref_image or Path(ref_image),
            speech_tokens=dialogue_model_result.require_speech_tokens(),
            output_path=output_path,
            **video_options,
        )
        mux_error = None
        if context.is_rank0 and dialogue_model_result.speech_audio_path is not None:
            try:
                self._mux_generated_audio(
                    video_result.output_path,
                    dialogue_model_result.speech_audio_path,
                    video_result.fps,
                    int(video_result.frames or 0),
                    self._ffmpeg_path,
                )
            except Exception as exc:
                mux_error = f"{type(exc).__name__}: {exc}"
        if context.is_distributed:
            import torch.distributed as dist

            status = [mux_error]
            dist.broadcast_object_list(
                status,
                src=0,
                group=context.process_group,
            )
            mux_error = status[0]
        if mux_error is not None:
            raise RuntimeError("failed to mux generated speech: " + mux_error)
        return EndToEndResult(dialogue_model_result, video_result)

    def stream_chat_to_video(
        self,
        text: str,
        *,
        ref_image: str | Path | None = None,
        session_id: str = "default",
        role_card: str | None = None,
        ref_audio: str | Path | None = None,
        request_id: str | None = None,
        dialogue_model_kwargs: Mapping[str, Any] | None = None,
        video_kwargs: Mapping[str, Any] | None = None,
        vtp_overrides: Mapping[str, str] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """Stream VTP → text → units → Streaming frames on one compute thread."""
        from .distributed.context import current_context

        context = current_context()
        request_id = request_id or str(uuid.uuid4())
        if context.is_distributed:
            import torch.distributed as dist

            payload = [
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "text": text,
                    "ref_image": str(ref_image) if ref_image is not None else None,
                    "ref_audio": str(ref_audio) if ref_audio is not None else None,
                    "role_card": role_card,
                }
                if context.is_rank0
                else None
            ]
            dist.broadcast_object_list(payload, src=0, group=context.process_group)
            request = payload[0]
            request_id = request["request_id"]
            session_id = request["session_id"]
            text = request["text"]
            ref_image = (
                Path(request["ref_image"])
                if request["ref_image"] is not None
                else None
            )
            ref_audio = (
                Path(request["ref_audio"])
                if request["ref_audio"] is not None
                else None
            )
            role_card = request["role_card"]
        ref_image, ref_audio, role_card = self._resolve_session_role_inputs(
            session_id,
            ref_image=ref_image,
            ref_audio=ref_audio,
            role_card=role_card,
        )
        # Build models and create every distributed group (native TP for the
        # dialogue_model, FSDP for the Streaming DiT) on the main thread. The compute
        # worker below only runs forward passes, so every rank creates groups
        # and shards parameters in the same deterministic order.
        if context.is_distributed:
            import torch

            torch.cuda.set_device(context.local_rank)
            self.dialogue_model.ensure_loaded()
            self.video_generator.ensure_loaded()
        remote_async = bool(
            getattr(self.video_generator, "is_remote", False)
        )
        if remote_async and context.is_distributed:
            raise ValueError(
                "remote video service requires a one-process Dialogue Model runtime"
            )
        event_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        sentinel = object()
        cancelled = cancel_event or threading.Event()

        def compute() -> None:
            device_binding_error = None
            if context.is_distributed:
                try:
                    import torch

                    # CUDA's current device is thread-local. The process group
                    # is initialized on the main thread, while model work runs
                    # here, so bind this worker explicitly on every rank.
                    torch.cuda.set_device(context.local_rank)
                    if torch.cuda.current_device() != context.local_rank:
                        raise RuntimeError(
                            "compute thread CUDA device binding failed: "
                            f"expected {context.local_rank}, got "
                            f"{torch.cuda.current_device()}"
                        )
                except Exception as exc:
                    device_binding_error = exc
            sequence = 0
            video_session = None
            vtp_seen = False
            effective_vtp = None
            total_frames = 0
            pending_video = None
            pending_waveforms: list[Mapping[str, Any]] = []
            video_commands = queue.Queue() if remote_async else None
            video_worker_errors: list[Exception] = []
            video_worker_failed = threading.Event()
            video_worker = None

            def emit(event_class, **payload):
                nonlocal sequence
                if context.is_rank0:
                    event_queue.put(
                        event_class(request_id=request_id, seq=sequence, **payload)
                    )
                    sequence += 1

            def on_diffusion_progress(
                step: int,
                total_steps: int,
                chunk: int = 1,
                total_chunks: int | None = None,
            ) -> None:
                emit(
                    DiffusionProgress,
                    step=int(step),
                    total_steps=int(total_steps),
                    chunk=int(chunk),
                    total_chunks=(
                        int(total_chunks)
                        if total_chunks is not None
                        else None
                    ),
                )

            def on_vtp_ready(vtp: str) -> None:
                nonlocal video_session, vtp_seen, effective_vtp
                if vtp_seen:
                    raise RuntimeError("dialogue_model emitted vtp more than once")
                vtp_seen = True
                effective_vtp = apply_vtp_overrides(
                    vtp,
                    **dict(vtp_overrides or {}),
                )
                emit(VTPReady, vtp=effective_vtp)
                video_options = {
                    **dict(video_kwargs or {}),
                    "progress_callback": on_diffusion_progress,
                }
                if remote_async:
                    video_commands.put(
                        (
                            "start",
                            {
                                "vtp": effective_vtp,
                                "ref_image": ref_image,
                                **video_options,
                            },
                        )
                    )
                    return
                video_session = self.video_generator.start_stream(
                    vtp=effective_vtp,
                    ref_image=ref_image,
                    **video_options,
                )

            def on_vtp_delta(delta: str, vtp: str) -> None:
                emit(VTPDelta, delta=delta, vtp=vtp)

            def on_text_delta(delta: str, accumulated: str) -> None:
                emit(TextDelta, delta=delta, text=accumulated)

            def on_waveform(payload: Mapping[str, Any]) -> None:
                audio = dict(payload)
                pending_waveforms.append(audio)
                if context.is_rank0:
                    emit(
                        SpeechAudioChunk,
                        waveform=audio["waveform"],
                        sample_rate=int(audio["sample_rate"]),
                        start_sample=int(audio["start_sample"]),
                        units=int(audio["units"]),
                        final=bool(audio.get("final", False)),
                    )

            def emit_video(payload: Mapping[str, Any]) -> None:
                nonlocal total_frames
                frame_count = int(payload["frames"].shape[0])
                total_frames += frame_count
                if not context.is_rank0:
                    return
                emit(
                    VideoChunk,
                    frames=payload["frames"],
                    waveform=payload["waveform"],
                    audio_sample_rate=int(payload["audio_sample_rate"]),
                    audio_start_sample=int(payload["audio_start_sample"]),
                    start_frame=int(payload["start_frame"]),
                    valid_units=int(payload["valid_units"]),
                    padded_units=int(payload["padded_units"]),
                    final=bool(payload["final"]),
                    timing_seconds=dict(payload.get("timing_seconds", {})),
                )

            def queue_video(payload: Mapping[str, Any]) -> None:
                nonlocal pending_video
                paired = dict(payload)
                pairing_error = None
                if context.is_rank0:
                    try:
                        while (
                            pending_waveforms
                            and not len(pending_waveforms[0]["waveform"])
                        ):
                            pending_waveforms.pop(0)
                        if not pending_waveforms:
                            raise RuntimeError(
                                "video chunk has no synchronized waveform"
                            )
                        audio = pending_waveforms.pop(0)
                        if int(audio["units"]) != int(payload["valid_units"]):
                            raise RuntimeError(
                                "video/audio unit mismatch: "
                                f"{payload['valid_units']} != {audio['units']}"
                            )
                        paired["waveform"] = audio["waveform"]
                        paired["audio_sample_rate"] = int(audio["sample_rate"])
                        paired["audio_start_sample"] = int(
                            audio["start_sample"]
                        )
                    except Exception as exc:
                        pairing_error = exc
                boundary_cancelled, boundary_failed = context.boundary_status(
                    cancelled=cancelled.is_set(),
                    failed=pairing_error is not None,
                )
                if boundary_failed:
                    if pairing_error is not None:
                        raise pairing_error
                    raise RuntimeError(
                        "a distributed rank failed while pairing video audio"
                    )
                if boundary_cancelled:
                    raise InterruptedError("stream cancelled at video boundary")
                if pending_video is not None:
                    previous = dict(pending_video)
                    previous["final"] = False
                    emit_video(previous)
                pending_video = paired

            def run_remote_video() -> None:
                nonlocal video_session
                try:
                    while True:
                        operation, payload = video_commands.get()
                        if operation == "start":
                            video_session = (
                                self.video_generator.start_stream(**payload)
                            )
                        elif operation == "push":
                            if video_session is None:
                                raise RuntimeError(
                                    "speech units arrived before remote video setup"
                                )
                            for chunk in video_session.push_units(payload):
                                queue_video(chunk)
                        elif operation == "finish":
                            if video_session is None:
                                raise RuntimeError(
                                    "remote video session was not started"
                                )
                            finish_options = ({"chunk_callback": queue_video}
                                              if os.environ.get("EX_OMNI_STREAM_FINISH") == "1" else {})
                            for chunk in video_session.finish(**finish_options):
                                queue_video(chunk)
                            return
                        elif operation == "abort":
                            if video_session is not None:
                                video_session.abort()
                            return
                        else:
                            raise RuntimeError(
                                f"unknown remote video operation: {operation}"
                            )
                except Exception as exc:
                    video_worker_errors.append(exc)
                    video_worker_failed.set()

            def raise_video_worker_error() -> None:
                if video_worker_failed.is_set():
                    error = video_worker_errors[0]
                    raise RuntimeError(
                        "asynchronous video worker failed: " + str(error)
                    ) from error

            def on_units(units) -> None:
                if not vtp_seen:
                    raise RuntimeError("speech units arrived before vtp")
                emit(SpeechUnitsChunk, units=units, final=False)
                if remote_async:
                    raise_video_worker_error()
                    video_commands.put(("push", units.copy()))
                    return
                if video_session is None:
                    raise RuntimeError("video session was not initialized")
                for chunk in video_session.push_units(units):
                    queue_video(chunk)

            try:
                if device_binding_error is not None:
                    raise device_binding_error
                if remote_async:
                    video_worker = threading.Thread(
                        target=run_remote_video,
                        name=f"ex-omni-video-{request_id[:8]}",
                        daemon=True,
                    )
                    video_worker.start()
                dialogue_model_options = dict(dialogue_model_kwargs or {})
                dialogue_model_options.setdefault("role_card", role_card)
                dialogue_model_options.setdefault("ref_audio", ref_audio)
                dialogue_model_config = getattr(self.dialogue_model, "config", {})
                if bool(dialogue_model_config.get("waveform_stream", False)):
                    dialogue_model_options.setdefault(
                        "waveform_chunk_callback", on_waveform
                    )
                    dialogue_model_options.setdefault(
                        "waveform_before_units_callback",
                        not remote_async,
                    )
                try:
                    result = self.dialogue_model.generate(
                        text,
                        session_id=session_id,
                        ref_image=ref_image,
                        stream=True,
                        vtp_delta_callback=on_vtp_delta,
                        vtp_ready_callback=on_vtp_ready,
                        text_delta_callback=on_text_delta,
                        units_chunk_callback=on_units,
                        cancel_check=cancelled.is_set,
                        **dialogue_model_options,
                    )
                except BaseException:
                    if remote_async and video_worker.is_alive():
                        video_commands.put(("abort", None))
                        video_worker.join()
                    raise
                if not vtp_seen:
                    raise RuntimeError("streaming dialogue_model did not emit vtp")
                if effective_vtp is not None and effective_vtp != result.vtp:
                    result = replace(result, vtp=effective_vtp)
                if remote_async:
                    raise_video_worker_error()
                    video_commands.put(("finish", None))
                    video_worker.join()
                    raise_video_worker_error()
                else:
                    if video_session is None:
                        raise RuntimeError(
                            "streaming dialogue_model did not initialize video"
                        )
                    for chunk in video_session.finish():
                        queue_video(chunk)
                if pending_video is not None:
                    final_video = dict(pending_video)
                    final_video["final"] = True
                    emit_video(final_video)
                if context.is_rank0:
                    pending_waveforms[:] = [
                        payload
                        for payload in pending_waveforms
                        if len(payload["waveform"])
                    ]
                    if pending_waveforms:
                        raise RuntimeError(
                            "waveform stream has unpaired audio chunks"
                        )
                if cancelled.is_set():
                    emit(Cancelled, reason="cancelled at stream boundary")
                else:
                    emit(Completed, result=result, total_frames=total_frames)
            except InterruptedError as exc:
                emit(Cancelled, reason=str(exc))
            except Exception as exc:
                if os.environ.get("EX_OMNI_DEBUG_STACKS") == "1":
                    import traceback

                    traceback.print_exc()
                emit(
                    StreamError,
                    message=str(exc),
                    error_type=type(exc).__name__,
                )
            finally:
                if context.is_rank0:
                    event_queue.put(sentinel)

        worker = threading.Thread(
            target=compute,
            name=f"ex-omni-stream-{request_id[:8]}",
            daemon=True,
        )
        worker.start()
        try:
            if context.is_rank0:
                while True:
                    event = event_queue.get()
                    if event is sentinel:
                        break
                    yield event
            else:
                worker.join()
        finally:
            if worker.is_alive():
                cancelled.set()
                if context.is_rank0:
                    while worker.is_alive():
                        try:
                            event = event_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if event is sentinel:
                            break
                worker.join()

    def clear_session(self, session_id: str = "default") -> None:
        with self._session_lock:
            self._session_role_inputs.pop(session_id, None)
        if hasattr(self.dialogue_model, "clear_session"):
            self.dialogue_model.clear_session(session_id)

    def close(self) -> None:
        close_video = getattr(self.video_generator, "close", None)
        if callable(close_video):
            close_video()
        close_dialogue = getattr(self.dialogue_model, "close", None)
        if callable(close_dialogue):
            close_dialogue()
