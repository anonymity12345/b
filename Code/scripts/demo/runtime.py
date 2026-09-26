"""Shared runtime for the terminal and Gradio demos."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping
import uuid
import wave

import numpy as np

from ex_omni.deployment import (
    SplitDeploymentPlan,
    VideoServiceManager,
    apply_main_process_env,
    build_split_deployment_plan,
    derive_single_gpu_demo_config,
    derive_split_dialogue_config,
    inject_remote_video_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = {
    "streaming": (
        PROJECT_ROOT / "configs/inference.streaming.local.yaml",
        PROJECT_ROOT / "configs/inference.streaming.example.yaml",
    ),
    "full_sequence": (
        PROJECT_ROOT / "configs/inference.full_sequence.local.yaml",
        PROJECT_ROOT / "configs/inference.full_sequence.example.yaml",
    ),
}
STREAMING_DEPLOYMENT_MODES = {"auto", "split", "single_gpu"}
GPU_RESOURCE_ERROR_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "invalid device ordinal",
    "no cuda gpus are available",
    "no cuda-capable device",
    "cuda-capable device(s) is/are busy or unavailable",
    "cuda error: all cuda-capable devices are busy or unavailable",
    "insufficient free memory",
    "not enough free memory",
)


class StreamingGPUResourceError(RuntimeError):
    """A split Streaming startup failure that permits single-GPU fallback."""


def normalize_streaming_deployment(value: str) -> str:
    normalized = str(value).lower()
    if normalized not in STREAMING_DEPLOYMENT_MODES:
        choices = ", ".join(sorted(STREAMING_DEPLOYMENT_MODES))
        raise ValueError(f"streaming_deployment must be one of: {choices}")
    return normalized


def is_gpu_resource_error(
    error: BaseException,
    *,
    supplemental_text: str = "",
) -> bool:
    if isinstance(error, StreamingGPUResourceError):
        return True
    chain = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    text = "\n".join(
        [f"{type(item).__name__}: {item}" for item in chain]
        + [str(supplemental_text)]
    ).lower()
    return any(marker in text for marker in GPU_RESOURCE_ERROR_MARKERS)


def query_gpu_free_memory() -> dict[int, int]:
    """Return physical GPU free memory in MiB without importing CUDA."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            result[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return result


def streaming_split_gpu_ids(config: str | Path | Mapping[str, Any]) -> tuple[int, ...]:
    plan = build_split_deployment_plan(config, project_root=PROJECT_ROOT)
    topology = plan.topology
    return tuple(
        dict.fromkeys(
            topology.main_visible_gpu_ids + topology.video_visible_gpu_ids
        )
    )


def require_streaming_split_gpus(
    config: str | Path | Mapping[str, Any],
) -> tuple[int, ...]:
    required = streaming_split_gpu_ids(config)
    available = query_gpu_free_memory()
    if available:
        missing = [gpu_id for gpu_id in required if gpu_id not in available]
        if missing:
            raise StreamingGPUResourceError(
                "Streaming split deployment requires GPU IDs "
                f"{list(required)}, but these IDs are unavailable: {missing}"
            )
    return required


def select_single_gpu_id(
    config: str | Path | Mapping[str, Any],
    *,
    requested_gpu: int | None = None,
) -> int:
    if requested_gpu is not None:
        return int(requested_gpu)
    candidates = streaming_split_gpu_ids(config)
    free_memory = query_gpu_free_memory()
    available = {
        gpu_id: free_memory[gpu_id]
        for gpu_id in candidates
        if gpu_id in free_memory
    }
    if available:
        return max(available, key=available.get)
    return int(candidates[0])


def default_config_path(mode: str) -> Path:
    """Return the first available config for a demo mode."""
    normalized = str(mode).lower()
    if normalized not in DEFAULT_CONFIGS:
        raise ValueError("mode must be streaming or full_sequence")
    for candidate in DEFAULT_CONFIGS[normalized]:
        if candidate.is_file():
            return candidate
    return DEFAULT_CONFIGS[normalized][0]


def _validate_demo_config(config: Mapping[str, Any], mode: str) -> None:
    """Reject valid configurations that this demo cannot launch safely."""
    video = config.get("video", {})
    if not isinstance(video, Mapping):
        raise TypeError("video must be a mapping")
    configured_mode = str(video.get("mode", "")).lower()
    if configured_mode != mode:
        profile = str(config.get("inference_profile", "")).lower() or "unspecified"
        raise ValueError(
            f"--mode {mode} is incompatible with inference_profile={profile} "
            f"(video.mode={configured_mode or 'unspecified'}); use the matching "
            f"{mode} configuration"
        )
    if mode == "full_sequence":
        return

    raw_execution = config.get("execution", {})
    raw_video_execution = (
        raw_execution.get("video", {})
        if isinstance(raw_execution, Mapping)
        else {}
    )
    parallelism = str(
        raw_video_execution.get("parallelism", "single")
        if isinstance(raw_video_execution, Mapping)
        else "single"
    ).lower()
    if parallelism not in {"single", "sequence_parallel"}:
        raise ValueError(
            "scripts/demo supports execution.video.parallelism=single or sequence_parallel; "
            f"got {parallelism!r}. Use the distributed launcher for "
            "pipeline or fsdp Video"
        )
    from ex_omni.execution import normalize_execution_config

    execution = normalize_execution_config(config)
    if not execution["enabled"]:
        raise ValueError("Streaming demo requires execution.enabled=true")


def _validate_normalized_plan(
    bound_plan: SplitDeploymentPlan,
    normalized_plan: SplitDeploymentPlan,
) -> None:
    if bound_plan.topology != normalized_plan.topology:
        raise ValueError(
            "The selected profile changes the split deployment topology after "
            "GPU binding. Define Demo GPU groups and video.engine at the root "
            f"execution/video sections (bound={bound_plan.topology}, "
            f"normalized={normalized_plan.topology})"
        )


@dataclass(frozen=True)
class DemoUpdate:
    response_text: str = ""
    vtp: str = ""
    status: str = ""
    speech_path: Path | None = None
    video_path: Path | None = None
    video_chunk_path: Path | None = None
    total_frames: int = 0
    complete: bool = False


class DemoRuntime:
    """Own one loaded pipeline and, for streaming, one split Video service."""

    def __init__(
        self,
        *,
        mode: str = "streaming",
        config: str | Path | None = None,
        output_dir: str | Path = "outputs/demo",
        full_sequence_gpu: int = 0,
        service_timeout_seconds: float = 1200,
        streaming_deployment: str = "split",
        single_gpu_id: int | None = None,
    ) -> None:
        self.mode = str(mode).lower()
        if self.mode not in {"streaming", "full_sequence"}:
            raise ValueError("mode must be streaming or full_sequence")
        requested_deployment = normalize_streaming_deployment(streaming_deployment)
        self.streaming_deployment = (
            "split"
            if requested_deployment == "auto"
            else requested_deployment
        )
        self.config_path = Path(
            config or default_config_path(self.mode)
        ).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.service_timeout_seconds = float(service_timeout_seconds)
        self.single_gpu_id = (
            None if single_gpu_id is None else int(single_gpu_id)
        )
        self._deployment_plan: SplitDeploymentPlan | None = None
        self._video_log_offset = 0
        if self.mode == "streaming" and self.streaming_deployment == "split":
            self._deployment_plan = build_split_deployment_plan(
                self.config_path,
                connect_timeout_seconds=self.service_timeout_seconds,
                project_root=PROJECT_ROOT,
            )
            if self._deployment_plan.topology.video_parallelism not in {
                "single",
                "sequence_parallel",
            }:
                raise ValueError(
                    "scripts/demo supports execution.video.parallelism=single "
                    "or sequence_parallel; got "
                    f"{self._deployment_plan.topology.video_parallelism!r}"
                )
            apply_main_process_env(self._deployment_plan)
        elif self.mode == "streaming":
            self.single_gpu_id = select_single_gpu_id(
                self.config_path,
                requested_gpu=self.single_gpu_id,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.single_gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(full_sequence_gpu))
        self._pipeline = None
        self._video_service: VideoServiceManager | None = None
        self._config: dict[str, Any] | None = None
        self._fps = 25.0
        self._ffmpeg_path: str | None = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise RuntimeError("DemoRuntime has not been started")
        return self._pipeline

    def prepare_idle(self, ref_image: str | Path) -> Path:
        from scripts.demo.idle import prepare_idle

        return prepare_idle(self, ref_image)

    @property
    def generation_defaults(self) -> dict[str, float | int]:
        config = self._config or {}
        dialogue_model = dict(
            config.get("dialogue_model") or config.get("llm") or {}
        )
        video = dict(config.get("video") or {})
        return {
            "temperature": float(dialogue_model.get("temperature", 0.7)),
            "top_p": float(dialogue_model.get("top_p", 0.9)),
            "max_new_tokens": int(dialogue_model.get("max_new_tokens", 512)),
            "diffusion_steps": int(
                video.get("steps", 50 if self.mode == "full_sequence" else 8)
            ),
            "text_cfg": float(
                video.get("cfg", video.get("guidance_scale", 1.0))
            ),
            "audio_cfg": float(
                video.get("audio_cfg", video.get("audio_scale", 1.0))
            ),
        }

    def start(self) -> "DemoRuntime":
        if self._pipeline is not None:
            return self
        from ex_omni.config import load_config, validate_config

        validate_config(
            self.config_path,
            check_weights=False,
            check_world_size=False,
        )
        source = load_config(self.config_path)
        _validate_demo_config(source, self.mode)
        self._config = source
        self._fps = float(dict(source.get("video", {})).get("fps", 25))
        self._ffmpeg_path = dict(source.get("runtime", {})).get("ffmpeg_path")
        try:
            if self.mode == "streaming":
                if self.streaming_deployment == "split":
                    assert self._deployment_plan is not None
                    normalized_plan = build_split_deployment_plan(
                        source,
                        config_path=self.config_path,
                        address=self._deployment_plan.address,
                        authkey=self._deployment_plan.authkey,
                        connect_timeout_seconds=self.service_timeout_seconds,
                        project_root=PROJECT_ROOT,
                    )
                    _validate_normalized_plan(
                        self._deployment_plan, normalized_plan
                    )
                    self._deployment_plan = normalized_plan
                    request_config = derive_split_dialogue_config(
                        source, world_size=1
                    )
                    request_config = inject_remote_video_runtime(
                        request_config, normalized_plan
                    )
                else:
                    request_config = derive_single_gpu_demo_config(source)
                lora_checkpoint = str(
                    dict(request_config.get("video", {})).get(
                        "lora_checkpoint", ""
                    )
                )
                print(
                    f"[streaming] deployment={self.streaming_deployment} "
                    f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
                    flush=True,
                )
                print(
                    "[streaming] Loading video LoRA checkpoint: "
                    f"{lora_checkpoint}",
                    flush=True,
                )
                if self.streaming_deployment == "split":
                    log_path = (
                        self.output_dir / "streaming-video-service.log"
                    )
                    self._video_log_offset = (
                        log_path.stat().st_size if log_path.is_file() else 0
                    )
                    self._video_service = VideoServiceManager(
                        normalized_plan,
                        log_path=log_path,
                    ).start()
            else:
                from ex_omni.execution import derive_request_topology_config

                request_config = derive_request_topology_config(
                    source, world_size=1
                )
            from ex_omni import ExOmni2DPipeline

            self._pipeline = ExOmni2DPipeline(request_config)
            # Materialize every backend on the caller thread. Full-sequence progress
            # reporting delegates only the already-loaded inference work to a
            # worker; TP/FSDP groups must never be created in that worker.
            self._pipeline.dialogue_model.ensure_loaded()
            self._pipeline.video_generator.ensure_loaded()
            if self.mode == "streaming":
                # Do not expose the web UI until every Streaming backend is
                # materialized and the split Video service answers a health
                # check. This keeps the first user request free of lazy-load
                # latency and makes Gradio's startup message a true ready
                # signal.
                print(
                    "[streaming] Video backend ready with LoRA checkpoint: "
                    f"{lora_checkpoint}",
                    flush=True,
                )
        except Exception as exc:
            supplemental = self._current_video_log()
            resource_error = (
                self.mode == "streaming"
                and self.streaming_deployment == "split"
                and is_gpu_resource_error(
                    exc, supplemental_text=supplemental
                )
            )
            self.close()
            if resource_error and not isinstance(
                exc, StreamingGPUResourceError
            ):
                raise StreamingGPUResourceError(str(exc)) from exc
            raise
        return self

    def _current_video_log(self) -> str:
        path = self.output_dir / "streaming-video-service.log"
        if not path.is_file():
            return ""
        try:
            with path.open("rb") as handle:
                handle.seek(self._video_log_offset)
                return handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _video_gpu_ids(self) -> list[int]:
        if self.mode == "streaming" and self.streaming_deployment == "single_gpu":
            return [int(self.single_gpu_id)]
        if self._deployment_plan is None:
            return []
        return list(self._deployment_plan.topology.video_visible_gpu_ids)

    def _output_path(self, session_id: str) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        session = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in session_id
        )[:48]
        return self.output_dir / f"{self.mode}-{session}-{timestamp}-{uuid.uuid4().hex[:6]}.mp4"

    def _encode_progressive_preview(
        self,
        frames: np.ndarray,
        waveform: np.ndarray,
        sample_rate: int,
        output_path: Path,
    ) -> Path:
        from ex_omni.media import ffmpeg_executable

        chunk_dir = output_path.parent / f".{output_path.stem}-preview"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        preview_path = chunk_dir / f"{len(frames):08d}.mp4"
        raw_path = preview_path.with_suffix(".rgb")
        audio_path = preview_path.with_suffix(".wav")
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        height, width = frames.shape[1:3]
        raw_path.write_bytes(frames.tobytes())
        waveform = np.asarray(waveform, dtype=np.float32)
        pcm16 = (
            np.clip(waveform, -1.0, 1.0) * 32767
        ).astype("<i2", copy=False)
        with wave.open(str(audio_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(int(sample_rate))
            writer.writeframes(pcm16.tobytes())
        duration = len(frames) / self._fps
        try:
            subprocess.run(
                [
                    ffmpeg_executable(self._ffmpeg_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    f"{self._fps:g}",
                    "-i",
                    str(raw_path),
                    "-i",
                    str(audio_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    str(max(round(self._fps), 1)),
                    "-c:a",
                    "aac",
                    "-af",
                    f"apad,atrim=duration={duration:.6f}",
                    "-t",
                    f"{duration:.6f}",
                    "-movflags",
                    "+faststart",
                    str(preview_path),
                ],
                check=True,
                timeout=120,
            )
            return preview_path
        finally:
            raw_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)

    @staticmethod
    def _write_speech_audio(
        path: Path,
        chunks: list[np.ndarray],
        sample_rate: int,
    ) -> Path:
        waveform = np.concatenate(chunks) if chunks else np.empty(0)
        pcm16 = (
            np.clip(waveform, -1.0, 1.0) * 32767
        ).astype("<i2", copy=False)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(int(sample_rate))
            writer.writeframes(pcm16.tobytes())
        return path

    def run_turn(
        self,
        text: str,
        *,
        session_id: str,
        ref_image: str | Path | None,
        ref_audio: str | Path | None,
        role_card: str | None = None,
        speech_file: str | Path | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        diffusion_steps: int | None = None,
        text_cfg: float | None = None,
        audio_cfg: float | None = None,
        emotion: str | None = None,
        movement_style: str | None = None,
    ) -> Iterator[DemoUpdate]:
        prompt = str(text or "").strip()
        if not prompt and speech_file is None:
            raise ValueError("text or speech_file is required")
        dialogue_model_kwargs = {}
        if speech_file is not None:
            dialogue_model_kwargs["speech_file"] = speech_file
        if temperature is not None:
            dialogue_model_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            dialogue_model_kwargs["top_p"] = float(top_p)
        if max_new_tokens is not None:
            dialogue_model_kwargs["max_new_tokens"] = int(max_new_tokens)
        vtp_overrides = {
            key: value
            for key, value in {
                "emotion": emotion,
                "movement_style": movement_style,
            }.items()
            if value is not None
        }
        selected_steps = int(
            diffusion_steps
            if diffusion_steps is not None
            else self.generation_defaults["diffusion_steps"]
        )
        if self.mode == "streaming" and selected_steps != 8:
            raise ValueError(
                "Fast (Streaming) uses exactly 8 diffusion steps"
            )
        if self.mode == "full_sequence" and selected_steps != 50:
            raise ValueError("Full-sequence diffusion_steps must be 50")
        video_kwargs = {"steps": selected_steps}
        if text_cfg is not None:
            selected_text_cfg = float(text_cfg)
            if self.mode == "streaming" and selected_text_cfg != 1.0:
                raise ValueError("Streaming text_cfg must be 1.0")
            video_kwargs["cfg"] = selected_text_cfg
        if audio_cfg is not None:
            selected_audio_cfg = float(audio_cfg)
            if self.mode == "streaming" and selected_audio_cfg != 1.0:
                raise ValueError("Streaming audio_cfg must be 1.0")
            video_kwargs["audio_cfg"] = selected_audio_cfg

        def diffusion_status(
            step: int,
            total_steps: int,
            chunk: int = 1,
            total_chunks: int | None = None,
        ) -> str:
            status = f"Video diffusion: {step}/{total_steps}"
            if total_chunks is not None and total_chunks > 1:
                status += f" (clip {chunk}/{total_chunks})"
            elif total_chunks is None:
                status += f" (chunk {chunk})"
            return status

        output_path = self._output_path(session_id)
        if self.mode == "full_sequence":
            yield DemoUpdate(status="Generating results...")
            # Keep this idempotent guard in the request path as well: tests and
            # embedders may inject a pipeline without calling DemoRuntime.start.
            self.pipeline.dialogue_model.ensure_loaded()
            self.pipeline.video_generator.ensure_loaded()
            progress_updates: queue.Queue = queue.Queue()
            outcome: dict[str, Any] = {}

            def report_progress(
                step: int,
                total_steps: int,
                chunk: int = 1,
                total_chunks: int | None = None,
            ) -> None:
                progress_updates.put(
                    diffusion_status(
                        step,
                        total_steps,
                        chunk,
                        total_chunks,
                    )
                )

            def report_dialogue(dialogue_result) -> None:
                progress_updates.put(
                    (
                        "dialogue",
                        dialogue_result.response_text,
                        dialogue_result.vtp,
                        dialogue_result.speech_audio_path,
                    )
                )

            def report_preview(
                frames: np.ndarray,
                fps: float,
                chunk: int,
                total_chunks: int,
                speech_path: str | Path | None,
            ) -> None:
                progress_updates.put(
                    (
                        "preview",
                        frames,
                        float(fps),
                        int(chunk),
                        int(total_chunks),
                        speech_path,
                    )
                )

            def generate_full_sequence() -> None:
                try:
                    outcome["result"] = self.pipeline.chat_to_video(
                        prompt,
                        ref_image=ref_image,
                        ref_audio=ref_audio,
                        role_card=role_card,
                        session_id=session_id,
                        output_path=output_path,
                        dialogue_model_kwargs=(
                            dialogue_model_kwargs or None
                        ),
                        video_kwargs={
                            **video_kwargs,
                            "progress_callback": report_progress,
                            "preview_callback": report_preview,
                            "dialogue_callback": report_dialogue,
                        },
                        vtp_overrides=vtp_overrides,
                    )
                except BaseException as exc:
                    outcome["error"] = exc
                finally:
                    progress_updates.put(None)

            worker = threading.Thread(
                target=generate_full_sequence,
                name="demo-full-sequence-generation",
            )
            worker.start()
            while True:
                progress = progress_updates.get()
                if progress is None:
                    break
                if isinstance(progress, str):
                    yield DemoUpdate(status=progress)
                    continue
                if progress[0] == "dialogue":
                    _, response_text, vtp, speech_path = progress
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        speech_path=speech_path,
                        status="Generating results...",
                    )
                    continue
                _, frames, fps, chunk, total_chunks, speech_path = progress
                if speech_path is not None:
                    import soundfile as sf

                    waveform, sample_rate = sf.read(
                        str(speech_path),
                        dtype="float32",
                        always_2d=True,
                    )
                    waveform = waveform.mean(axis=1)
                else:
                    sample_rate = 24_000
                    waveform = np.zeros(
                        round(len(frames) / fps * sample_rate),
                        dtype=np.float32,
                    )
                preview_path = self._encode_progressive_preview(
                    frames,
                    waveform,
                    int(sample_rate),
                    output_path,
                )
                yield DemoUpdate(
                    status=(
                        f"Preview ready: clip {chunk}/{total_chunks} "
                        f"({len(frames) / fps:.1f}s)"
                    ),
                    video_chunk_path=preview_path,
                    total_frames=len(frames),
                )
            worker.join()
            if "error" in outcome:
                raise outcome["error"]
            result = outcome["result"]
            yield DemoUpdate(
                response_text=result.dialogue_model.response_text,
                vtp=result.dialogue_model.vtp,
                status=f"Completed: {result.video.output_path}",
                speech_path=getattr(
                    result.dialogue_model, "speech_audio_path", None
                ),
                video_path=result.video.output_path,
                total_frames=int(result.video.frames or 0),
                complete=True,
            )
            return

        from ex_omni.distributed.runner import StreamingStreamRecorder
        from ex_omni.schemas import (
            Cancelled,
            Completed,
            DiffusionProgress,
            SpeechAudioChunk,
            StreamError,
            TextDelta,
            VTPDelta,
            VTPReady,
            VideoChunk,
        )

        fragmented_media = os.environ.get("EX_OMNI_FRAGMENTED_MEDIA") == "1"
        recorder_class = StreamingStreamRecorder
        if fragmented_media:
            from scripts.demo.stream_media import FragmentedMediaRecorder
            recorder_class = FragmentedMediaRecorder
        recorder = recorder_class(
            output_path,
            fps=self._fps,
            ffmpeg_path=self._ffmpeg_path,
        )
        speech_output_path = output_path.with_suffix(".speech.wav")
        speech_chunks: list[np.ndarray] = []
        speech_sample_rate: int | None = None
        speech_next_sample = 0
        speech_complete = False
        speech_snapshot_index = 0
        previous_speech_snapshot: Path | None = None
        preview_frames: list[np.ndarray] = []
        preview_waveform = np.empty(0, dtype=np.float32)
        preview_sample_rate: int | None = None
        previous_video_snapshot: Path | None = None
        response_text = ""
        vtp = ""
        try:
            for event in self.pipeline.stream_chat_to_video(
                prompt,
                ref_image=ref_image,
                ref_audio=ref_audio,
                role_card=role_card,
                session_id=session_id,
                dialogue_model_kwargs=dialogue_model_kwargs or None,
                video_kwargs=video_kwargs,
                vtp_overrides=vtp_overrides,
            ):
                if isinstance(event, VTPDelta):
                    vtp = event.vtp
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status="Streaming Visual Thought Plan...",
                    )
                elif isinstance(event, VTPReady):
                    vtp = event.vtp
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status="VTP ready; streaming response...",
                    )
                elif isinstance(event, TextDelta):
                    response_text = event.text
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status="Streaming response; preparing speech and video...",
                    )
                elif isinstance(event, SpeechAudioChunk):
                    if (
                        speech_sample_rate is not None
                        and speech_sample_rate != event.sample_rate
                    ):
                        raise RuntimeError(
                            "speech audio sample rate changed during streaming"
                        )
                    if event.start_sample != speech_next_sample:
                        raise RuntimeError(
                            "speech audio chunks are not contiguous: "
                            f"{event.start_sample} != {speech_next_sample}"
                        )
                    speech_sample_rate = event.sample_rate
                    if len(event.waveform):
                        speech_chunks.append(event.waveform)
                        speech_next_sample += len(event.waveform)
                    if event.final:
                        current_speech_path = speech_output_path
                        speech_complete = True
                    else:
                        speech_snapshot_index += 1
                        current_speech_path = output_path.with_name(
                            f".{output_path.stem}.speech-"
                            f"{speech_snapshot_index:04d}.wav"
                        )
                    if not fragmented_media or event.final:
                        self._write_speech_audio(
                            current_speech_path,
                            speech_chunks,
                            speech_sample_rate,
                        )
                        if previous_speech_snapshot is not None:
                            previous_speech_snapshot.unlink(missing_ok=True)
                        previous_speech_snapshot = (
                            None if event.final else current_speech_path
                        )
                    speech_seconds = (
                        speech_next_sample / speech_sample_rate
                    )
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status=(
                            "Speech ready; preparing first video chunk..."
                            if speech_complete
                            else (
                                f"Streaming speech ({speech_seconds:.1f}s); "
                                "preparing video..."
                            )
                        ),
                        speech_path=(current_speech_path if not fragmented_media or event.final else None),
                    )
                elif isinstance(event, DiffusionProgress):
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status=diffusion_status(
                            event.step,
                            event.total_steps,
                            event.chunk,
                            event.total_chunks,
                        ),
                    )
                elif isinstance(event, VideoChunk):
                    recorder.append(event)
                    if fragmented_media:
                        snapshot_path = recorder.manifest_path
                    else:
                        preview_frames.append(event.frames)
                        if (
                            preview_sample_rate is not None
                            and preview_sample_rate != event.audio_sample_rate
                        ):
                            raise RuntimeError(
                                "preview audio sample rate changed during streaming"
                            )
                        preview_sample_rate = int(event.audio_sample_rate)
                        waveform = np.asarray(event.waveform, dtype=np.float32)
                        audio_start = int(event.audio_start_sample)
                        if audio_start > len(preview_waveform):
                            preview_waveform = np.concatenate(
                                [
                                    preview_waveform,
                                    np.zeros(
                                        audio_start - len(preview_waveform),
                                        dtype=np.float32,
                                    ),
                                ]
                            )
                        elif audio_start < len(preview_waveform):
                            overlap = len(preview_waveform) - audio_start
                            waveform = waveform[min(overlap, len(waveform)) :]
                        if len(waveform):
                            preview_waveform = np.concatenate(
                                [preview_waveform, waveform]
                            )
                        snapshot_path = self._encode_progressive_preview(
                            np.concatenate(preview_frames),
                            preview_waveform,
                            preview_sample_rate,
                            output_path,
                        )
                        if previous_video_snapshot is not None:
                            previous_video_snapshot.unlink(missing_ok=True)
                        previous_video_snapshot = snapshot_path
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status=f"Received {recorder.frames} video frames...",
                        video_chunk_path=snapshot_path,
                        total_frames=recorder.frames,
                    )
                elif isinstance(event, StreamError):
                    raise RuntimeError(f"{event.error_type}: {event.message}")
                elif isinstance(event, Cancelled):
                    raise InterruptedError(event.reason)
                elif isinstance(event, Completed):
                    if event.result is not None:
                        response_text = event.result.response_text
                        vtp = event.result.vtp
                    if (
                        not speech_complete
                        and speech_sample_rate is not None
                    ):
                        self._write_speech_audio(
                            speech_output_path,
                            speech_chunks,
                            speech_sample_rate,
                        )
                        if previous_speech_snapshot is not None:
                            previous_speech_snapshot.unlink(missing_ok=True)
                            previous_speech_snapshot = None
                        speech_complete = True
                    final_path = recorder.finalize()
                    if previous_video_snapshot is not None:
                        previous_video_snapshot.unlink(missing_ok=True)
                        previous_video_snapshot = None
                    yield DemoUpdate(
                        response_text=response_text,
                        vtp=vtp,
                        status=f"Completed: {final_path}",
                        speech_path=(
                            speech_output_path if speech_complete else None
                        ),
                        video_path=(recorder.manifest_path if fragmented_media else final_path),
                        total_frames=event.total_frames,
                        complete=True,
                    )
                    return
            raise RuntimeError("Streaming stream ended without a Completed event")
        except Exception:
            if previous_video_snapshot is not None:
                previous_video_snapshot.unlink(missing_ok=True)
            recorder.abort()
            raise

    def clear_session(self, session_id: str) -> None:
        if self._pipeline is not None:
            self._pipeline.clear_session(session_id)

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass
        service, self._video_service = self._video_service, None
        if service is not None:
            service.close()

    def __enter__(self) -> "DemoRuntime":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SwitchableDemoRuntime:
    """Own one active Demo runtime and switch models without concurrent VRAM use."""

    def __init__(
        self,
        *,
        initial_mode: str = "full_sequence",
        streaming_config: str | Path | None = None,
        full_sequence_config: str | Path | None = None,
        output_dir: str | Path = "outputs/demo",
        full_sequence_gpu: int = 0,
        service_timeout_seconds: float = 1200,
        streaming_deployment: str = "split",
        single_gpu_id: int | None = None,
    ) -> None:
        self.initial_mode = str(initial_mode).lower()
        if self.initial_mode not in {"streaming", "full_sequence"}:
            raise ValueError("initial_mode must be streaming or full_sequence")
        self.streaming_deployment = normalize_streaming_deployment(
            streaming_deployment
        )
        self.config_paths = {
            "streaming": Path(
                streaming_config or default_config_path("streaming")
            ).expanduser().resolve(),
            "full_sequence": Path(
                full_sequence_config or default_config_path("full_sequence")
            ).expanduser().resolve(),
        }
        for config_path in self.config_paths.values():
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.full_sequence_gpu = int(full_sequence_gpu)
        self.service_timeout_seconds = float(service_timeout_seconds)
        self.single_gpu_id = (
            None if single_gpu_id is None else int(single_gpu_id)
        )
        self._runtime: DemoRuntime | None = None
        self._lock = threading.RLock()
        if self.streaming_deployment == "single_gpu":
            self.single_gpu_id = select_single_gpu_id(
                self.config_paths["streaming"],
                requested_gpu=self.single_gpu_id,
            )
            # Keep exactly one physical GPU visible across streaming/full-sequence
            # switches; exposing the original Full-sequence GPU would re-enable
            # automatic multi-GPU Speech/VAE placement.
            self.full_sequence_gpu = self.single_gpu_id
            visible_gpu_ids = [self.single_gpu_id]
        else:
            streaming_plan = build_split_deployment_plan(
                self.config_paths["streaming"],
                connect_timeout_seconds=self.service_timeout_seconds,
                project_root=PROJECT_ROOT,
            )
            visible_gpu_ids = list(
                streaming_plan.topology.main_visible_gpu_ids
            )
        if self.full_sequence_gpu not in visible_gpu_ids:
            visible_gpu_ids.append(self.full_sequence_gpu)
        self._main_visible_gpu_ids = tuple(visible_gpu_ids)

    @property
    def mode(self) -> str:
        if self._runtime is not None:
            return self._runtime.mode
        return self.initial_mode

    @property
    def generation_defaults(self) -> dict[str, float | int]:
        if self._runtime is not None:
            return self._runtime.generation_defaults
        from ex_omni.config import load_config

        config = load_config(self.config_paths[self.mode])
        dialogue_model = dict(
            config.get("dialogue_model") or config.get("llm") or {}
        )
        video = dict(config.get("video") or {})
        return {
            "temperature": float(dialogue_model.get("temperature", 0.7)),
            "top_p": float(dialogue_model.get("top_p", 0.9)),
            "max_new_tokens": int(dialogue_model.get("max_new_tokens", 512)),
            "diffusion_steps": int(
                video.get("steps", 50 if self.mode == "full_sequence" else 8)
            ),
            "text_cfg": float(
                video.get("cfg", video.get("guidance_scale", 1.0))
            ),
            "audio_cfg": float(
                video.get("audio_cfg", video.get("audio_scale", 1.0))
            ),
        }

    def _restore_visible_gpus(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id) for gpu_id in self._main_visible_gpu_ids
        )

    @staticmethod
    def _release_model_memory() -> None:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()

    def start(self) -> "SwitchableDemoRuntime":
        return self.switch_mode(self.initial_mode)

    def switch_mode(self, mode: str) -> "SwitchableDemoRuntime":
        requested = str(mode).lower()
        if requested not in {"streaming", "full_sequence"}:
            raise ValueError("mode must be streaming or full_sequence")
        with self._lock:
            if self._runtime is not None and self._runtime.mode == requested:
                return self
            if requested == "streaming":
                import yaml
                from ex_omni.hub import (
                    is_hf_reference,
                    materialize_hf_reference,
                )

                with self.config_paths[requested].open("r", encoding="utf-8") as handle:
                    candidate = yaml.safe_load(handle) or {}
                checkpoint = candidate.get("video", {}).get("lora_checkpoint")
                if is_hf_reference(checkpoint):
                    # Resolve before releasing Full-sequence so a missing optional
                    # Streaming artifact leaves the active demo untouched.
                    materialize_hf_reference(str(checkpoint))
            previous, self._runtime = self._runtime, None
            if previous is not None:
                previous.close()
                self._release_model_memory()
            runtime = DemoRuntime(
                mode=requested,
                config=self.config_paths[requested],
                output_dir=self.output_dir,
                full_sequence_gpu=self.full_sequence_gpu,
                service_timeout_seconds=self.service_timeout_seconds,
                streaming_deployment=self.streaming_deployment,
                single_gpu_id=self.single_gpu_id,
            )
            # Keep the Streaming main-process GPU set visible for the lifetime of
            # this process, so switching back after CUDA initialization is safe.
            self._restore_visible_gpus()
            try:
                self._runtime = runtime.start()
            except Exception:
                runtime.close()
                self._release_model_memory()
                raise
        return self

    def run_turn(self, *args, **kwargs) -> Iterator[DemoUpdate]:
        with self._lock:
            runtime = self._runtime
            if runtime is None:
                raise RuntimeError("No demo model is loaded")
        # Gradio may resume a streaming generator on a different worker thread.
        # Never hold a thread-owned RLock across yields, or generator teardown
        # can fail with "cannot release un-acquired lock".
        yield from runtime.run_turn(*args, **kwargs)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if self._runtime is not None:
                self._runtime.clear_session(session_id)

    def prepare_idle(self, ref_image: str | Path) -> Path:
        with self._lock:
            if self._runtime is None:
                raise RuntimeError("No demo model is loaded")
            return self._runtime.prepare_idle(ref_image)

    def close(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
            if runtime is not None:
                runtime.close()
                self._release_model_memory()

    def __enter__(self) -> "SwitchableDemoRuntime":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
