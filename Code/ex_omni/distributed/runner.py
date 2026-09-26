"""torchrun entry point for rank0 JSONL streaming."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence
import wave

import numpy as np

from ex_omni.media import ffmpeg_executable

from .context import initialize_distributed


def _event_json(
    event,
    *,
    include_waveform: bool = False,
    output_path: str | Path | None = None,
) -> str:
    payload = asdict(event)
    payload["kind"] = event.kind.value
    if isinstance(payload.get("units"), np.ndarray):
        units = payload.pop("units")
        payload["units_shape"] = list(units.shape)
        payload["units"] = units.tolist()
    if "frames" in payload:
        frames = payload.pop("frames")
        payload["frames_shape"] = list(frames.shape)
        # JSONL is a control/example transport; frame bytes stay in Python API.
        payload["frames_dtype"] = str(frames.dtype)
    if "waveform" in payload:
        waveform = np.asarray(payload.pop("waveform"), dtype=np.float32)
        payload["waveform_shape"] = list(waveform.shape)
        if include_waveform:
            pcm16 = (
                np.clip(waveform, -1.0, 1.0) * 32767
            ).astype("<i2", copy=False)
            payload["waveform_pcm16_b64"] = base64.b64encode(
                pcm16.tobytes()
            ).decode("ascii")
    result = payload.get("result")
    if result is not None:
        payload["result"] = {
            "vtp": result["vtp"]
            if isinstance(result, dict)
            else result.vtp,
            "response_text": result["response_text"]
            if isinstance(result, dict)
            else result.response_text,
        }
    if output_path is not None:
        payload["output_path"] = str(output_path)
    return json.dumps(payload, ensure_ascii=False)


def _temporary_media_path(output_path: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.stream-",
        suffix=suffix,
        dir=output_path.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _open_video_writer(path: Path, fps: float):
    import imageio.v2 as imageio

    return imageio.get_writer(path, fps=fps)


class StreamingStreamRecorder:
    """Incrementally persist rank-0 Streaming chunks and atomically mux audio."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        fps: float,
        ffmpeg_path: str | Path | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("stream output fps must be positive")
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.ffmpeg_path = ffmpeg_path
        self._video_path = _temporary_media_path(self.output_path, ".mp4")
        self._audio_path = _temporary_media_path(self.output_path, ".wav")
        self._mux_path = _temporary_media_path(self.output_path, ".mux.mp4")
        self._video_writer = None
        self._audio_writer = None
        self._audio_sample_rate: int | None = None
        self._audio_samples = 0
        self._frames = 0
        self._finished = False

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def finished(self) -> bool:
        return self._finished

    def _ensure_open(self, sample_rate: int) -> None:
        if self._video_writer is None:
            self._video_writer = _open_video_writer(self._video_path, self.fps)
        if self._audio_writer is None:
            self._audio_sample_rate = int(sample_rate)
            self._audio_writer = wave.open(str(self._audio_path), "wb")
            self._audio_writer.setnchannels(1)
            self._audio_writer.setsampwidth(2)
            self._audio_writer.setframerate(self._audio_sample_rate)
        elif int(sample_rate) != self._audio_sample_rate:
            raise ValueError(
                "stream audio sample rate changed: "
                f"{self._audio_sample_rate} -> {sample_rate}"
            )

    def append(self, event) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished stream recording")
        if int(event.start_frame) != self._frames:
            raise ValueError(
                "non-contiguous stream video: "
                f"expected frame {self._frames}, got {event.start_frame}"
            )
        self._ensure_open(int(event.audio_sample_rate))
        for frame in event.frames:
            self._video_writer.append_data(frame)
        self._frames += len(event.frames)

        waveform = np.asarray(event.waveform, dtype=np.float32)
        pcm16 = (
            np.clip(waveform, -1.0, 1.0) * 32767
        ).astype("<i2", copy=False)
        start_sample = int(event.audio_start_sample)
        if start_sample > self._audio_samples:
            silence = np.zeros(start_sample - self._audio_samples, dtype="<i2")
            self._audio_writer.writeframes(silence.tobytes())
            self._audio_samples = start_sample
        elif start_sample < self._audio_samples:
            overlap = self._audio_samples - start_sample
            pcm16 = pcm16[min(overlap, len(pcm16)) :]
        if len(pcm16):
            self._audio_writer.writeframes(pcm16.tobytes())
            self._audio_samples += len(pcm16)

    def _close_writers(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if self._audio_writer is not None:
            self._audio_writer.close()
            self._audio_writer = None

    def _cleanup_temporaries(self) -> None:
        for path in (self._video_path, self._audio_path, self._mux_path):
            path.unlink(missing_ok=True)

    def finalize(self) -> Path:
        if self._finished:
            return self.output_path
        if self._frames <= 0 or self._audio_sample_rate is None:
            raise RuntimeError("Streaming stream completed without video chunks")
        duration = self._frames / self.fps
        target_samples = round(duration * self._audio_sample_rate)
        if self._audio_samples < target_samples:
            silence = np.zeros(target_samples - self._audio_samples, dtype="<i2")
            self._audio_writer.writeframes(silence.tobytes())
            self._audio_samples = target_samples
        self._close_writers()
        try:
            subprocess.run(
                [
                    ffmpeg_executable(self.ffmpeg_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(self._video_path),
                    "-i",
                    str(self._audio_path),
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
                        f"apad=whole_dur={duration:.6f},"
                        f"atrim=duration={duration:.6f}"
                    ),
                    "-t",
                    f"{duration:.6f}",
                    str(self._mux_path),
                ],
                check=True,
                timeout=120,
            )
            os.replace(self._mux_path, self.output_path)
            self._finished = True
            return self.output_path
        finally:
            self._cleanup_temporaries()

    def abort(self) -> None:
        if self._finished:
            return
        self._close_writers()
        self._cleanup_temporaries()
        self._finished = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ex_omni.distributed")
    parser.add_argument("--config", default="configs/inference.streaming.example.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-img", "--image", dest="ref_img", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--role-card")
    parser.add_argument("--session-id", default="default")
    parser.add_argument(
        "--output",
        default="outputs/result.mp4",
        help="Full-sequence or Streaming MP4 output path.",
    )
    parser.add_argument(
        "--log-waveform-base64",
        action="store_true",
        help="Include PCM16 Base64 data in Streaming JSONL video events.",
    )
    return parser


def _run_offline(pipeline, args, context) -> int:
    """Full-sequence offline pipeline: full VTP + speech units, then one video pass."""
    import json

    result = pipeline.chat_to_video(
        args.text,
        ref_image=Path(args.ref_img),
        output_path=Path(args.output),
        ref_audio=Path(args.ref_audio) if args.ref_audio else None,
        role_card=args.role_card,
        session_id=args.session_id,
    )
    if context.is_rank0:
        print(
            json.dumps(
                {
                    "kind": "completed",
                    "vtp": result.dialogue_model.vtp,
                    "response_text": result.dialogue_model.response_text,
                    "output_path": str(result.video.output_path),
                    "mode": result.video.mode,
                    "frames": result.video.frames,
                    "fps": result.video.fps,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def _resolve_streaming(config_path: str) -> bool:
    """Resolve the explicitly validated distributed streaming mode."""
    from ex_omni.config import load_config

    config = load_config(config_path)
    mode = str(config.get("video", {}).get("mode", "full_sequence"))
    if mode != "streaming":
        return False
    return bool(config.get("dialogue_model", {})["stream"])


def _derive_all_shared_config(config, *, world_size: int):
    """Build a native TP/FSDP topology for one 1/4/8-GPU request."""
    from ex_omni.deployment import derive_embedded_worker_config

    return derive_embedded_worker_config(config, world_size=world_size)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ex_omni.config import load_config, validate_config

    split_video = bool(os.environ.get("EX_OMNI_VIDEO_SERVICE_ADDRESS"))
    validate_config(args.config, check_world_size=not split_video)
    do_stream = _resolve_streaming(args.config)
    config = load_config(args.config)
    context = initialize_distributed("nccl")
    if split_video:
        if context.world_size != 1:
            raise ValueError(
                "split video runtime requires a one-process Dialogue Model"
            )
        from ex_omni.deployment import derive_split_dialogue_config

        config = derive_split_dialogue_config(config, world_size=1)
    else:
        config = _derive_all_shared_config(
            config,
            world_size=context.world_size,
        )
    from ex_omni.pipeline import ExOmni2DPipeline
    from ex_omni.schemas import Cancelled, Completed, StreamError, VideoChunk

    try:
        pipeline = ExOmni2DPipeline(config)
        if not do_stream:
            failed = bool(_run_offline(pipeline, args, context))
            if context.is_distributed:
                import torch
                import torch.distributed as dist

                status = torch.tensor(
                    [int(failed)],
                    dtype=torch.int32,
                    device=torch.device("cuda", context.local_rank),
                )
                dist.broadcast(status, src=0, group=context.process_group)
                failed = bool(status.item())
            return int(failed)
        failed = False
        save_error = None
        recorder = None
        if context.is_rank0:
            try:
                recorder = StreamingStreamRecorder(
                    args.output,
                    fps=float(config.get("video", {}).get("fps", 25)),
                    ffmpeg_path=dict(config.get("runtime", {})).get(
                        "ffmpeg_path"
                    ),
                )
            except Exception as exc:
                save_error = f"{type(exc).__name__}: {exc}"
        for event in pipeline.stream_chat_to_video(
            args.text,
            ref_image=Path(args.ref_img),
            ref_audio=Path(args.ref_audio) if args.ref_audio else None,
            role_card=args.role_card,
            session_id=args.session_id,
        ):
            if context.is_rank0:
                saved_output = None
                if recorder is not None and save_error is None:
                    try:
                        if isinstance(event, VideoChunk):
                            recorder.append(event)
                        elif isinstance(event, Completed):
                            saved_output = recorder.finalize()
                        elif isinstance(event, (StreamError, Cancelled)):
                            recorder.abort()
                    except Exception as exc:
                        recorder.abort()
                        save_error = f"{type(exc).__name__}: {exc}"
                print(
                    _event_json(
                        event,
                        include_waveform=args.log_waveform_base64,
                        output_path=saved_output,
                    ),
                    flush=True,
                )
                if isinstance(event, Completed) and save_error is not None:
                    print(
                        json.dumps(
                            {
                                "kind": "error",
                                "error_type": "OutputSaveError",
                                "message": save_error,
                                "output_path": str(
                                    Path(args.output).expanduser().resolve()
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                failed = failed or isinstance(event, StreamError)
        if context.is_rank0 and recorder is not None and not recorder.finished:
            recorder.abort()
        failed = failed or save_error is not None
        if context.is_distributed:
            import torch
            import torch.distributed as dist

            status = torch.tensor(
                [int(failed)],
                dtype=torch.int32,
                device=torch.device("cuda", context.local_rank),
            )
            dist.broadcast(status, src=0, group=context.process_group)
            failed = bool(status.item())
        return int(failed)
    finally:
        if "pipeline" in locals():
            pipeline.close()
        if context.is_distributed:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
