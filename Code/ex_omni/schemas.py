"""Stable, dependency-light public schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class StreamEventKind(str, Enum):
    VTP_DELTA = "vtp_delta"
    VTP_READY = "vtp_ready"
    TEXT_DELTA = "text_delta"
    SPEECH_UNITS_CHUNK = "speech_units_chunk"
    SPEECH_AUDIO_CHUNK = "speech_audio_chunk"
    DIFFUSION_PROGRESS = "diffusion_progress"
    VIDEO_CHUNK = "video_chunk"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class StreamEvent:
    request_id: str
    seq: int
    kind: StreamEventKind = field(init=False)


@dataclass(frozen=True)
class VTPDelta(StreamEvent):
    delta: str
    vtp: str
    kind: StreamEventKind = field(default=StreamEventKind.VTP_DELTA, init=False)


@dataclass(frozen=True)
class VTPReady(StreamEvent):
    vtp: str
    kind: StreamEventKind = field(default=StreamEventKind.VTP_READY, init=False)


@dataclass(frozen=True)
class TextDelta(StreamEvent):
    delta: str
    text: str
    kind: StreamEventKind = field(default=StreamEventKind.TEXT_DELTA, init=False)


@dataclass(frozen=True)
class SpeechUnitsChunk(StreamEvent):
    units: np.ndarray
    final: bool = False
    kind: StreamEventKind = field(default=StreamEventKind.SPEECH_UNITS_CHUNK, init=False)

    def __post_init__(self) -> None:
        units = np.asarray(self.units)
        if units.ndim != 2 or units.shape[1] != 16:
            raise ValueError(f"speech units chunk must have shape [T,16], got {units.shape}")
        if not np.issubdtype(units.dtype, np.integer):
            raise TypeError("speech units chunk must contain integers")
        object.__setattr__(self, "units", np.ascontiguousarray(units, dtype=np.int64))


@dataclass(frozen=True)
class SpeechAudioChunk(StreamEvent):
    waveform: np.ndarray
    sample_rate: int
    start_sample: int
    units: int
    final: bool = False
    kind: StreamEventKind = field(
        default=StreamEventKind.SPEECH_AUDIO_CHUNK,
        init=False,
    )

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform)
        if waveform.ndim != 1:
            raise ValueError("speech audio chunk waveform must be one-dimensional")
        if not len(waveform) and not self.final:
            raise ValueError(
                "non-final speech audio chunk must contain a waveform"
            )
        if not np.issubdtype(waveform.dtype, np.floating):
            raise TypeError("speech audio chunk must contain floating-point PCM")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("speech audio chunk must contain finite PCM samples")
        if int(self.sample_rate) <= 0:
            raise ValueError("speech audio sample_rate must be positive")
        if int(self.start_sample) < 0:
            raise ValueError("speech audio start_sample must be non-negative")
        if int(self.units) < 0 or (int(self.units) == 0 and not self.final):
            raise ValueError(
                "speech audio units must be positive unless the chunk is final"
            )
        object.__setattr__(
            self,
            "waveform",
            np.ascontiguousarray(waveform, dtype=np.float32),
        )


@dataclass(frozen=True)
class DiffusionProgress(StreamEvent):
    step: int
    total_steps: int
    chunk: int = 1
    total_chunks: int | None = None
    kind: StreamEventKind = field(
        default=StreamEventKind.DIFFUSION_PROGRESS,
        init=False,
    )

    def __post_init__(self) -> None:
        if not 1 <= int(self.step) <= int(self.total_steps):
            raise ValueError("diffusion step must be in [1, total_steps]")
        if int(self.chunk) <= 0:
            raise ValueError("diffusion chunk must be positive")
        if self.total_chunks is not None and not (
            int(self.chunk) <= int(self.total_chunks)
        ):
            raise ValueError("diffusion chunk must not exceed total_chunks")


@dataclass(frozen=True)
class VideoChunk(StreamEvent):
    frames: np.ndarray
    waveform: np.ndarray
    audio_sample_rate: int
    audio_start_sample: int
    start_frame: int
    valid_units: int
    padded_units: int
    final: bool
    timing_seconds: dict[str, float] = field(default_factory=dict)
    kind: StreamEventKind = field(default=StreamEventKind.VIDEO_CHUNK, init=False)

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"video frames must have shape [F,H,W,3], got {frames.shape}")
        if frames.dtype != np.uint8:
            raise TypeError(f"video frames must be uint8, got {frames.dtype}")
        waveform = np.asarray(self.waveform)
        if waveform.ndim != 1:
            raise ValueError(
                f"video audio must have shape [samples], got {waveform.shape}"
            )
        if not np.issubdtype(waveform.dtype, np.floating):
            raise TypeError("video audio must contain floating-point PCM")
        if not len(waveform) or not np.all(np.isfinite(waveform)):
            raise ValueError("video audio must contain finite PCM samples")
        if int(self.audio_sample_rate) <= 0:
            raise ValueError("video audio_sample_rate must be positive")
        if int(self.audio_start_sample) < 0:
            raise ValueError("video audio_start_sample must be non-negative")
        object.__setattr__(self, "frames", np.ascontiguousarray(frames))
        object.__setattr__(
            self,
            "waveform",
            np.ascontiguousarray(waveform, dtype=np.float32),
        )


@dataclass(frozen=True)
class Completed(StreamEvent):
    result: DialogueResult | None = None
    total_frames: int = 0
    kind: StreamEventKind = field(default=StreamEventKind.COMPLETED, init=False)


@dataclass(frozen=True)
class StreamError(StreamEvent):
    message: str
    error_type: str = "RuntimeError"
    kind: StreamEventKind = field(default=StreamEventKind.ERROR, init=False)


@dataclass(frozen=True)
class Cancelled(StreamEvent):
    reason: str = "cancelled"
    kind: StreamEventKind = field(default=StreamEventKind.CANCELLED, init=False)


@dataclass(frozen=True)
class SpeechTokens:
    """Integer Qwen speech tokens in canonical ``[T, 16]`` layout."""

    values: np.ndarray
    rate_hz: float = 12.5
    tokenizer_id: str | None = None
    revision: str | None = None
    codebook_source: str = "decoder"

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(f"speech tokens must be 2D, got {values.shape}")
        if values.shape[1] != 16 and values.shape[0] == 16:
            values = values.T
        if values.shape[1] != 16:
            raise ValueError(f"speech tokens must have 16 codebooks, got {values.shape}")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"speech tokens must be integers, got {values.dtype}")
        if float(self.rate_hz) != 12.5:
            raise ValueError(f"speech token rate must be 12.5 Hz, got {self.rate_hz}")
        if self.codebook_source not in {"decoder", "encoder"}:
            raise ValueError("codebook_source must be 'decoder' or 'encoder'")
        object.__setattr__(self, "values", np.ascontiguousarray(values, dtype=np.int64))

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @classmethod
    def from_npy(
        cls,
        path: str | Path,
        *,
        tokenizer_id: str | None = None,
        revision: str | None = None,
        codebook_source: str = "decoder",
    ) -> "SpeechTokens":
        return cls(
            np.load(Path(path), allow_pickle=False),
            tokenizer_id=tokenizer_id,
            revision=revision,
            codebook_source=codebook_source,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, self.values, allow_pickle=False)
        return target


@dataclass(frozen=True)
class DialogueResult:
    vtp: str
    response_text: str
    speech_tokens: SpeechTokens | None = None
    speech_tokens_path: Path | None = None
    speech_audio_path: Path | None = None
    ref_image: Path | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def require_speech_tokens(self) -> SpeechTokens | Path:
        if self.speech_tokens is not None:
            return self.speech_tokens
        if self.speech_tokens_path is not None:
            return self.speech_tokens_path
        raise ValueError("dialogue_model did not return speech tokens")


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    ref_image: Path
    speech_tokens: SpeechTokens | Path
    output_path: Path
    mode: str = "full_sequence"
    seed: int | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoResult:
    output_path: Path
    mode: str
    frames: int | None = None
    fps: float = 25.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndToEndResult:
    dialogue_model: DialogueResult
    video: VideoResult
