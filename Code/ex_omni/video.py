"""Lazy OmniAvatar video generator facade."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from .vtp import vtp_to_video_prompt
from .config import load_config, normalize_video_config
from .schemas import SpeechTokens, VideoRequest, VideoResult


def _section(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    payload = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    return normalize_video_config(payload.get("video", payload))


class OmniAvatarVideoGenerator:
    """Load OmniAvatar weights only when ``generate`` is called."""

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        *,
        backend_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.config = _section(config)
        self._backend_factory = backend_factory
        self._backend: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    def load(self) -> Any:
        if self._backend is None:
            from .hub import materialize_hf_references

            self.config = materialize_hf_references(self.config)
            factory = self._backend_factory
            if factory is None:
                from .model.video_generator.inference import WanInferenceBackend

                factory = WanInferenceBackend
            self._backend = factory(self.config)
        return self._backend

    def ensure_loaded(self) -> Any:
        """Materialize the backend runtime (weights + FSDP sharding).

        FSDP wrapping, auxiliary-model migration and their ``all_gather_object``
        collectives run here. Callers must invoke this on the main thread so
        distributed group creation never happens on a background worker thread.
        """
        backend = self.load()
        loader = getattr(backend, "load", None)
        if callable(loader):
            loader()
        return backend

    def generate(
        self,
        *,
        ref_image: str | Path,
        speech_tokens: SpeechTokens | str | Path,
        output_path: str | Path,
        video_prompt: str | None = None,
        vtp: str | None = None,
        strict_vtp: bool = True,
        mode: str | None = None,
        seed: int | None = None,
        **overrides: Any,
    ) -> VideoResult:
        if (video_prompt is None) == (vtp is None):
            raise ValueError("provide exactly one of video_prompt or vtp")
        condition = video_prompt if video_prompt is not None else vtp_to_video_prompt(
            vtp or "", strict=strict_vtp
        )
        selected_mode = mode or str(self.config.get("mode", "full_sequence"))
        if selected_mode not in {"full_sequence", "streaming"}:
            raise ValueError("mode must be 'full_sequence' or 'streaming'")
        request_config = normalize_video_config(
            self.config,
            mode=selected_mode,
            overrides=overrides,
        )
        if self._backend is None:
            self.config = request_config
        image_path = Path(ref_image)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ex-omni-2d-tokens-") as temp_dir:
            token_input: Path | SpeechTokens
            if isinstance(speech_tokens, SpeechTokens):
                token_input = speech_tokens.save(Path(temp_dir) / "speech_tokens.npy")
            else:
                token_input = Path(speech_tokens)
            request = VideoRequest(
                prompt=condition,
                ref_image=image_path,
                speech_tokens=token_input,
                output_path=output,
                mode=selected_mode,
                seed=seed,
                overrides=overrides,
            )
            backend = self.load()
            raw = backend.generate(request) if hasattr(backend, "generate") else backend(request)

        if isinstance(raw, VideoResult):
            return raw
        if isinstance(raw, (str, Path)):
            return VideoResult(Path(raw), selected_mode, fps=float(self.config.get("fps", 25)))
        if isinstance(raw, Mapping):
            return VideoResult(
                Path(raw.get("output_path", output)),
                str(raw.get("mode", selected_mode)),
                frames=raw.get("frames"),
                fps=float(raw.get("fps", self.config.get("fps", 25))),
                metadata=dict(raw.get("metadata", {})),
            )
        if raw is None and output.exists():
            return VideoResult(output, selected_mode, fps=float(self.config.get("fps", 25)))
        raise TypeError("video backend must return VideoResult, path, mapping, or write output")

    def start_stream(
        self,
        *,
        ref_image: str | Path,
        video_prompt: str | None = None,
        vtp: str | None = None,
        strict_vtp: bool = True,
        seed: int | None = None,
        **overrides: Any,
    ):
        """Create one persistent Streaming session; no temporary token file is used."""
        if (video_prompt is None) == (vtp is None):
            raise ValueError("provide exactly one of video_prompt or vtp")
        if str(self.config.get("mode", "full_sequence")) != "streaming":
            raise ValueError("video streaming requires video.mode=streaming")
        condition = video_prompt if video_prompt is not None else vtp_to_video_prompt(
            vtp or "", strict=strict_vtp
        )
        backend = self.load()
        method = getattr(backend, "start_stream", None)
        if method is None:
            raise RuntimeError("configured video backend does not support streaming")
        return method(
            prompt=condition,
            ref_image=Path(ref_image),
            seed=seed,
            overrides=overrides,
        )
