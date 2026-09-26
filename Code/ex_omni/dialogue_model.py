"""Lazy Ex-Omni dialogue_model facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .config import load_config
from .schemas import DialogueResult, SpeechTokens


def _section(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    payload = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    if "dialogue_model" not in payload:
        return dict(payload)
    from .execution import dialogue_model_config_with_execution

    return dialogue_model_config_with_execution(payload)


class ExOmniDialogueModel:
    """Load Ex-Omni only at the first generation request."""

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
                from ex_omni.dialogue_runtime import ExOmniDialogueRuntime

                factory = ExOmniDialogueRuntime
            self._backend = factory(self.config)
        return self._backend

    def ensure_loaded(self) -> Any:
        """Fully materialize the backend (weights + any tensor-parallel wrapping).

        Distributed dialogue_models create DTensor/NCCL tensor-parallel groups here.
        Callers invoke this on the main thread so every rank creates groups and
        shards decoder parameters in the same order.
        """
        backend = self.load()
        loader = getattr(backend, "load", None)
        if callable(loader):
            loader()
        return backend

    def generate(
        self,
        text: str,
        *,
        session_id: str = "default",
        ref_image: str | Path | None = None,
        ref_audio: str | Path | None = None,
        role_card: str | None = None,
        speech_file: str | Path | None = None,
        **generation: Any,
    ) -> DialogueResult:
        if not text and speech_file is None:
            raise ValueError("text or speech_file is required")
        backend = self.load()
        if hasattr(backend, "generate"):
            raw = backend.generate(
                text=text,
                session_id=session_id,
                ref_image=ref_image,
                ref_audio=ref_audio,
                role_card=role_card,
                speech_file=speech_file,
                **generation,
            )
        else:
            raw = backend.generate_response(
                session_id=session_id,
                user_input=text,
                speech_file=str(speech_file) if speech_file else None,
                **generation,
            )
        if isinstance(raw, DialogueResult):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("dialogue_model backend must return DialogueResult or a mapping")

        token_path_value = raw.get("speech_tokens_path") or raw.get("output_units_path")
        token_path = Path(token_path_value) if token_path_value else None
        audio_path_value = raw.get("speech_audio_path") or raw.get("audio_path")
        audio_path = Path(audio_path_value) if audio_path_value else None
        in_memory = raw.get("speech_tokens")
        if in_memory is None:
            in_memory = raw.get("output_units")
        speech_tokens = None
        if isinstance(in_memory, SpeechTokens):
            speech_tokens = in_memory
        elif in_memory is not None:
            speech_tokens = SpeechTokens(
                np.asarray(in_memory),
                tokenizer_id=self.config.get("speech_tokenizer_id"),
                revision=self.config.get("speech_tokenizer_revision"),
                codebook_source=self.config.get("speech_token_codebook_source", "decoder"),
            )
        return DialogueResult(
            vtp=str(raw.get("vtp") or ""),
            response_text=str(raw.get("response_text") or raw.get("text") or ""),
            speech_tokens=speech_tokens,
            speech_tokens_path=token_path,
            speech_audio_path=audio_path,
            ref_image=Path(raw.get("ref_image") or ref_image)
            if (raw.get("ref_image") or ref_image)
            else None,
            raw=dict(raw),
        )

    def clear_session(self, session_id: str = "default") -> None:
        if self._backend is not None and hasattr(self._backend, "clear_session"):
            self._backend.clear_session(session_id)

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def set_session_history(
        self,
        messages,
        *,
        session_id: str = "default",
        role_card: str | None = None,
        ref_image: str | Path | None = None,
        ref_audio: str | Path | None = None,
    ) -> None:
        backend = self.load()
        setter = getattr(backend, "set_session_history", None)
        if setter is None:
            raise RuntimeError("dialogue_model backend does not support history injection")
        setter(
            session_id,
            messages,
            role_card=role_card,
            ref_image=ref_image,
            ref_audio=ref_audio,
        )
