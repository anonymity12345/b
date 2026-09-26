"""Stateful Dialogue Model runtime used by the public inference pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import random
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .constants import (
    VTP_CLOSE_TOKEN,
    VTP_OPEN_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    DEFAULT_OMNI_SYSTEM_MESSAGE,
    IMAGE_TOKEN_INDEX,
    RESPONSE_CLOSE_TOKEN,
    RESPONSE_OPEN_TOKEN,
    S2SV_EN_QA_TEMPLATES,
    S2SV_ZH_QA_TEMPLATES,
    SPEECH_TOKEN_INDEX,
)
from .model.builder import load_model_for_inference
from .response_protocol import (
    StrictAssistantProtocolLogitsProcessor,
    TensorParallelTokenSyncLogitsProcessor,
    make_strict_logits_processor,
    parse_assistant_response,
)
from .utils import detect_language


_SPEECH_SENTENCE_ENDINGS = frozenset(".!?;。！？；")
_SPEECH_TRAILING_CLOSERS = frozenset("\"'”’」』】）)]")


def should_flush_speech_span(
    text: str,
    span_token_count: int,
    min_token_count: int,
) -> bool:
    """Return whether an async Speech span reached a complete sentence."""
    if int(span_token_count) < int(min_token_count):
        return False
    stripped = str(text).rstrip()
    while stripped and stripped[-1] in _SPEECH_TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    return bool(stripped and stripped[-1] in _SPEECH_SENTENCE_ENDINGS)


def leading_system_message_count(history: list[dict[str, str]]) -> int:
    count = 0
    for message in history:
        if message.get("role") != "system":
            break
        count += 1
    return count


def history_contains_image_token(history: list[dict[str, str]]) -> bool:
    return any(
        "<image>" in str(message.get("content", ""))
        for message in history
    )


def build_generation_message(
    history: list[dict[str, str]],
    text: str,
    *,
    qa_prompt: str,
    has_reference_image: bool,
    has_speech_input: bool = False,
) -> str:
    message = f"{qa_prompt}\n{text}"
    if has_speech_input:
        message = f"{DEFAULT_SPEECH_TOKEN}\n{message}"
    if has_reference_image and not history_contains_image_token(history):
        message = f"<image>\n{message}"
    return message


def chat_template_input_ids(rendered: Any) -> Any:
    if isinstance(rendered, Mapping):
        if "input_ids" not in rendered:
            raise ValueError("chat template output is missing input_ids")
        return rendered["input_ids"]
    return rendered


def truncate_conversation_history(
    history: list[dict[str, str]],
    max_turns: int,
) -> None:
    if max_turns <= 0:
        return
    system_count = leading_system_message_count(history)
    dialogue = history[system_count:]
    if len(dialogue) > max_turns * 2:
        recent_start = len(dialogue) - max_turns * 2
        image_anchor = next(
            (
                index
                for index, message in enumerate(dialogue)
                if "<image>" in str(message.get("content", ""))
            ),
            None,
        )
        recent = dialogue[recent_start:]
        if image_anchor is not None and image_anchor < recent_start:
            anchor_end = min(image_anchor + 2, len(dialogue))
            recent = dialogue[image_anchor:anchor_end] + recent
        history[:] = history[:system_count] + recent


class ExOmniDialogueRuntime:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        configured_temp = self.config.get("temp_dir")
        self.temp_dir = (
            Path(configured_temp).expanduser()
            if configured_temp
            else Path(tempfile.gettempdir()) / "ex-omni-2d"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._audio_decoder = None
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._session_images: dict[str, Path] = {}
        self._session_ref_audios: dict[str, Path] = {}
        self._session_role_cards: dict[str, str] = {}
        self._session_role_configured: set[str] = set()
        self._session_ref_audio_inputs: dict[
            str, tuple[Any, Any, Any]
        ] = {}
        self._session_image_inputs: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.RLock()
        self._speech_async_enabled = False
        self._speech_device = None
        self._speech_client = None
        self._llm_engine = str(self.config.get("engine", "native")).lower()
        self._llm_client = None
        self._llm_compiled = False
        self._last_thinker_stats: dict[str, float | int] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is None:
            model_path = self.config.get("model_path")
            if not model_path:
                raise ValueError("dialogue_model.model_path is required")
            self._tokenizer, self._model = load_model_for_inference(
                str(model_path),
                load_bf16=self.config.get("load_bf16"),
                dtype=self.config.get("dtype"),
                device_map=self.config.get("device_map", "auto"),
                attn_implementation=self.config.get("attn_implementation"),
                decode_attn_implementation=self.config.get(
                    "decode_attn_implementation"
                ),
                model_class=self.config.get("model_class"),
                load_method=self.config.get("load_method", "hf_auto_full"),
                trust_remote_code=bool(self.config.get("trust_remote_code", True)),
                pretrained_reference_overrides=self.config.get(
                    "pretrained_reference_overrides"
                ),
            )
            processor_path = None
            if bool(self.config.get("load_vision_processor", True)):
                processor_path = self.config.get(
                    "vision_processor_path"
                ) or getattr(
                    self._model.config,
                    "pretrain_vision_encoder_weights",
                    None,
                )
            if processor_path not in (None, "", "none", "None"):
                from transformers import Qwen3VLProcessor

                processor = Qwen3VLProcessor.from_pretrained(
                    processor_path,
                    trust_remote_code=bool(self.config.get("trust_remote_code", True)),
                )
                self._image_processor = getattr(processor, "image_processor", processor)
            from .distributed.context import current_context

            context = current_context()
            self.distributed_context = context
            self._configure_speech_execution(context)
            if context.is_distributed:
                from .distributed.tp import wrap_dialogue_model_tensor_parallel

                wrap_dialogue_model_tensor_parallel(self, self.config, context)
            self._configure_llm_compile()
            self._configure_llm_engine(context)
        return self._tokenizer, self._model

    def _configure_llm_compile(self) -> None:
        compile_config = self.config.get("llm_compile", {})
        if not compile_config.get("enabled", False):
            return
        if self._llm_engine != "native":
            raise ValueError("compile: true requires the native Thinker engine")
        import torch
        from transformers.cache_utils import Cache

        # Keep multimodal input preparation, sampling and cache ownership eager.
        # The transformer itself runs for both prompt prefill and KV-cache decode.
        transformer = self._model.get_model()
        # DynamicCache mutates a different layer on every attention call. Dynamo
        # otherwise specializes on each intermediate cache layout and exhausts
        # its compile cache before all decoder layers have been compiled.
        if not getattr(Cache.update, "_ex_omni_compile_boundary", False):
            cache_update = torch.compiler.disable(Cache.update)
            cache_update._ex_omni_compile_boundary = True
            Cache.update = cache_update
        # FA3 is an eager boundary, so the shared attention forward is traced
        # once for each successive prefill layer as its DynamicCache grows.
        # The default Dynamo limit of 8 otherwise leaves most layers eager.
        torch._dynamo.config.recompile_limit = int(
            compile_config.get("recompile_limit", 64)
        )
        torch._dynamo.config.suppress_errors = bool(
            compile_config.get("fallback_on_error", True)
        )
        mode = (
            "reduce-overhead" if compile_config.get("cuda_graphs", False)
            else compile_config.get("mode", "default")
        )
        transformer.forward = torch.compile(
            transformer.forward,
            backend=compile_config.get("backend", "inductor"),
            mode=mode,
            fullgraph=bool(compile_config.get("fullgraph", False)),
            dynamic=bool(compile_config.get("dynamic", True)),
        )
        self._llm_compiled = True
        print("[dialogue_model] native Thinker transformer torch.compile enabled", flush=True)

    def _forward_native_thinker(self, **kwargs):
        if not self._llm_compiled:
            return self._model(**kwargs)
        # Qwen3's output_hidden_states capture temporarily wraps every decoder
        # layer on each call. Those fresh closures exhaust Dynamo's recompile
        # limit and leave most layers eager. The final captured hidden state is
        # exactly the backbone's normalized last_hidden_state.
        backbone_kwargs = {
            key: value for key, value in kwargs.items()
            if key not in {"output_hidden_states", "return_dict"}
        }
        output = self._model.get_model()(**backbone_kwargs)
        hidden = output.last_hidden_state
        logits = self._model.lm_head(hidden[:, -1:, :])
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=output.past_key_values,
        )

    def _configure_llm_engine(self, context) -> None:
        if self._llm_engine == "native":
            return
        if self._llm_engine != "vllm_omni":
            raise ValueError(
                "llm.engine must be either 'native' or 'vllm_omni'"
            )
        if context.is_distributed:
            raise ValueError(
                "llm.engine=vllm_omni currently requires a single Dialogue "
                "Model process"
            )
        settings = self.config.get("vllm")
        if not isinstance(settings, Mapping):
            raise ValueError("llm.vllm must be a mapping for engine=vllm_omni")
        from .llm_engines.vllm_omni_client import VllmOmniClient

        client = VllmOmniClient(settings)
        try:
            client.start()
        except Exception:
            client.close()
            if not bool(settings.get("fallback_to_native", True)):
                raise
            print(
                "[llm] vLLM-Omni startup failed; falling back to native "
                f"decoder. See {client.log_path}",
                flush=True,
            )
            self._llm_engine = "native"
            return
        self._llm_client = client

    def _configure_speech_execution(self, context) -> None:
        """Place Speech on an isolated GPU process for async pipeline mode."""
        requested = bool(self.config.get("speech_async", False))
        if not requested:
            return

        import torch
        speech_generator = self._model.get_model().speech_generator
        parameter = next(speech_generator.parameters())
        configured = str(self.config.get("speech_device", "auto"))
        if configured == "auto":
            configured = (
                f"cuda:{torch.cuda.device_count() - 1}"
                if torch.cuda.is_available() and torch.cuda.device_count() > 1
                else str(parameter.device)
            )
        target = torch.device(configured)
        enabled = requested
        reason = None
        if requested and context.is_distributed:
            enabled = False
            reason = "distributed Dialogue Model collectives cannot run from a Speech worker thread"
        elif requested and target == parameter.device:
            enabled = False
            reason = "Speech and Dialogue Model resolve to the same device"
        elif requested and target.type == "cuda":
            index = target.index if target.index is not None else torch.cuda.current_device()
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                enabled = False
                reason = f"Speech device {target} is not visible"

        process_isolation = bool(
            self.config.get("speech_process_isolation", False)
        )
        if enabled and process_isolation:
            from .speech_engines.native_client import NativeSpeechClient

            physical_gpu_id = int(
                os.environ.get(
                    "EX_OMNI_SPEECH_GPU_ID",
                    self.config.get("speech_gpu_id", target.index or 0),
                )
            )
            client = NativeSpeechClient(
                {
                    "python": self.config.get("speech_service_python"),
                    "model_path": self.config["model_path"],
                    "gpu_id": physical_gpu_id,
                    "attention_backend": self.config.get(
                        "speech_attn_implementation",
                        self.config.get("attn_implementation", "sdpa"),
                    ),
                    "startup_timeout_seconds": self.config.get(
                        "speech_service_startup_timeout_seconds", 1200
                    ),
                    "compile": dict(
                        self.config.get("speech_compile", {})
                    ),
                }
            )
            try:
                client.start()
            except Exception:
                client.close()
                raise
            self._speech_client = client
            self._speech_device = target
            self._speech_async_enabled = True
            print(
                "[dialogue_model] asynchronous Speech Generator enabled in "
                f"an isolated process on physical GPU {physical_gpu_id}",
                flush=True,
            )
        elif enabled:
            speech_generator.to(device=target, dtype=parameter.dtype)
            self._speech_device = target
            self._speech_async_enabled = True
            print(
                f"[dialogue_model] asynchronous Speech Generator enabled on {target}",
                flush=True,
            )
        else:
            self._speech_device = parameter.device
            self._speech_async_enabled = False
            if requested:
                print(
                    f"[dialogue_model] asynchronous Speech Generator disabled: {reason}",
                    flush=True,
                )

    def close(self) -> None:
        with self._lock:
            speech_client, self._speech_client = self._speech_client, None
            llm_client, self._llm_client = self._llm_client, None
            for client in (speech_client, llm_client):
                if client is not None:
                    client.close()
            self._model = None
            self._tokenizer = None
            self._image_processor = None
            self._audio_decoder = None
            self._histories.clear()
            self._session_images.clear()
            self._session_ref_audios.clear()
            self._session_role_cards.clear()
            self._session_role_configured.clear()
            self._session_ref_audio_inputs.clear()
            self._session_image_inputs.clear()

    def _load_audio_decoder(self):
        if self._audio_decoder is not None:
            return self._audio_decoder
        model_path = self.config.get("code2wav_model_path")
        if not model_path:
            return None
        import torch

        from .model.speech_generator.code2wav import Qwen3OmniCode2WavDecoder

        configured_device = os.environ.get("EX_OMNI_CODE2WAV_DEVICE")
        if not configured_device:
            configured_device = str(self.config.get("code2wav_device", "cuda"))
        waveform_gpu_id = self.config.get("code2wav_gpu_id")
        if configured_device == "auto" and waveform_gpu_id is not None:
            configured_device = f"cuda:{int(waveform_gpu_id)}"
        if configured_device == "auto":
            configured_device = str(
                self._speech_device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
        dtype = (
            next(self._model.parameters()).dtype
            if self._model is not None
            else torch.bfloat16
        )
        self._audio_decoder = Qwen3OmniCode2WavDecoder(
            str(model_path),
            config_path=self.config.get("code2wav_config_path"),
            device=configured_device,
            dtype=dtype,
        )
        return self._audio_decoder

    def _decode_speech_audio(self, units, session_id: str) -> Path | None:
        decoder = self._load_audio_decoder()
        if decoder is None or units is None:
            return None
        import soundfile as sf

        waveform = decoder.offline_inference(units).detach().cpu().float()
        audio_path = self.temp_dir / f"{session_id}_response.wav"
        sf.write(
            str(audio_path),
            waveform.squeeze(0).numpy(),
            samplerate=int(decoder.sample_rate),
            subtype="PCM_16",
        )
        return audio_path

    def _prepare_ref_image(self, path: Path):
        if self._image_processor is None:
            raise ValueError(
                "ref_image was provided but dialogue_model.vision_processor_path is not "
                "configured and the checkpoint has no vision processor metadata"
            )
        from PIL import Image

        resize_options = {}
        max_pixels = self.config.get("vision_max_pixels")
        if max_pixels is not None:
            min_pixels = int(self.config.get("vision_min_pixels", 65536))
            if not 0 < min_pixels <= int(max_pixels):
                raise ValueError("vision pixel limits must satisfy 0 < min <= max")
            resize_options = {"size": {"shortest_edge": min_pixels,
                                       "longest_edge": int(max_pixels)}}
        with Image.open(path) as source:
            original_size = source.size
            image_inputs = self._image_processor(
                images=source.convert("RGB"), return_tensors="pt", **resize_options,
            )
        grid = image_inputs["image_grid_thw"]
        merge = int(getattr(self._image_processor, "merge_size", 2))
        print(f"[thinker] reference_size={original_size} vision_grid={grid.tolist()} "
              f"visual_tokens={int(grid.prod(dim=-1).sum()) // (merge * merge)} "
              f"max_pixels={max_pixels}", flush=True)
        pixel_values = image_inputs["pixel_values"]
        if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = image_inputs["image_grid_thw"]
        if image_grid_thw.ndim == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        return (
            pixel_values.detach().cpu(),
            image_grid_thw.detach().cpu(),
        )

    def _prepare_ref_audio(
        self,
        path: Path,
        *,
        crop_seed: int | None = None,
    ):
        import torch
        import torchaudio

        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            waveform, sample_rate = torchaudio.load(path)
        except (ImportError, RuntimeError):
            import soundfile as sf

            samples, sample_rate = sf.read(
                path, dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(samples.T.copy())
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != 24_000:
            waveform = torchaudio.functional.resample(
                waveform,
                int(sample_rate),
                24_000,
            )
        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(-1.0, 1.0)
        if waveform.shape[-1] <= 0:
            raise ValueError(f"reference audio is empty: {path}")
        min_seconds = float(
            self.config.get("s2sv_ref_audio_min_seconds", 3.0)
        )
        max_seconds = float(
            self.config.get("s2sv_ref_audio_max_seconds", 10.0)
        )
        if max_seconds > 0:
            min_seconds = max(0.0, min(min_seconds, max_seconds))
            min_samples = int(min_seconds * 24_000)
            max_samples = int(max_seconds * 24_000)
            if waveform.shape[-1] > min_samples:
                upper = min(max_samples, waveform.shape[-1])
                lower = min(min_samples, upper)
                crop_rng = (
                    random.Random(crop_seed)
                    if crop_seed is not None
                    else random
                )
                crop_samples = (
                    crop_rng.randint(lower, upper)
                    if lower < upper
                    else upper
                )
                if waveform.shape[-1] > crop_samples:
                    start = crop_rng.randint(
                        0,
                        waveform.shape[-1] - crop_samples,
                    )
                    waveform = waveform[..., start : start + crop_samples]
        lengths = torch.tensor(
            [waveform.shape[-1]],
            dtype=torch.long,
        )
        has_ref_audio = torch.ones(
            1,
            dtype=torch.bool,
        )
        return waveform.contiguous(), lengths, has_ref_audio

    def _prepare_query_speech(self, path: Path):
        import torch
        import torchaudio

        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            waveform, sample_rate = torchaudio.load(path)
        except (ImportError, RuntimeError):
            import soundfile as sf

            samples, sample_rate = sf.read(
                path, dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(samples.T.copy())
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != 16_000:
            waveform = torchaudio.functional.resample(
                waveform,
                int(sample_rate),
                16_000,
            )
        waveform = torch.nan_to_num(
            waveform,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(-1.0, 1.0)
        if waveform.shape[-1] <= 0:
            raise ValueError(f"query speech is empty: {path}")

        speech_encoder = self._model.get_model().speech_encoder
        if speech_encoder is None:
            raise ValueError(
                "speech_file was provided but the checkpoint has no speech encoder"
            )
        feature_extractor = getattr(speech_encoder, "feature_extractor", None)
        if feature_extractor is None:
            raise ValueError(
                "speech_file was provided but the speech encoder has no "
                "feature extractor"
            )
        features = feature_extractor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=16_000,
            return_tensors="pt",
        )
        speech_parameter = next(speech_encoder.parameters())
        speech = features["input_features"].to(
            device=speech_parameter.device,
            dtype=speech_parameter.dtype,
            non_blocking=True,
        )
        speech_lengths = torch.tensor(
            [speech.shape[-1]],
            dtype=torch.long,
            device=speech_parameter.device,
        )
        return speech, speech_lengths

    def _history(self, session_id: str, role_card: str | None) -> list[dict[str, str]]:
        if session_id not in self._histories:
            history = [{"role": "system", "content": DEFAULT_OMNI_SYSTEM_MESSAGE}]
            if role_card:
                history.append({"role": "system", "content": role_card})
            self._histories[session_id] = history
        return self._histories[session_id]

    @staticmethod
    def _units_array(value: Any) -> np.ndarray:
        if isinstance(value, tuple):
            value = value[1] if len(value) > 1 else value[0]
        if isinstance(value, str):
            value = [
                [int(piece) for piece in frame.replace("|", ",").split(",")]
                for frame in value.strip().split()
            ]
        try:
            import torch

            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
        except ModuleNotFoundError:
            pass
        array = np.asarray(value, dtype=np.int64)
        if array.size == 0:
            return np.empty((0, 16), dtype=np.int64)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 2 and array.shape[0] == 16 and array.shape[1] != 16:
            array = array.T
        if array.ndim != 2 or array.shape[1] != 16:
            raise ValueError(f"generated speech tokens must have shape [T,16], got {array.shape}")
        return array

    def _sample_next_token(self, logits, temperature: float, top_p: float) -> int:
        import torch

        logits = logits.float()
        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())
        logits = logits / max(float(temperature), 1e-5)
        if 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probabilities = torch.softmax(sorted_logits, dim=-1)
            remove = probabilities.cumsum(dim=-1) - probabilities > top_p
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            filtered = torch.full_like(logits, -torch.inf)
            filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            logits = filtered
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    def _prepare_generation_inputs(
        self,
        input_ids,
        speech=None,
        speech_lengths=None,
        pixel_values=None,
        image_grid_thw=None,
    ):
        attention_mask = input_ids.ne(self._tokenizer.pad_token_id)
        if speech is not None or pixel_values is not None:
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
            ) = self._model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths,
                pixel_values,
                image_grid_thw,
            )
        else:
            position_ids = None
            inputs_embeds = self._model.get_model().embed_tokens(input_ids)
        return inputs_embeds, attention_mask, position_ids

    def _predict_units_for_span(
        self,
        hidden_states,
        token_ids,
        *,
        roleplay_embedding=None,
        prefix_units=None,
        max_speech_tokens=750,
        speech_do_sample=True,
        speech_top_k=50,
        speech_top_p=1.0,
        speech_temperature=0.9,
        speech_repetition_penalty=1.05,
        unit_chunk_size=0,
        unit_chunk_callback=None,
        ref_audio_waveform=None,
        ref_audio_waveform_lengths=None,
        has_ref_audio=None,
        generation_seed=None,
        sampling_generator=None,
    ):
        if not token_ids:
            return np.empty((0, 16), dtype=np.int64)
        import torch

        eos_id = self._tokenizer.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
        if eos_id is None:
            eos_id = self._tokenizer.pad_token_id
        complete_text_ids = list(token_ids) + [int(eos_id)]
        config = getattr(self, "config", {})
        use_kv_cache = bool(config.get("speech_use_kv_cache", True))
        residual_use_kv_cache = bool(
            config.get("speech_residual_use_kv_cache", use_kv_cache)
        )
        hidden = torch.cat(hidden_states, dim=1)
        speech_client = getattr(self, "_speech_client", None)
        if speech_client is not None:
            return speech_client.predict(
                hidden,
                complete_text_ids,
                ref_audio_waveform=ref_audio_waveform,
                ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                has_ref_audio=has_ref_audio,
                prefix_units=prefix_units,
                max_speech_tokens=max_speech_tokens,
                do_sample=speech_do_sample,
                top_k=speech_top_k,
                top_p=speech_top_p,
                temperature=speech_temperature,
                repetition_penalty=speech_repetition_penalty,
                use_kv_cache=use_kv_cache,
                residual_use_kv_cache=residual_use_kv_cache,
                unit_chunk_size=unit_chunk_size,
                unit_chunk_callback=unit_chunk_callback,
                seed=generation_seed,
            )

        speech_generator = self._model.get_model().speech_generator
        parameter = next(speech_generator.parameters())
        hidden = hidden.to(device=parameter.device, dtype=parameter.dtype)
        text_ids = torch.tensor(
            [complete_text_ids],
            dtype=torch.long,
            device=parameter.device,
        )
        predicted = speech_generator.predict(
            hidden,
            text_ids,
            embedding=roleplay_embedding,
            prefix_speech_tokens=prefix_units,
            max_speech_tokens=max_speech_tokens,
            do_sample=speech_do_sample,
            top_k=speech_top_k,
            top_p=speech_top_p,
            temperature=speech_temperature,
            repetition_penalty=speech_repetition_penalty,
            return_hidden_states=False,
            use_kv_cache=bool(
                getattr(self, "config", {}).get(
                    "speech_use_kv_cache",
                    True,
                )
            ),
            residual_use_kv_cache=bool(
                getattr(self, "config", {}).get(
                    "speech_residual_use_kv_cache",
                    getattr(self, "config", {}).get(
                        "speech_use_kv_cache",
                        True,
                    ),
                )
            ),
            sampling_generator=sampling_generator,
            unit_chunk_size=unit_chunk_size,
            unit_chunk_callback=(
                (
                    lambda chunk: unit_chunk_callback(
                        self._units_array(chunk)
                    )
                )
                if unit_chunk_callback is not None
                else None
            ),
        )
        return self._units_array(predicted)

    def _generate_stream(
        self,
        input_ids,
        *,
        speech,
        speech_lengths,
        pixel_values,
        image_grid_thw,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        output_speech: bool,
        max_speech_tokens: int,
        speech_do_sample: bool,
        speech_top_k: int,
        speech_top_p: float,
        speech_temperature: float,
        speech_repetition_penalty: float,
        ref_audio_waveform=None,
        ref_audio_waveform_lengths=None,
        has_ref_audio=None,
        vtp_start_callback=None,
        vtp_delta_callback=None,
        vtp_ready_callback=None,
        text_delta_callback=None,
        response_token_callback=None,
        speech_generation_start_callback=None,
        speech_generation_complete_callback=None,
        units_chunk_callback=None,
        waveform_chunk_callback=None,
        codec_timing_callback=None,
        waveform_before_units_callback: bool = False,
        cancel_check=None,
        generation_seed=None,
    ):
        """Manual decode with immediate VTP/text/unit callbacks."""
        import torch

        use_text_kv_cache = bool(self.config.get("text_use_kv_cache", True))
        inputs_embeds, attention_mask, position_ids = self._prepare_generation_inputs(
            input_ids,
            speech,
            speech_lengths,
            pixel_values,
            image_grid_thw,
        )
        thinker_started_at = time.perf_counter()
        first_token_at = None
        last_token_at = None
        capture_thinker = os.environ.get("EX_OMNI_CAPTURE_THINKER")
        if capture_thinker:
            torch.save({"inputs_embeds": inputs_embeds.detach().cpu(),
                        "attention_mask": attention_mask.detach().cpu(),
                        "position_ids": position_ids.detach().cpu() if position_ids is not None else None,
                        "input_ids": input_ids.detach().cpu(),
                        "temperature": temperature, "top_p": top_p,
                        "max_new_tokens": max_new_tokens, "seed": generation_seed}, capture_thinker)
        decode_inputs_embeds = inputs_embeds
        decode_position_ids = position_ids
        outputs = None
        past_key_values = None
        vllm_events = None
        if self._llm_engine == "vllm_omni":
            if inputs_embeds.shape[0] != 1 or not bool(attention_mask.all()):
                raise ValueError("vLLM thinker currently requires one unpadded prompt")
            if position_ids is not None:
                expected_positions = torch.arange(inputs_embeds.shape[1], device=position_ids.device)
                if not torch.equal(position_ids.reshape(-1), expected_positions):
                    raise ValueError("vLLM thinker requires contiguous zero-based prompt positions")
            if self._llm_client is None:
                raise RuntimeError("vLLM-Omni client is not initialized")
            vllm_events = iter(
                self._llm_client.generate(
                    inputs_embeds,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    seed=generation_seed,
                )
            )
        else:
            outputs = self._forward_native_thinker(
                inputs_embeds=decode_inputs_embeds,
                attention_mask=attention_mask,
                position_ids=decode_position_ids,
                use_cache=use_text_kv_cache,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = (
                outputs.past_key_values if use_text_kv_cache else None
            )
        roleplay_embedding = None
        roleplay_error = None
        if output_speech:
            try:
                if self._speech_client is not None:
                    self._speech_client.prepare_roleplay(
                        ref_audio_waveform=ref_audio_waveform,
                        ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                        has_ref_audio=has_ref_audio,
                    )
                else:
                    roleplay_embedding = self._model.build_roleplay_embedding(
                        ref_audio_waveform=ref_audio_waveform,
                        ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                        has_ref_audio=has_ref_audio,
                    )
            except Exception as exc:
                roleplay_error = exc
        distributed_context = getattr(self, "distributed_context", None)
        if distributed_context is not None and distributed_context.is_distributed:
            import torch.distributed as dist

            local_error = (
                f"{type(roleplay_error).__name__}: {roleplay_error}"
                if roleplay_error is not None
                else None
            )
            roleplay_errors = [None] * distributed_context.world_size
            dist.all_gather_object(
                roleplay_errors,
                local_error,
                group=distributed_context.process_group,
            )
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(roleplay_errors)
                if error is not None
            ]
            if failures:
                raise RuntimeError(
                    "role-play speaker embedding failed: " + "; ".join(failures)
                )
        elif roleplay_error is not None:
            raise roleplay_error
        vtp_open_id = self._tokenizer.convert_tokens_to_ids(VTP_OPEN_TOKEN)
        vtp_close_id = self._tokenizer.convert_tokens_to_ids(VTP_CLOSE_TOKEN)
        response_open_id = self._tokenizer.convert_tokens_to_ids(RESPONSE_OPEN_TOKEN)
        response_close_id = self._tokenizer.convert_tokens_to_ids(RESPONSE_CLOSE_TOKEN)
        eos = self._tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        protocol = StrictAssistantProtocolLogitsProcessor(
            self._tokenizer, max_new_tokens=max_new_tokens
        )
        tp_sync = None
        if int(getattr(self, "distributed_mp_size", 1)) > 1:
            tp_sync = TensorParallelTokenSyncLogitsProcessor(
                group=self.distributed_model_group,
                src_rank=int(getattr(self, "distributed_src_rank", 0)),
                temperature=temperature,
                top_p=top_p,
            )

        generated_ids: list[int] = []
        plan_ids: list[int] = []
        response_ids: list[int] = []
        response_hidden = []
        span_ids: list[int] = []
        span_hidden = []
        unit_chunks: list[np.ndarray] = []
        speech_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="ex-omni-speech")
            if self._speech_async_enabled and output_speech
            else None
        )
        speech_futures = []
        speech_text_chunking = str(
            self.config.get("speech_text_chunking", "complete")
        ).lower()
        speech_min_text_chunk_tokens = int(
            self.config.get("speech_min_text_chunk_tokens", 8)
        )
        sentence_chunking = (
            speech_executor is not None
            and speech_text_chunking == "sentence"
        )
        speech_sampling_generator = None
        if (
            output_speech
            and generation_seed is not None
            and self._speech_client is None
        ):
            speech_parameter = next(
                self._model.get_model().speech_generator.parameters()
            )
            speech_sampling_generator = torch.Generator(
                device=speech_parameter.device
            )
            speech_sampling_generator.manual_seed(int(generation_seed))
        state = "before_plan"
        visible_vtp = ""
        visible_text = ""
        waveform_stream = None
        waveform_setup_error = None
        if waveform_chunk_callback is not None and not output_speech:
            raise ValueError(
                "waveform streaming requires output_speech=true"
            )
        if waveform_chunk_callback is not None and (
            distributed_context is None or distributed_context.is_rank0
        ):
            try:
                decoder = self._load_audio_decoder()
                if decoder is None:
                    raise ValueError(
                        "dialogue_model.code2wav_model_path is required for waveform streaming"
                    )
                waveform_stream = decoder.start_stream(
                    initial_chunk_size=int(
                        self.config.get("waveform_initial_chunk_units", 6)
                    ),
                    chunk_size=int(
                        self.config.get("waveform_chunk_units", 6)
                    ),
                    left_context_size=int(
                        self.config.get("waveform_left_context_units", 72)
                    ),
                )
            except Exception as exc:
                waveform_setup_error = exc
        if distributed_context is not None and distributed_context.is_distributed:
            local_error = (
                f"{type(waveform_setup_error).__name__}: {waveform_setup_error}"
                if waveform_setup_error is not None
                else None
            )
            setup_errors = [None] * distributed_context.world_size
            dist.all_gather_object(
                setup_errors,
                local_error,
                group=distributed_context.process_group,
            )
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(setup_errors)
                if error is not None
            ]
            if failures:
                raise RuntimeError(
                    "waveform stream initialization failed: "
                    + "; ".join(failures)
                )
        elif waveform_setup_error is not None:
            raise waveform_setup_error

        def synchronize_boundary() -> None:
            context = getattr(self, "distributed_context", None)
            cancelled = bool(cancel_check and cancel_check())
            if context is not None:
                cancelled, failed = context.boundary_status(cancelled=cancelled)
                if failed:
                    raise RuntimeError("a distributed rank failed at a stream boundary")
            if cancelled:
                raise InterruptedError("stream generation cancelled")

        def deliver_units(units) -> None:
            if not len(units):
                return
            # Remote Video can consume units while PCM is decoded. An in-process
            # Video consumer is synchronous, so decode PCM first to keep the
            # audio/video chunk pair available when Video returns.
            if (
                units_chunk_callback is not None
                and not waveform_before_units_callback
            ):
                units_chunk_callback(units)
            waveform_chunks = []
            waveform_error = None
            codec_started = time.perf_counter()
            if waveform_stream is not None:
                try:
                    waveform_chunks = waveform_stream.push_units(units)
                except Exception as exc:
                    waveform_error = exc
            codec_seconds = time.perf_counter() - codec_started
            if codec_timing_callback is not None and waveform_stream is not None:
                codec_timing_callback(
                    seconds=codec_seconds,
                    units=len(units),
                    output_chunks=len(waveform_chunks),
                )
            context = getattr(self, "distributed_context", None)
            if context is not None and context.is_distributed:
                local_error = (
                    f"{type(waveform_error).__name__}: {waveform_error}"
                    if waveform_error is not None
                    else None
                )
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
                if failures:
                    raise RuntimeError(
                        "waveform chunk decoding failed: "
                        + "; ".join(failures)
                    )
            elif waveform_error is not None:
                raise waveform_error
            if waveform_chunk_callback is not None:
                for chunk in waveform_chunks:
                    waveform_chunk_callback(chunk)
            if (
                units_chunk_callback is not None
                and waveform_before_units_callback
            ):
                units_chunk_callback(units)

        def synthesize_span(hidden_states, token_ids) -> np.ndarray:
            emitted_units = 0

            def on_incremental_units(units) -> None:
                nonlocal emitted_units
                deliver_units(units)
                emitted_units += len(units)

            if speech_generation_start_callback is not None:
                speech_generation_start_callback()
            prefix_units = (
                np.concatenate(unit_chunks, axis=0)
                if unit_chunks
                else np.empty((0, 16), dtype=np.int64)
            )
            with torch.inference_mode():
                units = self._predict_units_for_span(
                    hidden_states,
                    token_ids,
                    roleplay_embedding=roleplay_embedding,
                    prefix_units=prefix_units,
                    max_speech_tokens=max_speech_tokens,
                    speech_do_sample=speech_do_sample,
                    speech_top_k=speech_top_k,
                    speech_top_p=speech_top_p,
                    speech_temperature=speech_temperature,
                    speech_repetition_penalty=speech_repetition_penalty,
                    unit_chunk_size=int(
                        self.config.get("speech_stream_chunk_units", 6)
                    ),
                    unit_chunk_callback=(
                        on_incremental_units
                        if units_chunk_callback is not None
                        or waveform_chunk_callback is not None
                        else None
                    ),
                    ref_audio_waveform=ref_audio_waveform,
                    ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                    has_ref_audio=has_ref_audio,
                    generation_seed=generation_seed,
                    sampling_generator=speech_sampling_generator,
                )
            if speech_generation_complete_callback is not None:
                speech_generation_complete_callback()
            if len(units):
                unit_chunks.append(units)
            if emitted_units < len(units):
                deliver_units(units[emitted_units:])
            return units

        def flush_span() -> None:
            nonlocal span_ids, span_hidden
            if not span_ids:
                return
            if len(span_ids) != len(span_hidden):
                raise RuntimeError("response token/hidden-state alignment mismatch")
            hidden_snapshot = list(response_hidden)
            token_snapshot = list(response_ids)
            span_ids, span_hidden = [], []
            if not output_speech:
                synchronize_boundary()
                return
            if speech_executor is not None:
                speech_futures.append(
                    speech_executor.submit(
                        synthesize_span,
                        hidden_snapshot,
                        token_snapshot,
                    )
                )
            else:
                synthesize_span(hidden_snapshot, token_snapshot)
                synchronize_boundary()

        for _ in range(int(max_new_tokens) + 5):
            token_hidden = None
            if vllm_events is not None:
                try:
                    next_id, token_hidden = next(vllm_events)
                except StopIteration:
                    break
            else:
                processor_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [generated_ids],
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        ),
                    ],
                    dim=1,
                )
                scores = protocol(
                    processor_ids, outputs.logits[:, -1, :].clone()
                )
                if tp_sync is not None:
                    scores = tp_sync(processor_ids, scores)
                next_id = self._sample_next_token(
                    scores, temperature, top_p
                )
            token_at = time.perf_counter()
            if first_token_at is None:
                first_token_at = token_at
            last_token_at = token_at
            generated_ids.append(next_id)
            if (
                next_id == vtp_open_id
                and vtp_start_callback is not None
            ):
                vtp_start_callback()
            if next_id in eos_ids:
                break
            flush_after_response_token = False
            sampled_response_content = (
                state == "response" and next_id != response_close_id
            )
            if sampled_response_content:
                response_ids.append(next_id)
                span_ids.append(next_id)
                if response_token_callback is not None:
                    response_token_callback(next_id)
                updated = self._tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                delta = (
                    updated[len(visible_text) :]
                    if updated.startswith(visible_text)
                    else updated
                )
                visible_text = updated
                if delta and text_delta_callback is not None:
                    text_delta_callback(delta, visible_text)
                if sentence_chunking:
                    flush_after_response_token = should_flush_speech_span(
                        visible_text,
                        len(span_ids),
                        speech_min_text_chunk_tokens,
                    )

            if vllm_events is None:
                next_input = torch.tensor(
                    [[next_id]], dtype=torch.long, device=input_ids.device
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                if use_text_kv_cache:
                    outputs = self._forward_native_thinker(
                        input_ids=next_input,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    next_embed = self._model.get_model().embed_tokens(next_input)
                    decode_inputs_embeds = torch.cat(
                        [decode_inputs_embeds, next_embed], dim=1
                    )
                    if decode_position_ids is not None:
                        decode_position_ids = torch.cat(
                            [
                                decode_position_ids,
                                decode_position_ids[..., -1:] + 1,
                            ],
                            dim=-1,
                        )
                    outputs = self._forward_native_thinker(
                        inputs_embeds=decode_inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=decode_position_ids,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                token_hidden = outputs.hidden_states[-1][:, -1:, :]
            elif token_hidden is None:
                raise RuntimeError(
                    "vLLM-Omni omitted a hidden state for a non-EOS token"
                )

            if next_id == vtp_open_id:
                state = "vtp"
            elif next_id == vtp_close_id:
                state = "after_plan"
                vtp = self._tokenizer.decode(
                    plan_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                synchronize_boundary()
                if vtp_ready_callback is not None:
                    vtp_ready_callback(vtp)
            elif next_id == response_open_id:
                state = "response"
            elif next_id == response_close_id:
                if state != "response":
                    raise RuntimeError("</response> occurred outside response")
                flush_span()
                state = "finished"
            elif state == "vtp":
                plan_ids.append(next_id)
                updated_vtp = self._tokenizer.decode(
                    plan_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                vtp_delta = (
                    updated_vtp[len(visible_vtp) :]
                    if updated_vtp.startswith(visible_vtp)
                    else updated_vtp
                )
                visible_vtp = updated_vtp
                if vtp_delta and vtp_delta_callback is not None:
                    vtp_delta_callback(vtp_delta, visible_vtp)
            elif state == "response":
                response_hidden.append(token_hidden)
                span_hidden.append(token_hidden)
                if flush_after_response_token:
                    flush_span()

        if self._llm_engine == "native" and first_token_at is not None:
            self._last_thinker_stats = {
                "ttft_seconds": first_token_at - thinker_started_at,
                "total_seconds": last_token_at - thinker_started_at,
                "output_tokens": len(generated_ids),
            }
        if vllm_events is not None:
            unexpected_ids = [
                int(token_id) for token_id, _hidden in vllm_events
            ]
            if unexpected_ids:
                raise RuntimeError(
                    "vLLM-Omni emitted tokens after the runtime stopped decoding: "
                    f"{unexpected_ids[:16]}"
                )
        if state == "response":
            flush_span()
        if speech_executor is not None:
            try:
                for future in speech_futures:
                    future.result()
            finally:
                speech_executor.shutdown(wait=True)
        waveform_finish_chunks = []
        waveform_finish_error = None
        if waveform_stream is not None:
            try:
                waveform_finish_chunks = waveform_stream.finish()
            except Exception as exc:
                waveform_finish_error = exc
        context = getattr(self, "distributed_context", None)
        if context is not None and context.is_distributed:
            local_error = (
                f"{type(waveform_finish_error).__name__}: {waveform_finish_error}"
                if waveform_finish_error is not None
                else None
            )
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
            if failures:
                raise RuntimeError(
                    "waveform stream finalization failed: " + "; ".join(failures)
                )
        elif waveform_finish_error is not None:
            raise waveform_finish_error
        if waveform_chunk_callback is not None:
            for chunk in waveform_finish_chunks:
                waveform_chunk_callback(chunk)
        if not response_ids:
            diagnostic_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            raise RuntimeError(
                "stream generated no response content: "
                f"state={state!r}, generated_tokens={len(generated_ids)}, "
                f"raw={diagnostic_text[:1000]!r}"
            )
        output_ids = torch.tensor(
            [generated_ids], dtype=torch.long, device=input_ids.device
        )
        all_units = (
            np.concatenate(unit_chunks, axis=0)
            if unit_chunks
            else np.empty((0, 16), dtype=np.int64)
        )
        return output_ids, all_units

    def generate(
        self,
        *,
        text: str,
        session_id: str = "default",
        role_card: str | None = None,
        ref_image=None,
        ref_audio=None,
        speech_file=None,
        generation_seed: int | None = None,
        **generation: Any,
    ) -> dict[str, Any]:
        import torch
        from transformers import LogitsProcessorList

        tokenizer, model = self.load()
        context = getattr(self, "distributed_context", None)
        if context is not None and context.is_distributed:
            import torch.distributed as dist

            if context.is_rank0 and generation_seed is None:
                generation_seed = random.SystemRandom().randrange(2**63)
            seed_payload = [generation_seed if context.is_rank0 else None]
            dist.broadcast_object_list(
                seed_payload,
                src=0,
                group=context.process_group,
            )
            generation_seed = int(seed_payload[0])
        with self._lock, torch.inference_mode():
            output_speech = bool(generation.get("output_speech", True))
            use_role_references = bool(
                generation.get("use_role_references", True)
            )
            requested_role_card = str(role_card).strip() if role_card else None
            requested_image = (
                Path(ref_image).expanduser().resolve()
                if ref_image and use_role_references
                else None
            )
            requested_ref_audio = (
                Path(ref_audio).expanduser().resolve()
                if ref_audio and use_role_references
                else None
            )
            query_speech_path = (
                Path(speech_file).expanduser().resolve()
                if speech_file
                else None
            )
            role_configured = session_id in self._session_role_configured
            previous_role_card = self._session_role_cards.get(session_id)
            previous_image = self._session_images.get(session_id)
            previous_ref_audio = self._session_ref_audios.get(session_id)
            if not role_configured and use_role_references:
                if requested_image is None:
                    raise ValueError(
                        "ref_image is required on the first request of a session"
                    )
                if output_speech and requested_ref_audio is None:
                    raise ValueError(
                        "ref_audio is required on the first request of a session"
                    )
            if role_configured:
                if (
                    requested_role_card is not None
                    and requested_role_card != previous_role_card
                ):
                    raise ValueError(
                        "a session cannot replace or extend its role configuration; "
                        "clear it first"
                    )
                if requested_image is not None and requested_image != previous_image:
                    raise ValueError(
                        "a session cannot replace or extend its role configuration; "
                        "clear it first"
                    )
                if (
                    requested_ref_audio is not None
                    and requested_ref_audio != previous_ref_audio
                ):
                    raise ValueError(
                        "a session cannot replace or extend its role configuration; "
                        "clear it first"
                    )
            elif any(
                value is not None
                for value in (
                    requested_role_card,
                    requested_image,
                    requested_ref_audio,
                )
            ):
                if (
                    requested_role_card is not None
                    and session_id in self._histories
                    and any(
                        message.get("role") != "system"
                        for message in self._histories[session_id]
                    )
                ):
                    raise ValueError(
                        "role card must be set on the first session request; "
                        "clear the session before adding one"
                    )
                if requested_role_card is not None:
                    self._session_role_cards[session_id] = requested_role_card
                if requested_image is not None:
                    self._session_images[session_id] = requested_image
                if requested_ref_audio is not None:
                    self._session_ref_audios[session_id] = requested_ref_audio
                self._session_role_configured.add(session_id)
            effective_role_card = self._session_role_cards.get(session_id)
            history = self._history(session_id, effective_role_card)
            image_path = self._session_images.get(session_id)
            ref_audio_path = self._session_ref_audios.get(session_id)
            templates = (
                S2SV_ZH_QA_TEMPLATES
                if detect_language(text) == "zh"
                else S2SV_EN_QA_TEMPLATES
            )
            prompt_seed = generation.get("prompt_seed", generation_seed)
            qa_rng = (
                random.Random(prompt_seed)
                if prompt_seed is not None
                else random
            )
            qa_prompt = qa_rng.choice(templates)
            generation_message = build_generation_message(
                history,
                text,
                qa_prompt=qa_prompt,
                has_reference_image=image_path is not None,
                has_speech_input=query_speech_path is not None,
            )
            current = history + [{"role": "user", "content": generation_message}]
            input_ids = tokenizer.apply_chat_template(
                current, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = chat_template_input_ids(input_ids)
            print(f"[thinker] session={session_id} history_messages={len(history)} "
                  f"template_tokens={input_ids.shape[-1]} reference_image={image_path is not None} "
                  f"speech_input={query_speech_path is not None}", flush=True)
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            pixel_values = None
            image_grid_thw = None
            image_error = None
            if image_path is not None:
                try:
                    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
                    if image_token_id is None:
                        raise ValueError(
                            "dialogue_model tokenizer has no registered <image> token"
                        )
                    image_mask = input_ids.eq(int(image_token_id))
                    image_count = int(image_mask.sum().item())
                    if image_count != 1:
                        raise ValueError(
                            "reference image session requires one <image> token, "
                            f"found {image_count}"
                        )
                    input_ids[image_mask] = IMAGE_TOKEN_INDEX
                    cached_image = self._session_image_inputs.get(session_id)
                    if cached_image is None:
                        cached_image = self._prepare_ref_image(image_path)
                        self._session_image_inputs[session_id] = cached_image
                    vision_encoder = getattr(
                        model.get_model(),
                        "vision_encoder",
                        None,
                    )
                    if vision_encoder is None:
                        raise ValueError(
                            "ref_image was provided but the checkpoint has no "
                            "vision encoder"
                        )
                    vision_parameter = next(vision_encoder.parameters())
                    pixel_values = cached_image[0].to(
                        device=vision_parameter.device,
                        dtype=vision_parameter.dtype,
                    )
                    image_grid_thw = cached_image[1].to(
                        device=vision_parameter.device,
                    )
                except Exception as exc:
                    image_error = exc
            if context is not None and context.is_distributed:
                import torch.distributed as dist

                local_image_error = (
                    f"{type(image_error).__name__}: {image_error}"
                    if image_error is not None
                    else None
                )
                image_errors = [None] * context.world_size
                dist.all_gather_object(
                    image_errors,
                    local_image_error,
                    group=context.process_group,
                )
                image_failures = [
                    f"rank {rank}: {error}"
                    for rank, error in enumerate(image_errors)
                    if error is not None
                ]
                if image_failures:
                    raise RuntimeError(
                        "reference image preparation failed: "
                        + "; ".join(image_failures)
                    )
            elif image_error is not None:
                raise image_error
            speech = None
            speech_lengths = None
            speech_error = None
            if query_speech_path is not None:
                try:
                    speech_token_id = tokenizer.convert_tokens_to_ids(
                        DEFAULT_SPEECH_TOKEN
                    )
                    if speech_token_id is None:
                        raise ValueError(
                            "dialogue_model tokenizer has no registered "
                            f"{DEFAULT_SPEECH_TOKEN} token"
                        )
                    speech_mask = input_ids.eq(int(speech_token_id))
                    speech_count = int(speech_mask.sum().item())
                    if speech_count != 1:
                        raise ValueError(
                            "query speech requires one <speech> token, "
                            f"found {speech_count}"
                        )
                    input_ids[speech_mask] = SPEECH_TOKEN_INDEX
                    speech, speech_lengths = self._prepare_query_speech(
                        query_speech_path
                    )
                except Exception as exc:
                    speech_error = exc
            if context is not None and context.is_distributed:
                import torch.distributed as dist

                local_error = (
                    f"{type(speech_error).__name__}: {speech_error}"
                    if speech_error is not None
                    else None
                )
                speech_errors = [None] * context.world_size
                dist.all_gather_object(
                    speech_errors,
                    local_error,
                    group=context.process_group,
                )
                failures = [
                    f"rank {rank}: {error}"
                    for rank, error in enumerate(speech_errors)
                    if error is not None
                ]
                if failures:
                    raise RuntimeError(
                        "query speech preparation failed: " + "; ".join(failures)
                    )
            elif speech_error is not None:
                raise speech_error
            ref_audio_waveform = None
            ref_audio_waveform_lengths = None
            has_ref_audio = None
            ref_audio_error = None
            if ref_audio_path is not None and output_speech:
                try:
                    cached_ref_audio = self._session_ref_audio_inputs.get(
                        session_id
                    )
                    if cached_ref_audio is None:
                        cached_ref_audio = self._prepare_ref_audio(
                            ref_audio_path,
                            crop_seed=generation_seed,
                        )
                        self._session_ref_audio_inputs[session_id] = (
                            cached_ref_audio[0].detach().cpu(),
                            cached_ref_audio[1].detach().cpu(),
                            cached_ref_audio[2].detach().cpu(),
                        )
                        cached_ref_audio = self._session_ref_audio_inputs[
                            session_id
                        ]
                    speech_generator = model.get_model().speech_generator
                    speech_parameter = next(speech_generator.parameters())
                    ref_audio_waveform = cached_ref_audio[0].to(
                        device=speech_parameter.device,
                        dtype=speech_parameter.dtype,
                    )
                    ref_audio_waveform_lengths = cached_ref_audio[1].to(
                        device=speech_parameter.device,
                        dtype=torch.long,
                    )
                    has_ref_audio = cached_ref_audio[2].to(
                        device=speech_parameter.device,
                        dtype=torch.bool,
                    )
                except Exception as exc:
                    ref_audio_error = exc
            if context is not None and context.is_distributed:
                import torch.distributed as dist

                local_error = (
                    f"{type(ref_audio_error).__name__}: {ref_audio_error}"
                    if ref_audio_error is not None
                    else None
                )
                ref_audio_errors = [None] * context.world_size
                dist.all_gather_object(
                    ref_audio_errors,
                    local_error,
                    group=context.process_group,
                )
                failures = [
                    f"rank {rank}: {error}"
                    for rank, error in enumerate(ref_audio_errors)
                    if error is not None
                ]
                if failures:
                    raise RuntimeError(
                        "reference audio preparation failed: " + "; ".join(failures)
                    )
            elif ref_audio_error is not None:
                raise ref_audio_error
            max_new_tokens = int(generation.get("max_new_tokens", self.config.get("max_new_tokens", 512)))
            max_speech_tokens = int(
                generation.get(
                    "max_speech_tokens",
                    self.config.get("max_speech_tokens", 750),
                )
            )
            temperature = float(generation.get("temperature", self.config.get("temperature", 0.7)))
            top_p = float(generation.get("top_p", self.config.get("top_p", 0.9)))
            num_beams = int(generation.get("num_beams", self.config.get("num_beams", 1)))
            speech_do_sample = bool(
                generation.get(
                    "speech_do_sample",
                    self.config.get("speech_do_sample", True),
                )
            )
            speech_top_k = int(
                generation.get(
                    "speech_top_k", self.config.get("speech_top_k", 50)
                )
            )
            speech_top_p = float(
                generation.get(
                    "speech_top_p", self.config.get("speech_top_p", 1.0)
                )
            )
            speech_temperature = float(
                generation.get(
                    "speech_temperature",
                    self.config.get("speech_temperature", 0.9),
                )
            )
            speech_repetition_penalty = float(
                generation.get(
                    "speech_repetition_penalty",
                    self.config.get("speech_repetition_penalty", 1.05),
                )
            )
            vtp_start_callback = generation.get("vtp_start_callback")
            vtp_delta_callback = generation.get("vtp_delta_callback")
            vtp_ready_callback = generation.get("vtp_ready_callback")
            text_delta_callback = generation.get("text_delta_callback")
            response_token_callback = generation.get("response_token_callback")
            speech_generation_start_callback = generation.get(
                "speech_generation_start_callback"
            )
            speech_generation_complete_callback = generation.get(
                "speech_generation_complete_callback"
            )
            units_chunk_callback = generation.get("units_chunk_callback")
            waveform_chunk_callback = generation.get("waveform_chunk_callback")
            codec_timing_callback = generation.get("codec_timing_callback")
            waveform_before_units_callback = bool(
                generation.get("waveform_before_units_callback", False)
            )
            generation_timing_callback = generation.get(
                "generation_timing_callback"
            )
            cancel_check = generation.get("cancel_check")
            stream = bool(
                generation.get("stream", False)
                or vtp_start_callback
                or vtp_delta_callback
                or vtp_ready_callback
                or text_delta_callback
                or response_token_callback
                or speech_generation_start_callback
                or speech_generation_complete_callback
                or units_chunk_callback
                or waveform_chunk_callback
                or codec_timing_callback
            )
            if stream and num_beams != 1:
                raise ValueError("manual streaming decode requires num_beams=1")
            if generation_seed is not None:
                torch.random.default_generator.manual_seed(int(generation_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(int(generation_seed))
            logits_processors = [
                make_strict_logits_processor(tokenizer, max_new_tokens)
            ]
            if int(getattr(self, "distributed_mp_size", 1)) > 1:
                logits_processors.append(
                    TensorParallelTokenSyncLogitsProcessor(
                        group=self.distributed_model_group,
                        src_rank=int(getattr(self, "distributed_src_rank", 0)),
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
            kwargs = {
                "max_new_tokens": max_new_tokens + 5,
                "do_sample": temperature > 0,
                "num_beams": num_beams,
                "use_cache": bool(
                    self.config.get("text_use_kv_cache", True)
                ),
                "pad_token_id": tokenizer.pad_token_id,
                "streaming_unit_gen": False,
                "faster_infer": False,
                "logits_processor": LogitsProcessorList(logits_processors),
                "response_open_token_id": tokenizer.convert_tokens_to_ids(
                    RESPONSE_OPEN_TOKEN
                ),
                "response_close_token_id": tokenizer.convert_tokens_to_ids(
                    RESPONSE_CLOSE_TOKEN
                ),
                "speech": speech,
                "speech_lengths": speech_lengths,
                "ref_audio_waveform": ref_audio_waveform,
                "ref_audio_waveform_lengths": ref_audio_waveform_lengths,
                "has_ref_audio": has_ref_audio,
                "output_speech": output_speech,
                "max_speech_tokens": max_speech_tokens,
                "speech_do_sample": speech_do_sample,
                "speech_top_k": speech_top_k,
                "speech_top_p": speech_top_p,
                "speech_temperature": speech_temperature,
                "speech_repetition_penalty": speech_repetition_penalty,
            }
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=top_p)
            if generation_timing_callback is not None:
                kwargs["generation_timing_callback"] = (
                    generation_timing_callback
                )
            if stream:
                output_ids, units = self._generate_stream(
                    input_ids,
                    speech=speech,
                    speech_lengths=speech_lengths,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    output_speech=output_speech,
                    max_speech_tokens=max_speech_tokens,
                    speech_do_sample=speech_do_sample,
                    speech_top_k=speech_top_k,
                    speech_top_p=speech_top_p,
                    speech_temperature=speech_temperature,
                    speech_repetition_penalty=speech_repetition_penalty,
                    ref_audio_waveform=ref_audio_waveform,
                    ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                    has_ref_audio=has_ref_audio,
                    vtp_start_callback=vtp_start_callback,
                    vtp_delta_callback=vtp_delta_callback,
                    vtp_ready_callback=vtp_ready_callback,
                    text_delta_callback=text_delta_callback,
                    response_token_callback=response_token_callback,
                    speech_generation_start_callback=(
                        speech_generation_start_callback
                    ),
                    speech_generation_complete_callback=(
                        speech_generation_complete_callback
                    ),
                    units_chunk_callback=units_chunk_callback,
                    waveform_chunk_callback=waveform_chunk_callback,
                    codec_timing_callback=codec_timing_callback,
                    waveform_before_units_callback=(
                        waveform_before_units_callback
                    ),
                    cancel_check=cancel_check,
                    generation_seed=generation_seed,
                )
            else:
                output = model.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )
                units = None
                if isinstance(output, tuple):
                    output_ids, units = output
                else:
                    output_ids = output.sequences if hasattr(output, "sequences") else output
            decode_ids = output_ids
            prompt_length = input_ids.shape[1]
            if (
                output_ids.ndim == 2
                and output_ids.shape[1] > prompt_length
                and torch.equal(output_ids[:, :prompt_length], input_ids)
            ):
                decode_ids = output_ids[:, prompt_length:]
            raw_text = tokenizer.batch_decode(
                decode_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            vtp, response_text = parse_assistant_response(raw_text)
            canonical = (
                f"<avatar_plan>{vtp}</avatar_plan>\n"
                f"<response>{response_text}</response>"
            )
            history.extend(
                [
                    {
                        "role": "user",
                        "content": build_generation_message(
                            history,
                            text,
                            qa_prompt=qa_prompt,
                            has_reference_image=image_path is not None,
                            has_speech_input=False,
                        ),
                    },
                    {"role": "assistant", "content": canonical},
                ]
            )
            max_turns = int(self.config.get("max_history_length", 10))
            truncate_conversation_history(history, max_turns)

        units_path = None
        units_array = None
        speech_audio_path = None
        if units is not None:
            units_array = self._units_array(units)
            context = getattr(self, "distributed_context", None)
            if context is None or context.is_rank0:
                units_path = self.temp_dir / f"{session_id}_speech_tokens.npy"
                np.save(units_path, units_array, allow_pickle=False)
        audio_decode_error = None
        context = getattr(self, "distributed_context", None)
        if (
            units_array is not None
            and not stream
            and (context is None or context.is_rank0)
        ):
            try:
                speech_audio_path = self._decode_speech_audio(
                    units_array,
                    session_id,
                )
            except Exception as exc:
                audio_decode_error = f"{type(exc).__name__}: {exc}"
        if context is not None and context.is_distributed:
            import torch.distributed as dist

            status = [audio_decode_error]
            dist.broadcast_object_list(
                status,
                src=0,
                group=context.process_group,
            )
            audio_decode_error = status[0]
        if audio_decode_error is not None:
            raise RuntimeError(
                "generated speech waveform decoding failed: "
                + audio_decode_error
            )
        return {
            "vtp": vtp or "",
            "text": response_text,
            "raw_text": raw_text,
            "speech_tokens": units_array,
            "speech_tokens_path": str(units_path) if units_path else None,
            "speech_audio_path": (
                str(speech_audio_path) if speech_audio_path else None
            ),
            "ref_image": str(image_path) if image_path else None,
            "ref_audio": str(ref_audio_path) if ref_audio_path else None,
            "role_card": effective_role_card,
            "generation_seed": generation_seed,
            "history": [dict(message) for message in history],
        }

    def generate_response(self, session_id: str, user_input: str, **kwargs: Any):
        return self.generate(text=user_input, session_id=session_id, **kwargs)

    def set_session_history(
        self,
        session_id: str,
        messages,
        *,
        role_card: str | None = None,
        ref_image=None,
        ref_audio=None,
    ) -> None:
        history = [{"role": "system", "content": DEFAULT_OMNI_SYSTEM_MESSAGE}]
        if role_card:
            history.append({"role": "system", "content": str(role_card)})
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"user", "assistant"}:
                raise ValueError(
                    "benchmark history messages must be user or assistant"
                )
            history.append({"role": role, "content": content})
        with self._lock:
            self.clear_session(session_id)
            self._histories[session_id] = history
            if role_card:
                self._session_role_cards[session_id] = str(role_card)
            if ref_image:
                self._session_images[session_id] = (
                    Path(ref_image).expanduser().resolve()
                )
            if ref_audio:
                self._session_ref_audios[session_id] = (
                    Path(ref_audio).expanduser().resolve()
                )
            if role_card or ref_image or ref_audio:
                self._session_role_configured.add(session_id)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._histories.pop(session_id, None)
            self._session_images.pop(session_id, None)
            self._session_ref_audios.pop(session_id, None)
            self._session_role_cards.pop(session_id, None)
            self._session_role_configured.discard(session_id)
            self._session_ref_audio_inputs.pop(session_id, None)
            self._session_image_inputs.pop(session_id, None)
