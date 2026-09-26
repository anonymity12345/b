"""Configuration loading and weight-free validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping

from ex_omni.model.attention import (
    resolve_attn_implementation,
    resolve_decode_attn_implementation,
)
from ex_omni.hub import is_hf_reference, parse_hf_reference


PATH_KEYS = {
    "model_path",
    "code2wav_model_path",
    "code2wav_config_path",
    "vision_processor_path",
    "text_encoder_path",
    "dit_path",
    "vae_path",
    "speech_tokenizer_path",
    "base_checkpoint",
    "lora_checkpoint",
    "temp_dir",
    "output_dir",
}
WEIGHT_PATH_KEYS = PATH_KEYS - {"temp_dir", "output_dir"}
@dataclass(frozen=True)
class ValidationReport:
    config_path: Path
    mode: str
    unresolved: tuple[str, ...]
    checked_weights: bool


def _expand_env(value: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return os.path.expanduser(pattern.sub(replace, value))


def _is_placeholder(value: str) -> bool:
    return bool(re.search(r"\$\{[^}]+}", value)) or value.startswith("<")


def _resolve_tree(value: Any, *, key: str | None, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _resolve_tree(item, key=item_key, base_dir=base_dir)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_tree(item, key=key, base_dir=base_dir) for item in value]
    if not isinstance(value, str):
        return value
    expanded = _expand_env(value)
    is_path = key in PATH_KEYS or bool(key and (key.endswith("_path") or key.endswith("_dir")))
    if is_path and is_hf_reference(expanded):
        parse_hf_reference(expanded)
        return expanded
    if is_path and expanded not in {"", "none", "None"} and not _is_placeholder(expanded):
        path = Path(expanded)
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())
    return expanded


def _merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_mapping(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_component_compile(
    raw: Any,
    *,
    component: str,
) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"compile.{component} must be a mapping")
    result = dict(raw)
    result.setdefault("enabled", False)
    result.setdefault("cuda_graphs", False)
    result.setdefault("backend", "inductor")
    result.setdefault("mode", "default")
    result.setdefault("fullgraph", False)
    result.setdefault("dynamic", True)
    result.setdefault("fallback_on_error", True)
    result.setdefault("recompile_limit", 64 if component == "llm" else 16)
    if component == "speech":
        result.setdefault("residual_cuda_graphs", False)
        if type(result["residual_cuda_graphs"]) is not bool:
            raise ValueError("compile.speech.residual_cuda_graphs must be a boolean")
    for key in (
        "enabled",
        "cuda_graphs",
        "fullgraph",
        "dynamic",
        "fallback_on_error",
    ):
        if type(result[key]) is not bool:
            raise ValueError(f"compile.{component}.{key} must be a boolean")
    if result["cuda_graphs"] and not result["enabled"]:
        raise ValueError(
            f"compile.{component}.cuda_graphs=true requires "
            f"compile.{component}.enabled=true"
        )
    if result["cuda_graphs"] and result["dynamic"]:
        raise ValueError(
            f"compile.{component}.cuda_graphs=true requires "
            f"compile.{component}.dynamic=false"
        )
    if (
        type(result["recompile_limit"]) is not int
        or not 1 <= result["recompile_limit"] <= 64
    ):
        raise ValueError(
            f"compile.{component}.recompile_limit must be an integer in [1, 64]"
        )
    return result


def _normalize_dialogue_sections(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten the llm/speech schema for weight loading."""
    if "planner" in result or "dialogue_model" in result:
        raise ValueError("use the release llm and speech sections")
    llm = result.pop("llm", None)
    speech = result.pop("speech", None)
    if not isinstance(llm, Mapping) or not isinstance(speech, Mapping):
        raise ValueError("configuration requires llm and speech mappings")

    flattened: dict[str, Any] = {}
    aliases = {
        "llm": {"use_kv_cache": "text_use_kv_cache"},
        "speech": {
            "async": "speech_async",
            "device": "speech_device",
            "use_kv_cache": "speech_use_kv_cache",
            "residual_use_kv_cache": "speech_residual_use_kv_cache",
            "max_tokens": "max_speech_tokens",
            "do_sample": "speech_do_sample",
            "top_k": "speech_top_k",
            "top_p": "speech_top_p",
            "temperature": "speech_temperature",
            "repetition_penalty": "speech_repetition_penalty",
            "stream_chunk_units": "speech_stream_chunk_units",
            "text_chunking": "speech_text_chunking",
            "min_text_chunk_tokens": "speech_min_text_chunk_tokens",
            "ref_audio_min_seconds": "s2sv_ref_audio_min_seconds",
            "ref_audio_max_seconds": "s2sv_ref_audio_max_seconds",
            "tokenizer_id": "speech_tokenizer_id",
            "tokenizer_revision": "speech_tokenizer_revision",
            "token_codebook_source": "speech_token_codebook_source",
        },
    }
    for name, section in (("llm", llm), ("speech", speech)):
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} must be a mapping")
        for key, value in section.items():
            internal_key = aliases[name].get(str(key), str(key))
            if internal_key in {"compile", "llm_compile", "speech_compile"}:
                raise ValueError("use the root compile switch")
            if internal_key in flattened:
                raise ValueError(
                    f"llm/speech sections both configure {internal_key}"
                )
            flattened[internal_key] = value
    return flattened


def normalize_runtime_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one public inference profile and one attention backend globally."""
    result = dict(config)
    raw_video = result.get("video", {})
    inferred_profile = (
        "quality"
        if isinstance(raw_video, Mapping)
        and str(raw_video.get("mode", "streaming")).lower() == "full_sequence"
        else "streaming"
    )
    profile = str(
        result.get("inference_profile", inferred_profile)
    ).strip().lower()
    if profile not in {"quality", "streaming", "offline_performance"}:
        raise ValueError(
            "inference_profile must be quality, streaming, or offline_performance"
        )

    raw_profiles = result.pop("profiles", {})
    if raw_profiles is None:
        raw_profiles = {}
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("profiles must be a mapping")
    selected_override = raw_profiles.get(profile, {})
    if selected_override is None:
        selected_override = {}
    if not isinstance(selected_override, Mapping):
        raise ValueError(f"profiles.{profile} must be a mapping")
    result = _merge_mapping(result, selected_override)
    result["inference_profile"] = profile
    if "llm_compile" in result or "speech_compile" in result:
        raise ValueError("use the root compile switch")

    raw_runtime = result.get("runtime", {})
    if raw_runtime is None:
        raw_runtime = {}
    if not isinstance(raw_runtime, Mapping):
        raise ValueError("runtime must be a mapping")
    result["runtime"] = dict(raw_runtime)

    dialogue_model = _normalize_dialogue_sections(result)
    video = dict(result.get("video", {}))
    if "compile" in video or "graph" in video:
        raise ValueError("use the root compile and graph switches")
    compiled = result.get("compile", False)
    graph = result.get("graph", False)
    if type(compiled) is not bool or type(graph) is not bool:
        raise ValueError("compile and graph must be booleans")
    compile_sections = {
        "llm": _normalize_component_compile(
            {"enabled": compiled, "fallback_on_error": False}, component="llm"
        ),
        "speech": _normalize_component_compile(
            {"enabled": compiled, "fallback_on_error": False,
             "residual_cuda_graphs": graph}, component="speech"
        ),
        "video": {
            "enabled": compiled, "cuda_graphs": False, "backend": "inductor",
            "mode": "default", "dynamic": True, "fullgraph": False,
            "fallback_on_error": False, "keep_model_on_gpu": True,
            "recompile_limit": 64,
        },
    }
    video["compile"] = compile_sections["video"]
    video["graph"] = graph
    dialogue_model["llm_compile"] = compile_sections["llm"]
    dialogue_model["speech_compile"] = compile_sections["speech"]
    result["compile"] = compiled
    result["graph"] = graph
    execution = dict(result.get("execution", {}))
    speech_execution = dict(execution.get("speech", {}))

    if "attention_backend" in result:
        raise ValueError("use the root attn_implementation setting")
    requested_attention = result.get("attn_implementation", "auto")
    legacy_values = {
        "dialogue_model.attn_implementation": dialogue_model.pop("attn_implementation", None),
        "dialogue_model.decode_attn_implementation": dialogue_model.pop(
            "decode_attn_implementation", None
        ),
        "dialogue_model.speech_attn_implementation": dialogue_model.pop(
            "speech_attn_implementation", None
        ),
        "dialogue_model.speech_decode_attn_implementation": dialogue_model.pop(
            "speech_decode_attn_implementation", None
        ),
        "video.attn_implementation": video.pop("attn_implementation", None),
    }
    configured_legacy = {
        key: value for key, value in legacy_values.items() if value is not None
    }
    if configured_legacy:
        keys = ", ".join(sorted(configured_legacy))
        raise ValueError(
            "component attention settings are unsupported; use the root "
            "attn_implementation and remove " + keys
        )
    requested_attention_name = str(requested_attention)
    attention_policy = requested_attention_name.strip().lower().replace("-", "_")
    if graph and profile == "streaming" and attention_policy not in {
        "auto", "flash_attention_3", "flash_attn_3", "fa3",
    }:
        raise ValueError("graph: true requires attn_implementation: flash_attention_3 or auto")
    if attention_policy == "auto":
        resolved_attention = resolve_attn_implementation(
            "flash_attention_3", warn_on_fallback=False
        )
    else:
        resolved_attention = resolve_attn_implementation(requested_attention_name)
    dialogue_attention = speech_attention = video_attention = resolved_attention
    result["requested_attn_implementation"] = requested_attention_name
    result["attn_implementation"] = resolved_attention
    result["attention_backends"] = {
        "dialogue_model": dialogue_attention,
        "speech": speech_attention,
        "video": video_attention,
    }
    dialogue_model["attn_implementation"] = dialogue_attention
    dialogue_model["decode_attn_implementation"] = dialogue_attention
    dialogue_model["speech_attn_implementation"] = speech_attention
    dialogue_model["speech_decode_attn_implementation"] = speech_attention
    video["attn_implementation"] = video_attention

    expected_video_mode = "streaming" if profile == "streaming" else "full_sequence"
    configured_video_mode = str(video.get("mode", expected_video_mode)).lower()
    if configured_video_mode != expected_video_mode:
        raise ValueError(
            f"inference_profile={profile} requires video.mode={expected_video_mode}; "
            f"got {configured_video_mode}. Add a profiles.{profile}.video override."
        )
    video["mode"] = expected_video_mode
    video["causal_inference"] = profile == "streaming"

    if profile == "quality":
        execution["enabled"] = False
        speech_execution["async"] = False
        dialogue_model.update(
            stream=False,
            waveform_stream=False,
            text_use_kv_cache=False,
            speech_use_kv_cache=False,
            speech_residual_use_kv_cache=False,
            speech_text_chunking="complete",
        )
        video["use_kv_cache"] = False
        video["merge_lora"] = False
        # Keeping loaded Full-sequence modules resident avoids repeated transfers.
        video.setdefault("resident_models", True)
        for component_compile in compile_sections.values():
            component_compile["enabled"] = False
        compile_sections["speech"]["residual_cuda_graphs"] = False
        video["compile"] = compile_sections["video"]
        video["graph"] = False
        result["compile"] = False
        result["graph"] = False
    elif profile == "streaming":
        dialogue_model.setdefault("stream", True)
        dialogue_model.setdefault("waveform_stream", True)
        dialogue_model.setdefault("text_use_kv_cache", True)
        dialogue_model.setdefault("speech_use_kv_cache", True)
        dialogue_model.setdefault("speech_residual_use_kv_cache", True)
        dialogue_model.setdefault("speech_text_chunking", "complete")
        dialogue_model.setdefault("speech_min_text_chunk_tokens", 8)
        video.setdefault("use_kv_cache", True)
        video.setdefault("merge_lora", True)
    else:
        dialogue_model.setdefault("stream", True)
        dialogue_model.setdefault("waveform_stream", True)
        dialogue_model.setdefault("text_use_kv_cache", True)
        dialogue_model.setdefault("speech_use_kv_cache", True)
        dialogue_model.setdefault("speech_residual_use_kv_cache", True)
        video.setdefault("use_kv_cache", False)
        video.setdefault("merge_lora", True)

    execution["speech"] = speech_execution
    result["execution"] = execution
    result["dialogue_model"] = dialogue_model
    result["video"] = video
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load configuration files") from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    resolved = _resolve_tree(payload, key=None, base_dir=config_path.parent)
    return normalize_runtime_policy(resolved)


def _collect_unresolved(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_collect_unresolved(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_collect_unresolved(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and _is_placeholder(value):
        result.append(prefix)
    return result


def normalize_video_config(
    video: Mapping[str, Any],
    *,
    mode: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject required architecture defaults and validate an inference mode."""
    result = dict(video)
    selected_mode = str(mode or result.get("mode", "full_sequence"))
    result["mode"] = selected_mode

    result.setdefault("use_audio", True)
    result.setdefault("speech_token_num_codebooks", 16)
    result.setdefault("speech_token_rate", 12.5)
    result.setdefault("fps", 25)
    requested_attention = result.get("attn_implementation")
    if requested_attention is None and selected_mode == "streaming":
        requested_attention = "sdpa"
    result["attn_implementation"] = resolve_attn_implementation(
        requested_attention
    )
    result.setdefault("use_kv_cache", True)
    if type(result["use_kv_cache"]) is not bool:
        raise ValueError("video.use_kv_cache must be a boolean")
    expected_causal = selected_mode == "streaming"
    configured_causal = bool(
        result.get("causal_inference", expected_causal)
    )
    if configured_causal != expected_causal:
        raise ValueError(
            f"video.mode={selected_mode} requires "
            f"video.causal_inference={expected_causal}"
        )
    result["causal_inference"] = expected_causal

    raw_model_config = result.get("model_config")
    if raw_model_config is None:
        model_config: dict[str, Any] = {}
    elif isinstance(raw_model_config, Mapping):
        model_config = dict(raw_model_config)
    else:
        raise ValueError("video.model_config must be a mapping")
    model_config.setdefault("in_dim", 33)
    model_config.setdefault("audio_hidden_size", 32)
    result["model_config"] = model_config

    result.setdefault("graph", False)
    if type(result["graph"]) is not bool:
        raise ValueError("video.graph must be a boolean")

    raw_compile = result.get("compile", {})
    if not isinstance(raw_compile, Mapping):
        raise ValueError("compile.video must be a mapping")
    compile_config = dict(raw_compile)
    compile_config.setdefault("enabled", False)
    compile_config.setdefault("cuda_graphs", False)
    compile_config.setdefault("backend", "inductor")
    compile_config.setdefault("mode", "default")
    compile_config.setdefault("fullgraph", False)
    compile_config.setdefault("dynamic", False)
    compile_config.setdefault("fallback_on_error", True)
    compile_config.setdefault("keep_model_on_gpu", False)
    compile_config.setdefault("recompile_limit", 64)
    for key in (
        "enabled",
        "cuda_graphs",
        "fullgraph",
        "dynamic",
        "fallback_on_error",
        "keep_model_on_gpu",
    ):
        if type(compile_config[key]) is not bool:
            raise ValueError(f"compile.video.{key} must be a boolean")
    if compile_config["cuda_graphs"] and not compile_config["enabled"]:
        raise ValueError(
            "compile.video.cuda_graphs=true requires compile.video.enabled=true"
        )
    if compile_config["cuda_graphs"] and compile_config["dynamic"]:
        raise ValueError(
            "compile.video.cuda_graphs=true requires compile.video.dynamic=false"
        )
    if selected_mode == "streaming" and compile_config["fullgraph"]:
        raise ValueError(
            "compile.video.fullgraph is unsupported for Streaming streaming"
        )
    if (
        type(compile_config["recompile_limit"]) is not int
        or not 1 <= compile_config["recompile_limit"] <= 64
    ):
        raise ValueError(
            "compile.video.recompile_limit must be an integer in [1, 64]"
        )
    result["compile"] = compile_config

    if "vae_pipeline" in result:
        raise ValueError(
            "video.vae_pipeline has been removed; Streaming VAE offload is "
            "automatically enabled for a single-process runtime with at least "
            "two visible CUDA devices"
        )
    if "distributed_replica_mode" in result:
        raise ValueError("video.distributed_replica_mode is unsupported")

    for key, default in (
        ("resident_models", False),
        ("rank0_vae", True),
        ("merge_lora", False),
    ):
        result.setdefault(key, default)
        if type(result[key]) is not bool:
            raise ValueError(f"video.{key} must be a boolean")

    if int(result["speech_token_num_codebooks"]) != 16:
        raise ValueError("video.speech_token_num_codebooks must be 16")
    if float(result["speech_token_rate"]) != 12.5:
        raise ValueError("video.speech_token_rate must be 12.5")
    if float(result["fps"]) != 25:
        raise ValueError("video.fps must be 25 for the supplied 1.3B recipe")
    if selected_mode not in {"full_sequence", "streaming"}:
        raise ValueError("video.mode must be 'full_sequence' or 'streaming'")
    if result["use_audio"] is not True:
        raise ValueError("video.use_audio must be true")
    if int(model_config["in_dim"]) != 33:
        raise ValueError("video.model_config.in_dim must be 33")
    if int(model_config["audio_hidden_size"]) != 32:
        raise ValueError("video.model_config.audio_hidden_size must be 32")

    window_frames = result.get("window_frames")
    if window_frames is not None:
        if selected_mode != "full_sequence":
            raise ValueError(
                "video.window_frames is supported only in Full-sequence mode"
            )
        if type(window_frames) is not int or window_frames < 5:
            raise ValueError(
                "video.window_frames must be an integer >= 5 or null"
            )
        if (window_frames - 1) % 4 != 0:
            raise ValueError(
                "video.window_frames must satisfy (frames - 1) % 4 == 0"
            )
    elif (
        selected_mode == "full_sequence"
        and int(result.get("max_tokens", 30000)) <= 0
    ):
        raise ValueError(
            "Full-sequence video.max_tokens must be positive when window_frames is null"
        )

    if selected_mode == "streaming":
        lora_checkpoint = result.get("lora_checkpoint")
        if lora_checkpoint in (None, "", "none", "None"):
            raise ValueError(
                "video.lora_checkpoint is required in Streaming mode"
            )
        schedule_overrides = dict(overrides or {})
        effective = {**result, **schedule_overrides}
        if float(effective.get("cfg", effective.get("guidance_scale", 1.0))) != 1.0:
            raise ValueError("Streaming video.cfg must be 1.0")
        if float(effective.get("audio_cfg", effective.get("audio_scale", 1.0))) != 1.0:
            raise ValueError("Streaming video.audio_cfg must be 1.0")
        base_steps = int(result.get("steps", result.get("num_steps", 8)))
        if "steps" in schedule_overrides:
            raw_steps = schedule_overrides["steps"]
        elif "num_steps" in schedule_overrides:
            raw_steps = schedule_overrides["num_steps"]
        else:
            raw_steps = base_steps
        steps = int(raw_steps)
        from ex_omni.model.video_generator.schedulers.contracts import FlowMapSchedule

        if steps != 8:
            raise ValueError("Streaming video requires exactly 8 diffusion steps")
        for key in ("num_steps", "flowmap_num_inference_steps", "preferred_steps"):
            if key in effective and int(effective[key]) != 8:
                raise ValueError(f"Streaming video.{key} must be 8")
        result["steps"] = steps
        if "num_steps" in result or "num_steps" in schedule_overrides:
            result["num_steps"] = steps
        if "flowmap_num_inference_steps" in result:
            result["flowmap_num_inference_steps"] = steps
        sequence_parallel = int(
            effective.get("sequence_parallel", effective.get("sp_size", 1))
        )
        if sequence_parallel <= 0:
            raise ValueError("Streaming video.sequence_parallel must be positive")
        result["sequence_parallel"] = sequence_parallel
        if int(effective.get("stream_chunk_latent_frames", 3)) != 3:
            raise ValueError(
                "FAR Streaming video.stream_chunk_latent_frames must be 3"
            )
        if any(key in effective for key in (
            "stream_rapr_anchor_frames", "stream_rapr_max_distance", "stream_persistent_sink",
        )):
            raise ValueError("Legacy RAPR/persistent_sink switches are unsupported; use a sink-trained Streaming checkpoint")
        for key, default in (
            ("stream_first_chunk_latent_frames", 1),
            ("stream_full_chunk_limit", 3),
            ("stream_max_compressed_latent_frames", 96),
        ):
            result[key] = int(effective.get(key, default))
        if result["stream_first_chunk_latent_frames"] != 1:
            raise ValueError("Streaming streaming requires one reference-only latent frame")
        for key, default in (
            ("stream_full_patch_size", [1, 2, 2]),
            ("stream_compressed_patch_size", [1, 4, 4]),
        ):
            result[key] = [int(item) for item in effective.get(key, default)]
        if "stream_enable_far_compression" in effective:
            raise ValueError("Streaming streaming always uses FAR compression; remove stream_enable_far_compression")
        if "stream_sink_rope_distance" in result:
            distance = result["stream_sink_rope_distance"]
            if type(distance) is not int or distance <= 0:
                raise ValueError("video.stream_sink_rope_distance must be a positive integer")
        if not bool(effective.get("use_kv_cache", True)):
            raise ValueError("FAR Streaming video.use_kv_cache must be true")
        result["use_kv_cache"] = True
        if float(effective.get("tea_cache_l1_thresh", 0)) != 0:
            raise ValueError(
                "AnyFlow Streaming video.tea_cache_l1_thresh must be 0"
            )
        shift = float(effective.get("sigma_shift", 5.0))
        gate = float(effective.get("flowmap_gate", 0.25))
        deltatime_type = str(effective.get("deltatime_type", "r"))
        training_timesteps = int(effective.get("training_timesteps", 1000))
        base_checkpoint = effective.get("base_checkpoint")
        if base_checkpoint in (None, "", "none", "None"):
            raise ValueError("Streaming video.base_checkpoint is required")
        schedule = FlowMapSchedule.shifted(
            steps,
            shift=shift,
            num_train_timesteps=training_timesteps,
        )
        derived = list(schedule.source_timesteps)
        result["streaming_timesteps"] = derived
        result["sigma_shift"] = shift
        result["flowmap_gate"] = gate
        result["deltatime_type"] = deltatime_type
        result["training_timesteps"] = training_timesteps
        result["base_checkpoint"] = base_checkpoint
    return result


def validate_config(
    path: str | Path,
    *,
    check_weights: bool = False,
    check_world_size: bool = True,
) -> ValidationReport:
    config_path = Path(path).expanduser().resolve()
    config = load_config(config_path)
    dialogue_model = config.get("dialogue_model")
    video = config.get("video")
    if not isinstance(dialogue_model, dict) or not isinstance(video, dict):
        raise ValueError("configuration requires 'dialogue_model' and 'video' mappings")

    normalized_video = normalize_video_config(video)
    from ex_omni.execution import normalize_execution_config

    execution = normalize_execution_config(config)
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping")
    ffmpeg_path = runtime.get("ffmpeg_path")
    if ffmpeg_path not in (None, "", "none", "None"):
        executable = Path(str(ffmpeg_path))
        if not executable.is_file():
            raise FileNotFoundError(
                f"configured runtime.ffmpeg_path does not exist: {executable}"
            )
        if not os.access(executable, os.X_OK):
            raise PermissionError(
                f"configured runtime.ffmpeg_path is not executable: {executable}"
            )
    mode = normalized_video["mode"]
    distributed = bool(normalized_video.get("distributed", False))
    if int(dialogue_model.get("max_speech_tokens", 750)) <= 0:
        raise ValueError("dialogue_model.max_speech_tokens must be positive")
    if type(dialogue_model.get("speech_do_sample", True)) is not bool:
        raise ValueError("dialogue_model.speech_do_sample must be a boolean")
    if int(dialogue_model.get("speech_top_k", 50)) < 0:
        raise ValueError("dialogue_model.speech_top_k must be non-negative")
    speech_top_p = float(dialogue_model.get("speech_top_p", 1.0))
    if not 0 < speech_top_p <= 1:
        raise ValueError("dialogue_model.speech_top_p must be in (0, 1]")
    if float(dialogue_model.get("speech_temperature", 0.9)) <= 0:
        raise ValueError("dialogue_model.speech_temperature must be positive")
    if float(dialogue_model.get("speech_repetition_penalty", 1.05)) <= 0:
        raise ValueError("dialogue_model.speech_repetition_penalty must be positive")
    if int(dialogue_model.get("speech_stream_chunk_units", 6)) <= 0:
        raise ValueError("dialogue_model.speech_stream_chunk_units must be positive")
    speech_text_chunking = str(
        dialogue_model.get("speech_text_chunking", "complete")
    ).lower()
    if speech_text_chunking not in {"complete", "sentence"}:
        raise ValueError(
            "speech.text_chunking must be complete or sentence"
        )
    if int(dialogue_model.get("speech_min_text_chunk_tokens", 8)) <= 0:
        raise ValueError("speech.min_text_chunk_tokens must be positive")
    resolve_attn_implementation(dialogue_model.get("attn_implementation"))
    resolve_decode_attn_implementation(
        dialogue_model.get("decode_attn_implementation")
    )
    if type(dialogue_model.get("text_use_kv_cache", True)) is not bool:
        raise ValueError("dialogue_model.text_use_kv_cache must be a boolean")
    if type(dialogue_model.get("speech_use_kv_cache", True)) is not bool:
        raise ValueError("dialogue_model.speech_use_kv_cache must be a boolean")
    waveform_stream = dialogue_model.get("waveform_stream", False)
    if type(waveform_stream) is not bool:
        raise ValueError("dialogue_model.waveform_stream must be a boolean")
    if waveform_stream:
        if not distributed and dialogue_model.get("stream") is not True:
            raise ValueError(
                "dialogue_model.waveform_stream=true requires dialogue_model.stream=true"
            )
        if dialogue_model.get("code2wav_model_path") in (None, "", "none", "None"):
            raise ValueError(
                "waveform streaming requires dialogue_model.code2wav_model_path"
            )
        initial_audio = int(dialogue_model.get("waveform_initial_chunk_units", 0))
        audio_chunk = int(dialogue_model.get("waveform_chunk_units", 0))
        audio_context = int(
            dialogue_model.get("waveform_left_context_units", -1)
        )
        if initial_audio <= 0 or audio_chunk < initial_audio or audio_context < 0:
            raise ValueError(
                "waveform streaming requires 0 < "
                "waveform_initial_chunk_units <= waveform_chunk_units "
                "and waveform_left_context_units >= 0"
            )
        if mode == "streaming" and (
            initial_audio != 6 or audio_chunk != 6
        ):
            raise ValueError(
                "Streaming audiovisual streaming requires "
                "waveform_initial_chunk_units=6 and waveform_chunk_units=6"
            )
    sequence_parallel = int(normalized_video.get("sequence_parallel", 1))
    use_fsdp = bool(normalized_video.get("use_fsdp", distributed))
    if distributed:
        if sequence_parallel > 1 and use_fsdp:
            raise ValueError(
                "streaming sequence parallelism replicates DiT parameters and "
                "requires video.use_fsdp=false"
            )
        if sequence_parallel == 1 and not use_fsdp:
            raise ValueError(
                "distributed video with sequence_parallel=1 requires "
                "video.use_fsdp=true"
            )
    # A split deployment can have distributed Video and a single-GPU LLM.
    # Shared-world LLM requirements must not be inferred from Video alone.
    split_single_llm = execution["enabled"] and execution["llm"]["parallelism"] == "single"
    if distributed and not split_single_llm:
        dialogue_model_backend = str(dialogue_model.get("distributed_backend", "")).lower()
        if dialogue_model_backend not in {"native_tp", "deepspeed"}:
            raise ValueError(
                "all-shared distributed mode requires "
                "dialogue_model.distributed_backend=native_tp or deepspeed"
            )
        tp_size = int(dialogue_model.get("tp_size", 0))
        if tp_size <= 0:
            raise ValueError("dialogue_model.tp_size must be positive")
        if str(dialogue_model.get("load_method", "")) != "full_model":
            raise ValueError("distributed dialogue_model requires load_method=full_model")
        if str(dialogue_model.get("device_map", "")) != "cuda":
            raise ValueError("distributed dialogue_model requires device_map=cuda")
        stream = dialogue_model.get("stream")
        if type(stream) is not bool:
            raise ValueError(
                "distributed dialogue_model requires an explicit boolean dialogue_model.stream"
            )
        expected_stream = mode == "streaming"
        if stream != expected_stream:
            raise ValueError(
                f"video.mode={mode} requires dialogue_model.stream={expected_stream}"
            )
        waveform_stream = dialogue_model.get("waveform_stream")
        if type(waveform_stream) is not bool:
            raise ValueError(
                "distributed dialogue_model requires an explicit boolean "
                "dialogue_model.waveform_stream"
            )
        if waveform_stream != expected_stream:
            raise ValueError(
                f"video.mode={mode} requires "
                f"dialogue_model.waveform_stream={expected_stream}"
            )
        if stream:
            if int(dialogue_model.get("num_beams", 0)) != 1:
                raise ValueError(
                    "streaming distributed dialogue_model requires dialogue_model.num_beams=1"
                )
        launched_world = int(os.environ.get("WORLD_SIZE", "0"))
        if (
            check_world_size
            and launched_world > 0
            and tp_size != launched_world
        ):
            raise ValueError(
                f"dialogue_model.tp_size must equal WORLD_SIZE ({tp_size}!={launched_world})"
            )

    unresolved = tuple(_collect_unresolved(config))
    if check_weights:
        if unresolved:
            raise ValueError(f"unresolved configuration values: {', '.join(unresolved)}")
        missing = []
        for section in (dialogue_model, video):
            for key, value in section.items():
                if key in WEIGHT_PATH_KEYS and value not in (None, "", "none", "None"):
                    if is_hf_reference(value):
                        parse_hf_reference(str(value))
                    elif not Path(str(value)).exists():
                        missing.append(f"{key}={value}")
        if missing:
            raise FileNotFoundError("missing configured weights: " + ", ".join(missing))
    return ValidationReport(config_path, mode, unresolved, check_weights)
