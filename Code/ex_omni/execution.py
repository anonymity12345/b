"""Validated GPU topology for low-latency pipeline execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _gpu_ids(value: Any, *, name: str, default: list[int]) -> list[int]:
    if value is None:
        result = list(default)
    elif isinstance(value, str):
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        result = [int(item) for item in value]
    else:
        raise ValueError(
            f"execution.{name}.gpu_ids must be a list or comma-separated string"
        )
    if not result:
        raise ValueError(f"execution.{name}.gpu_ids cannot be empty")
    if any(item < 0 for item in result):
        raise ValueError(f"execution.{name}.gpu_ids must be non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"execution.{name}.gpu_ids must be unique")
    return result


def _component(
    raw: Any,
    *,
    name: str,
    default_gpu_ids: list[int],
    default_parallelism: str,
    allowed_parallelism: set[str],
) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"execution.{name} must be a mapping")
    result = dict(raw)
    result["gpu_ids"] = _gpu_ids(
        result.get("gpu_ids"),
        name=name,
        default=default_gpu_ids,
    )
    parallelism = str(result.get("parallelism", default_parallelism)).lower()
    if parallelism not in allowed_parallelism:
        choices = ", ".join(sorted(allowed_parallelism))
        raise ValueError(
            f"execution.{name}.parallelism must be one of: {choices}"
        )
    result["parallelism"] = parallelism
    return result


def normalize_execution_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate the root ``execution`` topology mapping."""
    raw = config.get("execution", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("execution must be a mapping")
    result = dict(raw)
    result.setdefault("enabled", False)
    if type(result["enabled"]) is not bool:
        raise ValueError("execution.enabled must be a boolean")

    if "dialogue_model" in result or "planner" in result:
        raise ValueError("execution uses llm; legacy component names are unsupported")
    dialogue_model = _component(
        result.get("llm"),
        name="llm",
        default_gpu_ids=[0],
        default_parallelism="single",
        allowed_parallelism={"single", "tensor_parallel"},
    )
    speech = _component(
        result.get("speech"),
        name="speech",
        default_gpu_ids=[1],
        default_parallelism="single",
        allowed_parallelism={"single"},
    )
    video = _component(
        result.get("video"),
        name="video",
        default_gpu_ids=[2],
        default_parallelism="single",
        allowed_parallelism={
            "single",
            "pipeline",
            "fsdp",
            "sequence_parallel",
        },
    )

    speech.setdefault("async", True)
    if type(speech["async"]) is not bool:
        raise ValueError("execution.speech.async must be a boolean")
    speech.setdefault("process_isolation", True)
    if type(speech["process_isolation"]) is not bool:
        raise ValueError(
            "execution.speech.process_isolation must be a boolean"
        )
    speech.setdefault("device", "auto")
    waveform_gpu_id = speech.get("waveform_gpu_id")
    if waveform_gpu_id is not None:
        waveform_gpu_id = int(waveform_gpu_id)
        if waveform_gpu_id < 0:
            raise ValueError(
                "execution.speech.waveform_gpu_id must be non-negative"
            )
    speech["waveform_gpu_id"] = waveform_gpu_id
    if len(speech["gpu_ids"]) != 1:
        raise ValueError(
            "execution.speech currently supports exactly one GPU per Speech worker"
        )
    if dialogue_model["parallelism"] == "single" and len(dialogue_model["gpu_ids"]) != 1:
        raise ValueError(
            "execution.llm.parallelism=single requires exactly one GPU"
        )
    if (
        dialogue_model["parallelism"] == "tensor_parallel"
        and len(dialogue_model["gpu_ids"]) < 2
    ):
        raise ValueError(
            "execution.llm.parallelism=tensor_parallel requires at least two GPUs"
        )
    if video["parallelism"] == "single" and len(video["gpu_ids"]) != 1:
        raise ValueError(
            "execution.video.parallelism=single requires exactly one GPU"
        )
    if "gpus_per_request" in video:
        raise ValueError("execution.video.gpus_per_request is unsupported")
    if video["parallelism"] == "sequence_parallel":
        configured_sp = int(
            dict(config.get("video", {})).get("sequence_parallel", 1)
        )
        if len(video["gpu_ids"]) < 2 or configured_sp != len(video["gpu_ids"]):
            raise ValueError(
                "sequence_parallel requires at least two Video GPUs and "
                "video.sequence_parallel equal to the Video GPU count"
            )

    if result["enabled"]:
        groups = {
            "llm": set(dialogue_model["gpu_ids"]),
            "speech": set(speech["gpu_ids"]),
            "waveform": (
                {waveform_gpu_id} if waveform_gpu_id is not None else set()
            ),
            "video": set(video["gpu_ids"]),
        }
        for left, right in (
            ("llm", "speech"),
            ("llm", "waveform"),
            ("llm", "video"),
            ("speech", "waveform"),
            ("speech", "video"),
            ("waveform", "video"),
        ):
            overlap = sorted(groups[left] & groups[right])
            if overlap:
                raise ValueError(
                    f"execution GPU groups must be disjoint; "
                    f"{left}/{right} overlap on {overlap}"
                )

    result["llm"] = dialogue_model
    result["speech"] = speech
    result["video"] = video
    if "replica_groups" in result:
        raise ValueError("execution.replica_groups is unsupported; configure one GPU group")
    vae_gpu_id = result.get("vae_gpu_id")
    if vae_gpu_id is not None:
        vae_gpu_id = int(vae_gpu_id)
        if vae_gpu_id < 0:
            raise ValueError("execution.vae_gpu_id must be non-negative")
        occupied = set(dialogue_model["gpu_ids"] + speech["gpu_ids"] + video["gpu_ids"])
        if waveform_gpu_id is not None:
            occupied.add(waveform_gpu_id)
        if vae_gpu_id in occupied:
            raise ValueError("dedicated VAE GPU must be disjoint from the other GPU groups")
    result["vae_gpu_id"] = vae_gpu_id
    return result


def dialogue_model_config_with_execution(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Dialogue Model section with pipeline execution controls applied."""
    dialogue_model = dict(config.get("dialogue_model", config))
    if "dialogue_model" not in config:
        return dialogue_model
    execution = normalize_execution_config(config)
    if not execution["enabled"]:
        return dialogue_model
    llm = execution["llm"]
    speech = execution["speech"]
    if str(dialogue_model.get("engine", "native")).lower() == "vllm_omni":
        raw_vllm = dialogue_model.get("vllm")
        if not isinstance(raw_vllm, Mapping):
            raise ValueError(
                "llm.vllm must be a mapping for engine=vllm_omni"
            )
        vllm = dict(raw_vllm)
        vllm["gpu_ids"] = list(llm["gpu_ids"])
        vllm["tensor_parallel_size"] = len(llm["gpu_ids"])
        dialogue_model["vllm"] = vllm
    dialogue_model["speech_async"] = bool(speech["async"])
    dialogue_model["speech_process_isolation"] = bool(
        speech["process_isolation"]
    )
    dialogue_model["speech_gpu_id"] = int(speech["gpu_ids"][0])
    dialogue_model["speech_device"] = str(speech["device"])
    dialogue_model["code2wav_gpu_id"] = speech["waveform_gpu_id"]
    if str(dialogue_model.get("code2wav_device", "auto")) == "auto":
        dialogue_model["code2wav_device"] = str(speech["device"])
    return dialogue_model


def derive_request_topology_config(
    config: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Adapt a production topology to one request worker group."""
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    result = deepcopy(dict(config))
    dialogue_model = result.get("dialogue_model")
    video = result.get("video")
    if not isinstance(dialogue_model, dict) or not isinstance(video, dict):
        raise ValueError("configuration requires dialogue_model and video mappings")

    dialogue_model["tp_size"] = world_size
    execution = normalize_execution_config(result)
    configured_parallelism = execution["video"]["parallelism"]
    if world_size == 1:
        video["distributed"] = False
        video["use_fsdp"] = False
        video["sequence_parallel"] = 1
        video["sp_size"] = 1
    elif configured_parallelism == "sequence_parallel":
        if world_size != len(execution["video"]["gpu_ids"]):
            raise ValueError(
                "Video sequence parallel world_size must match its GPU count"
            )
        video["distributed"] = True
        video["use_fsdp"] = False
        video["sequence_parallel"] = world_size
        video["sp_size"] = world_size
    else:
        video["distributed"] = True
        video["use_fsdp"] = True
        video["sequence_parallel"] = 1
        video["sp_size"] = 1
    return result
