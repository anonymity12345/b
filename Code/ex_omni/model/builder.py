# Licensed under the Apache License, Version 2.0.
"""Inference-only Ex-Omni model loader."""

from __future__ import annotations

import importlib
import os
from typing import Any, Mapping

from ex_omni.chat_templates import configure_chat_template
from ex_omni.constants import ASSISTANT_PROTOCOL_TOKENS
from ex_omni.model.attention import (
    FLASH_ATTENTION_3,
    FLASH_ATTENTION_2,
    register_flash_decoding,
    resolve_attn_implementation,
    resolve_decode_attn_implementation,
)


def _dtype(torch, value: Any):
    if value in (True, "bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in (False, "fp32", "float32"):
        return torch.float32
    return None


def _resolve_model_class(path: str | None):
    if not path:
        return None
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError("model_class must be a fully-qualified class name")
    return getattr(importlib.import_module(module_name), class_name)


def load_model_for_inference(
    model_name_or_path: str,
    *,
    load_bf16: bool | None = None,
    dtype: str | None = None,
    device_map: Any = "auto",
    attn_implementation: str | None = None,
    decode_attn_implementation: str | None = None,
    model_class: str | None = None,
    load_method: str = "hf_auto_full",
    trust_remote_code: bool = True,
    pretrained_reference_overrides: Mapping[str, str] | None = None,
    **kwargs: Any,
):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    # Importing the model class registers ``llava_her_qwen3`` with
    # Transformers before AutoConfig reads the checkpoint.
    from ex_omni.model import LlavaHerQwen3ForCausalLM

    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    from ex_omni.hub import materialize_pretrained_reference

    attributes = (
        "pretrain_vision_encoder_weights",
        "pretrain_speech_encoder_weights",
        "pretrain_qwen_tts_weights",
    )
    overrides = dict(pretrained_reference_overrides or {})
    if set(overrides) - set(attributes):
        raise ValueError("unknown pretrained_reference_overrides: "
                         + ", ".join(sorted(set(overrides) - set(attributes))))
    for attribute in attributes:
        reference = overrides.get(attribute, getattr(config, attribute, None))
        if reference not in (None, "", "none", "None"):
            setattr(
                config,
                attribute,
                materialize_pretrained_reference(str(reference)),
            )
    config.inference = True
    config.model_name_or_path = model_name_or_path
    normalized_attention = resolve_attn_implementation(attn_implementation)
    config.attn_implementation = normalized_attention
    config._attn_implementation = normalized_attention
    normalized_decode_attention = resolve_decode_attn_implementation(
        decode_attn_implementation
    )
    config.decode_attn_implementation = normalized_decode_attention
    if normalized_attention in {FLASH_ATTENTION_3, FLASH_ATTENTION_2}:
        register_flash_decoding(
            normalized_attention,
            normalized_decode_attention,
        )
    resolved_dtype = _dtype(torch, dtype if dtype is not None else load_bf16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, trust_remote_code=trust_remote_code
    )
    # Register multimodal placeholders before applying the chat template. The
    # checkpoint reserves embedding rows for these tokens; without registration,
    # token lookup returns None and image-token counting fails.
    tokenizer.add_tokens(["<speech>", "<image>"], special_tokens=True)
    configure_chat_template(tokenizer)
    missing_protocol = [
        token for token in ASSISTANT_PROTOCOL_TOKENS if token not in tokenizer.get_vocab()
    ]
    if missing_protocol:
        raise ValueError(
            "checkpoint tokenizer lacks trained atomic assistant protocol tokens: "
            + ", ".join(missing_protocol)
        )
    cls = _resolve_model_class(model_class) or LlavaHerQwen3ForCausalLM
    load_kwargs = dict(kwargs)
    load_kwargs.update(
        config=config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if resolved_dtype is not None:
        load_kwargs["torch_dtype"] = resolved_dtype
    if load_method == "hf_auto_full":
        loader = getattr(cls, "from_pretrained_hf_auto_full", None)
        if loader is None:
            raise ValueError(
                f"{cls.__name__} does not support load_method='hf_auto_full'; "
                "set dialogue_model.load_method='from_pretrained' for this custom model class"
            )
        model = loader(model_name_or_path, **load_kwargs)
    elif load_method == "from_pretrained":
        model = cls.from_pretrained(model_name_or_path, **load_kwargs)
    elif load_method == "full_model":
        if device_map not in (None, "", "cuda", {"": "cuda"}):
            raise ValueError(
                "load_method='full_model' requires device_map='cuda' (the current "
                "LOCAL_RANK device); HF device_map='auto' is not valid for TP"
            )
        load_kwargs.pop("device_map", None)
        load_kwargs.pop("trust_remote_code", None)
        load_kwargs.pop("config", None)
        load_kwargs.pop("torch_dtype", None)
        model = cls.build_full_model_from_pretrained(
            config=config,
            torch_dtype=resolved_dtype,
            low_cpu_mem_usage=False,
            **load_kwargs,
        )
        if torch.cuda.is_available():
            local_rank = int(
                os.environ.get("LOCAL_RANK", torch.cuda.current_device())
            )
            torch.cuda.set_device(local_rank)
            target = torch.device("cuda", local_rank)
        else:
            target = torch.device("cpu")
        model = model.to(device=target, dtype=resolved_dtype)
    else:
        raise ValueError(
            "dialogue_model.load_method must be 'hf_auto_full', 'from_pretrained', or 'full_model'"
        )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return tokenizer, model
