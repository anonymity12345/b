# Shared inference attention backend selection.

from __future__ import annotations

from functools import lru_cache
import sys
import warnings


FLASH_ATTENTION_3 = "flash_attention_3"
FLASH_ATTENTION_2 = "flash_attention_2"
SDPA = "sdpa"


@lru_cache(maxsize=None)
def flash_attention_module(implementation: str):
    if implementation == FLASH_ATTENTION_3:
        try:
            from flash_attn_3 import flash_attn_interface
        except ImportError:
            try:
                import flash_attn_interface
            except ImportError:
                from .vllm_flash_attention import vllm_flash_attention_module

                return vllm_flash_attention_module(3)

        # Transformers 4.57 imports the FA3 interface by this top-level name.
        sys.modules.setdefault("flash_attn_interface", flash_attn_interface)
        return flash_attn_interface
    if implementation == FLASH_ATTENTION_2:
        try:
            import flash_attn
        except ImportError:
            from .vllm_flash_attention import vllm_flash_attention_module

            return vllm_flash_attention_module(2)
        return flash_attn
    raise ValueError(f"{implementation} is not a FlashAttention backend")


@lru_cache(maxsize=None)
def flash_attention_function(implementation: str):
    return flash_attention_module(implementation).flash_attn_func


def flash_attention_3_available() -> bool:
    try:
        module = flash_attention_module(FLASH_ATTENTION_3)
        import torch

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return False
        return hasattr(module, "flash_attn_func")
    except Exception:
        return False


def flash_attention_2_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False
        return hasattr(
            flash_attention_module(FLASH_ATTENTION_2),
            "flash_attn_func",
        )
    except Exception:
        return False


def resolve_attn_implementation(
    value: str | None,
    *,
    warn_on_fallback: bool = True,
) -> str:
    normalized = str(value or FLASH_ATTENTION_2).strip().lower()
    normalized = normalized.replace("-", "_")
    aliases = {
        "flash": FLASH_ATTENTION_2,
        "flash_attn": FLASH_ATTENTION_2,
        "flash_attention": FLASH_ATTENTION_2,
        "fa2": FLASH_ATTENTION_2,
        "flash_attn_2": FLASH_ATTENTION_2,
        "fa3": FLASH_ATTENTION_3,
        "flash_attn_3": FLASH_ATTENTION_3,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        FLASH_ATTENTION_3,
        FLASH_ATTENTION_2,
        SDPA,
        "eager",
    }:
        raise ValueError(
            "attn_implementation must be flash_attention_3, "
            "flash_attention_2, sdpa, or eager"
        )
    if normalized == FLASH_ATTENTION_3 and not flash_attention_3_available():
        fallback = FLASH_ATTENTION_2 if flash_attention_2_available() else SDPA
        if warn_on_fallback:
            warnings.warn(
                "flash-attn-3 is unavailable or incompatible; falling back to "
                + fallback,
                RuntimeWarning,
                stacklevel=2,
            )
        return fallback
    if normalized == FLASH_ATTENTION_2 and not flash_attention_2_available():
        if warn_on_fallback:
            warnings.warn(
                "flash-attn is unavailable or incompatible; falling back to SDPA",
                RuntimeWarning,
                stacklevel=2,
            )
        return SDPA
    return normalized


def resolve_decode_attn_implementation(value: str | None) -> str:
    return resolve_attn_implementation(value)


_FLASH_DECODING_REGISTERED: set[tuple[str, str]] = set()


def register_flash_decoding(
    implementation: str,
    decode_implementation: str | None = None,
) -> bool:
    # Combine the selected prefill backend with the configured decode kernel.
    implementation = resolve_attn_implementation(implementation)
    decode_implementation = resolve_decode_attn_implementation(
        decode_implementation or implementation
    )
    registration = (implementation, decode_implementation)
    if registration in _FLASH_DECODING_REGISTERED:
        return True
    if implementation not in {FLASH_ATTENTION_3, FLASH_ATTENTION_2}:
        return False
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_forward = ALL_ATTENTION_FUNCTIONS[implementation]
    sdpa_forward = ALL_ATTENTION_FUNCTIONS[SDPA]
    flash_attn_with_kvcache = None
    if decode_implementation in {FLASH_ATTENTION_3, FLASH_ATTENTION_2}:
        flash_attn_with_kvcache = getattr(
            flash_attention_module(decode_implementation),
            "flash_attn_with_kvcache",
            None,
        )
        if flash_attn_with_kvcache is None:
            return False

    def flash_decoding_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        sliding_window=None,
        **kwargs,
    ):
        can_decode = (
            not module.training
            and query.shape[-2] == 1
            and attention_mask is None
            and float(dropout) == 0.0
            and sliding_window in (None, -1)
            and not kwargs.get("output_attentions", False)
        )
        if can_decode and decode_implementation == SDPA:
            return sdpa_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=False,
                **kwargs,
            )
        if can_decode and flash_attn_with_kvcache is not None:
            output = flash_attn_with_kvcache(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                cache_seqlens=int(key.shape[-2]),
                softmax_scale=scaling,
                causal=True,
            )
            return output, None
        return original_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

    ALL_ATTENTION_FUNCTIONS.register(
        implementation,
        flash_decoding_forward,
    )
    _FLASH_DECODING_REGISTERED.add(registration)
    return True


def register_flash_attention_2_decoding() -> bool:
    # Backward-compatible FA2 registration helper.
    return register_flash_decoding(FLASH_ATTENTION_2)
