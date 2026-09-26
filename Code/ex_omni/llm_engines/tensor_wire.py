"""Small binary tensor codec for cross-environment runtime services."""

from __future__ import annotations

from typing import Any


def encode_bf16_tensor(tensor) -> dict[str, Any]:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return {
        "shape": list(value.shape),
        "dtype": "bfloat16",
        "data": value.view(torch.uint8).numpy().tobytes(),
    }


def decode_bf16_tensor(payload: dict[str, Any]):
    import torch

    if payload.get("dtype") != "bfloat16":
        raise ValueError(f"unsupported wire tensor dtype: {payload.get('dtype')}")
    raw = bytearray(payload["data"])
    return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(payload["shape"])

