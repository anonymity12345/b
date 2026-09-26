"""Side-effect-free Ex-Omni inference utilities."""

import os
import re


def get_local_rank() -> int:
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return int(os.environ.get("LOCAL_RANK", 0))
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("LOCAL_RANK", 0))


def rank0_print(*args, **kwargs) -> None:
    if get_local_rank() == 0:
        print(*args, flush=True, **kwargs)


def detect_language(text: str) -> str:
    english = len(re.findall(r"[a-zA-Z]", text or ""))
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return "zh" if chinese > english else "en"


def disable_torch_init() -> None:
    import torch

    torch.nn.Linear.reset_parameters = lambda self: None
    torch.nn.LayerNorm.reset_parameters = lambda self: None
