"""One-world distributed context shared by dialogue_model TP and video FSDP."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    process_group: Any = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    def boundary_status(self, *, cancelled: bool = False, failed: bool = False) -> tuple[bool, bool]:
        """Synchronize cancellation/error only at span and video chunk boundaries."""
        if not self.is_distributed:
            return bool(cancelled), bool(failed)
        import torch
        import torch.distributed as dist

        device = torch.device("cuda", self.local_rank)
        flags = torch.tensor(
            [int(cancelled), int(failed)], dtype=torch.int32, device=device
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.process_group)
        return bool(flags[0].item()), bool(flags[1].item())

    def broadcast_tensor(self, tensor, *, src: int = 0):
        if self.is_distributed:
            import torch.distributed as dist

            dist.broadcast(tensor, src=src, group=self.process_group)
        return tensor


def initialize_distributed(backend: str = "nccl") -> DistributedContext:
    """Initialize torchrun's process group and select the local CUDA device."""
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process inference requires CUDA")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            kwargs = {"backend": backend, "init_method": "env://"}
            try:
                dist.init_process_group(
                    **kwargs, device_id=torch.device("cuda", local_rank)
                )
            except TypeError:
                dist.init_process_group(**kwargs)
        return DistributedContext(
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            process_group=dist.group.WORLD,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistributedContext(local_rank=local_rank)


def current_context() -> DistributedContext:
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return DistributedContext(local_rank=int(os.environ.get("LOCAL_RANK", "0")))

    if dist.is_available() and dist.is_initialized():
        return DistributedContext(
            rank=dist.get_rank(),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=dist.get_world_size(),
            process_group=dist.group.WORLD,
        )
    return DistributedContext(local_rank=int(os.environ.get("LOCAL_RANK", "0")))
