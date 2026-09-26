"""Minimal WORLD-group collectives for streaming sequence parallelism."""

from __future__ import annotations

import torch


class WorldSequenceParallelGroup:
    @property
    def world_size(self) -> int:
        return get_sequence_parallel_world_size()

    @property
    def rank_in_group(self) -> int:
        return get_sequence_parallel_rank()

    @torch.compiler.disable
    def all_gather(self, tensor, *, dim: int = 0):
        # Keep the explicit NCCL wait outside Inductor's functional collectives.
        # The video service enables blocking waits before group initialization.
        import torch.distributed as dist

        if self.world_size == 1:
            return tensor
        # NCCL can write one contiguous allocation directly. In the common
        # batch-one sequence gather, the final flatten is a view: no per-rank
        # allocations or extra full K/V concatenation kernel are needed.
        dim = dim % tensor.ndim
        gathered = torch.empty(
            (self.world_size, *tensor.shape), dtype=tensor.dtype, device=tensor.device,
        )
        if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            # Work.wait() honors BLOCKING_WAIT and polls a CUDA event on the
            # host, which is illegal during capture. synchronize() only joins
            # the NCCL stream into the capturing stream (PyTorch 2.7).
            work = dist.all_gather_into_tensor(
                gathered.flatten(0, 1), tensor.contiguous(), async_op=True)
            work.synchronize()
        else:
            dist.all_gather_into_tensor(gathered.flatten(0, 1), tensor.contiguous())
        return gathered.movedim(0, dim).flatten(dim, dim + 1)

    @torch.compiler.disable
    def sequence_to_heads(self, tensor, *, num_heads: int):
        """[B, local tokens, 3, D] -> [B, all tokens, 3, local D].

        Uneven whole-head splits support the five-worker / twelve-head layout.
        No attention head is split between ranks.
        """
        import torch.distributed as dist

        world, rank = self.world_size, self.rank_in_group
        batch, tokens, groups, dim = tensor.shape
        if num_heads < world or dim % num_heads:
            raise ValueError("head parallelism requires at least one whole head per rank")
        head_dim = dim // num_heads
        widths = [(num_heads // world + (i < num_heads % world)) * head_dim
                  for i in range(world)]
        send_sizes = [batch * tokens * groups * width for width in widths]
        send = torch.cat([part.contiguous().view(-1)
                          for part in tensor.split(widths, dim=-1)])
        receive = tensor.new_empty(world * send_sizes[rank])
        capturing = tensor.is_cuda and torch.cuda.is_current_stream_capturing()
        work = dist.all_to_all_single(receive, send,
                               output_split_sizes=[send_sizes[rank]] * world,
                               input_split_sizes=send_sizes, async_op=capturing)
        if capturing:
            work.synchronize()
        return receive.view(world, batch, tokens, groups, widths[rank]).permute(
            1, 0, 2, 3, 4).flatten(1, 2)

    @torch.compiler.disable
    def heads_to_sequence(self, tensor, *, num_heads: int, head_dim: int):
        """Inverse of sequence_to_heads for attention outputs [B, all tokens, local D]."""
        import torch.distributed as dist

        world = self.world_size
        batch, total_tokens, local_dim = tensor.shape
        tokens = total_tokens // world
        widths = [(num_heads // world + (i < num_heads % world)) * head_dim
                  for i in range(world)]
        send = tensor.reshape(batch, world, tokens, local_dim).permute(1, 0, 2, 3).contiguous().view(-1)
        receive_sizes = [batch * tokens * width for width in widths]
        receive = tensor.new_empty(sum(receive_sizes))
        capturing = tensor.is_cuda and torch.cuda.is_current_stream_capturing()
        work = dist.all_to_all_single(receive, send, output_split_sizes=receive_sizes,
                               input_split_sizes=[batch * tokens * local_dim] * world,
                               async_op=capturing)
        if capturing:
            work.synchronize()
        return torch.cat([part.view(batch, tokens, width)
                          for part, width in zip(receive.split(receive_sizes), widths)], dim=-1)

    def broadcast(self, tensor, *, src: int = 0):
        import torch.distributed as dist

        if self.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor

    def broadcast_object_list(self, values, *, src: int = 0):
        import torch.distributed as dist

        if self.world_size > 1:
            dist.broadcast_object_list(values, src=src)
        return values


_WORLD_GROUP = WorldSequenceParallelGroup()


def get_sequence_parallel_world_size() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_sequence_parallel_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_sp_group() -> WorldSequenceParallelGroup:
    return _WORLD_GROUP
