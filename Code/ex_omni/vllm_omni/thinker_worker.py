"""Prompt-embedding bridge for the autoregressive worker.

The request decoder receives serialized prompt embeddings.
"""
from types import MethodType
import torch
from vllm_omni.worker.gpu_ar_worker import GPUARWorker


def prepare_prompt_embeddings(runner, scheduled_tokens):
    for req in runner.requests.values():
        if getattr(req, 'prompt_embeds_cpu', None) is None:
            embeds = getattr(req, 'prompt_embeds', None)
            if isinstance(embeds, torch.Tensor):
                req.prompt_embeds_cpu = embeds.detach().cpu().contiguous()
    return runner._ex_omni_original_prefill_collector(scheduled_tokens)


class ExOmniGPUARWorker(GPUARWorker):
    def init_device(self):
        super().init_device()
        runner = self.model_runner
        runner._ex_omni_original_prefill_collector = runner._collect_additional_information_for_prefill
        runner._collect_additional_information_for_prefill = MethodType(prepare_prompt_embeddings, runner)
