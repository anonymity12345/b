"""Ex-Omni Qwen3 decoder adapter for vLLM-Omni hidden-state streaming."""

from __future__ import annotations

import torch
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.output_templates import OmniOutput


MODEL_ARCH = "ExOmniQwen3ForCausalLM"


class ExOmniQwen3ForCausalLM(Qwen3ForCausalLM):
    """Stock vLLM Qwen3 decoder with vLLM-Omni latent output."""

    have_multimodal_outputs = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> OmniOutput | IntermediateTensors:
        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={},
        )

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return super().compute_logits(hidden_states)


def register_vllm_omni_llm_model() -> None:
    """Register lazily with both vLLM and vLLM-Omni registries."""
    from vllm.model_executor.models import ModelRegistry
    from vllm_omni.model_executor.models import OmniModelRegistry

    model_ref = (
        "ex_omni.vllm_omni.llm_model:"
        "ExOmniQwen3ForCausalLM"
    )
    for registry in (ModelRegistry, OmniModelRegistry):
        if MODEL_ARCH not in registry.get_supported_archs():
            registry.register_model(MODEL_ARCH, model_ref)
