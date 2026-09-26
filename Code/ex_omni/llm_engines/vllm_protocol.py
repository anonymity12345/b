"""vLLM V1 implementation of the strict VTP/response protocol."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class StrictAssistantProtocolV1LogitsProcessor(LogitsProcessor):
    """Force ``<avatar_plan>...<response>...`` in vLLM's persistent batch."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        del is_pin_memory
        model_config = vllm_config.model_config
        tokenizer_path = model_config.tokenizer or model_config.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=False,
            use_fast=False,
        )
        self.device = device
        self.vtp_open_id = tokenizer.convert_tokens_to_ids("<avatar_plan>")
        self.vtp_close_id = tokenizer.convert_tokens_to_ids("</avatar_plan>")
        self.response_open_id = tokenizer.convert_tokens_to_ids("<response>")
        self.response_close_id = tokenizer.convert_tokens_to_ids("</response>")
        eos = tokenizer.eos_token_id
        self.eos_ids = (
            [int(value) for value in eos]
            if isinstance(eos, (list, tuple))
            else [int(eos)]
        )
        ids = [
            self.vtp_open_id,
            self.vtp_close_id,
            self.response_open_id,
            self.response_close_id,
            *self.eos_ids,
        ]
        if any(value is None or int(value) < 0 for value in ids):
            raise ValueError("decoder tokenizer is missing Ex-Omni protocol tokens")
        self.structural_ids = {int(value) for value in ids}
        self.requests: dict[int, tuple[list[int], int]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def _request_state(
        params: SamplingParams,
        output_token_ids: list[int],
    ) -> tuple[list[int], int]:
        protocol_budget = max(2, int(params.max_tokens) - 5)
        return output_token_ids, max(protocol_budget // 2, 1)

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, _, output_token_ids in batch_update.added:
            self.requests[index] = self._request_state(params, output_token_ids)
        for source, target, direction in batch_update.moved:
            source_value = self.requests.get(source)
            target_value = self.requests.get(target)
            if source_value is not None:
                self.requests[target] = source_value
            else:
                self.requests.pop(target, None)
            if direction == MoveDirectionality.SWAP:
                if target_value is not None:
                    self.requests[source] = target_value
                else:
                    self.requests.pop(source, None)
            else:
                self.requests.pop(source, None)

    @staticmethod
    def _force(row: torch.Tensor, token_id: int) -> None:
        row.fill_(-torch.inf)
        row[token_id] = 0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, (ids, half_budget) in self.requests.items():
            if index >= logits.shape[0]:
                continue
            row = logits[index]
            if self.vtp_open_id not in ids:
                self._force(row, self.vtp_open_id)
                continue
            vtp_open = ids.index(self.vtp_open_id)
            if self.vtp_close_id not in ids[vtp_open + 1 :]:
                content_tokens = len(ids) - vtp_open - 1
                if content_tokens >= half_budget:
                    self._force(row, self.vtp_close_id)
                    continue
                banned = set(self.structural_ids)
                if content_tokens > 0:
                    banned.remove(self.vtp_close_id)
                row[list(banned)] = -torch.inf
                continue
            vtp_close = ids.index(self.vtp_close_id, vtp_open + 1)
            if self.response_open_id not in ids[vtp_close + 1 :]:
                self._force(row, self.response_open_id)
                continue
            response_open = ids.index(self.response_open_id, vtp_close + 1)
            if self.response_close_id not in ids[response_open + 1 :]:
                content_tokens = len(ids) - response_open - 1
                if content_tokens >= half_budget:
                    self._force(row, self.response_close_id)
                    continue
                banned = set(self.structural_ids)
                if content_tokens > 0:
                    banned.remove(self.response_close_id)
                row[list(banned)] = -torch.inf
                continue
            self._force(row, self.eos_ids[0])
        return logits

