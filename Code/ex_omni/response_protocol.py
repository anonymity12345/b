"""Helpers for the Ex-Omni ``vtp + response`` protocol."""

import re
from typing import Optional, Sequence, Tuple

from .constants import (
    ASSISTANT_PROTOCOL_TOKENS,
    VTP_CLOSE_TOKEN,
    VTP_OPEN_TOKEN,
    RESPONSE_CLOSE_TOKEN,
    RESPONSE_OPEN_TOKEN,
)


def register_assistant_protocol_tokens(tokenizer) -> int:
    added = tokenizer.add_tokens(list(ASSISTANT_PROTOCOL_TOKENS), special_tokens=False)
    for token in ASSISTANT_PROTOCOL_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"assistant protocol token is not atomic: {token!r} -> {ids}")
    return int(added)


def response_token_span(token_ids: Sequence[int], tokenizer) -> Tuple[int, int]:
    ids = list(token_ids)
    open_id = tokenizer.convert_tokens_to_ids(RESPONSE_OPEN_TOKEN)
    close_id = tokenizer.convert_tokens_to_ids(RESPONSE_CLOSE_TOKEN)
    opens = [index for index, value in enumerate(ids) if value == open_id]
    closes = [index for index, value in enumerate(ids) if value == close_id]
    if len(opens) != 1 or len(closes) != 1 or opens[0] + 1 >= closes[0]:
        raise ValueError("assistant response must contain one non-empty response boundary pair")
    return opens[0] + 1, closes[0]


def parse_assistant_response(raw_text: str) -> Tuple[Optional[str], str]:
    text = (raw_text or "").strip()
    text = re.sub(r"^\s*<\|im_start\|>\s*assistant\s*", "", text, flags=re.I)
    text = re.sub(r"\s*<\|im_end\|>\s*$", "", text, flags=re.I)
    flags = re.I | re.S
    vtp_match = re.search(r"<avatar_plan\s*>(.*?)</avatar_plan\s*>", text, flags)
    response = re.search(r"<response\s*>(.*?)</response\s*>", text, flags)
    vtp = vtp_match.group(1).strip() if vtp_match else None
    if response:
        return vtp, response.group(1).strip()
    open_response = re.search(r"<response\s*>(.*)$", text, flags)
    if open_response:
        return vtp, open_response.group(1).strip()
    visible = re.sub(r"<think\s*>.*?</think\s*>", "", text, flags=flags)
    visible = re.sub(r"<avatar_plan\s*>.*?</avatar_plan\s*>", "", visible, flags=flags)
    return vtp, visible.strip()


def make_strict_logits_processor(tokenizer, max_new_tokens: int):
    """Build the finite-state logits processor without import-time torch use."""
    return StrictAssistantProtocolLogitsProcessor(tokenizer, max_new_tokens)


class StrictAssistantProtocolLogitsProcessor:
    """Finite-state constraint for ``plan`` followed by ``response``."""

    def __init__(self, tokenizer, max_new_tokens: int):
        self.vtp_open_id = tokenizer.convert_tokens_to_ids(VTP_OPEN_TOKEN)
        self.vtp_close_id = tokenizer.convert_tokens_to_ids(VTP_CLOSE_TOKEN)
        self.response_open_id = tokenizer.convert_tokens_to_ids(RESPONSE_OPEN_TOKEN)
        self.response_close_id = tokenizer.convert_tokens_to_ids(RESPONSE_CLOSE_TOKEN)
        eos = tokenizer.eos_token_id
        self.eos_ids = (
            [int(value) for value in eos]
            if isinstance(eos, (list, tuple))
            else [int(eos)]
        )
        if int(max_new_tokens) < 2:
            raise ValueError("max_new_tokens must be at least 2")
        self.half_budget = max(int(max_new_tokens) // 2, 1)
        self.structural_ids = {
            self.vtp_open_id,
            self.vtp_close_id,
            self.response_open_id,
            self.response_close_id,
            *self.eos_ids,
        }
        self.initial_length = None

    @staticmethod
    def force(scores, token_id):
        import torch

        scores.fill_(-torch.inf)
        scores[:, token_id] = 0
        return scores

    def __call__(self, input_ids, scores):
        import torch

        if self.initial_length is None:
            self.initial_length = input_ids.shape[1]
        ids = input_ids[0, self.initial_length :].tolist()
        if self.vtp_open_id not in ids:
            return self.force(scores, self.vtp_open_id)
        vtp_open = ids.index(self.vtp_open_id)
        if self.vtp_close_id not in ids[vtp_open + 1 :]:
            content_tokens = len(ids) - vtp_open - 1
            if content_tokens >= self.half_budget:
                return self.force(scores, self.vtp_close_id)
            banned = set(self.structural_ids)
            if content_tokens > 0:
                banned.remove(self.vtp_close_id)
            scores[:, list(banned)] = -torch.inf
            return scores
        vtp_close = ids.index(self.vtp_close_id, vtp_open + 1)
        if self.response_open_id not in ids[vtp_close + 1 :]:
            return self.force(scores, self.response_open_id)
        response_open = ids.index(self.response_open_id, vtp_close + 1)
        if self.response_close_id not in ids[response_open + 1 :]:
            content_tokens = len(ids) - response_open - 1
            if content_tokens >= self.half_budget:
                return self.force(scores, self.response_close_id)
            banned = set(self.structural_ids)
            if content_tokens > 0:
                banned.remove(self.response_close_id)
            scores[:, list(banned)] = -torch.inf
            return scores
        return self.force(scores, self.eos_ids[0])


class TensorParallelTokenSyncLogitsProcessor:
    """Sample on the TP leader and broadcast one identical token ID."""

    def __init__(self, group, src_rank: int, temperature: float, top_p: float):
        if group is None:
            raise ValueError("tensor parallel token sync requires a process group")
        self.group = group
        self.src_rank = int(src_rank)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def _sample(self, scores):
        import torch

        if self.temperature <= 0:
            return scores.argmax(dim=-1)
        logits = scores.float() / self.temperature
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > self.top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        sampled = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
        return sorted_indices.gather(-1, sampled).squeeze(-1)

    def __call__(self, input_ids, scores):
        del input_ids
        import torch
        import torch.distributed as dist

        if scores.shape[0] > 1:
            dist.broadcast(scores, src=self.src_rank, group=self.group)
            return scores
        if dist.get_rank() == self.src_rank:
            next_ids = self._sample(scores).to(device=scores.device, dtype=torch.long)
        else:
            next_ids = torch.empty(scores.shape[0], device=scores.device, dtype=torch.long)
        dist.broadcast(next_ids, src=self.src_rank, group=self.group)
        scores.fill_(-torch.inf)
        scores.scatter_(1, next_ids.unsqueeze(1), 0.0)
        return scores
