from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ex_omni.constants import IGNORE_INDEX
from ex_omni.model.speech_generator.qwen3_tts_speaker import Qwen3TTSSpeakerEmbedding
from ex_omni.model.speech_generator.qwen3_tts_talker import Qwen3TTSTalker

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Speech generator built on top of the Qwen3-TTS talker."""

from einops import rearrange
from torch import einsum
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class SpeechGeneratorOutputWithPast(CausalLMOutputWithPast):
    main_loss: Optional[torch.FloatTensor] = None
    residual_loss: Optional[torch.FloatTensor] = None

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class TQGFCrossAttention(nn.Module):
    def __init__(self, dim=896, dim_head=256, heads=16):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        
        self.norm_text = nn.LayerNorm(dim)
        self.norm_query = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Gated Attention: head-specific elementwise (论文 Table 1 row 5)
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, text_reps, query_reps, text_mask=None):
        if text_reps.shape[1] == 0:
            return torch.zeros_like(query_reps)

        text_reps_normed = self.norm_text(text_reps)
        query_reps_normed = self.norm_query(query_reps)
        
        h, d = self.heads, self.dim_head
        
        # QKV
        q = rearrange(self.to_q(query_reps_normed), "b n (h d) -> b h n d", h=h, d=d)
        k, v = self.to_kv(text_reps_normed).chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=h, d=d)
        v = rearrange(v, "b n (h d) -> b h n d", h=h, d=d)
        
        # Attention
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        
        if text_mask is not None:
            mask = rearrange(text_mask, "b n -> b 1 1 n")
            has_context = mask.any(dim=-1, keepdim=True)
            sim = sim.masked_fill(~mask, float('-inf'))
            sim = torch.where(has_context, sim, torch.zeros_like(sim))
        
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)  # [B, H, N, Dh]
        
        # Gated Attention (SDPA output, G1, head-specific elementwise)
        gate_scores = self.gate_proj(query_reps_normed)  # [B, N, H*Dh]
        gate_scores = rearrange(gate_scores, "b n (h d) -> b h n d", h=h, d=d)  # [B, H, N, Dh]
        gate_scores = torch.sigmoid(gate_scores)
        out = out * gate_scores  # Head-specific elementwise gating
        
        # Output projection
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    

class TQGF(nn.Module):
    """Text-Query Gated Fusion.

    Fuse Ex-Omni thinker hidden states with text token embeddings before the
    acoustic talker/decoder consumes them.
    """

    def __init__(
        self,
        dim,
        depth=2,
        dim_head=256,
        heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TQGFCrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_reps, query_reps, text_mask=None):
        b,n,_=text_reps.shape
        for attn, ff in self.layers:
            query_reps = attn(text_reps, query_reps, text_mask=text_mask) + query_reps
            query_reps = ff(query_reps) + query_reps
        return self.norm(query_reps)

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute loss between x and target.
        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor): target signal masked with self.padding_id (batch, seqlen)
            reduction (str): 'mean', 'sum', or 'none'
        Returns:
            loss (torch.Tensor): The KL loss
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.reshape(-1, self.size)
        target = target.reshape(-1)
        
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        
        # 重新reshape为 (batch_size, seq_len)
        kl = kl.sum(dim=1)  # sum over vocab dimension if needed
        kl = kl.view(batch_size, seq_len)
        ignore = ignore.view(batch_size, seq_len)
        
        # 对每个样本计算loss
        kl_masked = kl.masked_fill(ignore, 0)
        
        if reduction == 'none':
            # 返回每个样本的loss
            if self.normalize_length:
                # 每个样本除以其有效长度
                valid_lengths = (~ignore).sum(dim=1).float()
                valid_lengths = torch.clamp(valid_lengths, min=1)  # 避免除零
                return kl_masked.sum(dim=1) / valid_lengths
            else:
                return kl_masked.sum(dim=1)
        elif reduction == 'sum':
            return kl_masked.sum()
        else:  # reduction == 'mean'
            denom = total if self.normalize_length else batch_size
            return kl_masked.sum() / denom

def _speech_generation_limits(
    text_len,
    prefix_len,
    *,
    min_token_text_ratio,
    max_token_text_ratio,
    max_speech_tokens,
):
    min_total = int(text_len) * int(min_token_text_ratio)
    max_total = int(text_len) * int(max_token_text_ratio)
    if int(max_speech_tokens) <= 0:
        raise ValueError("max_speech_tokens must be positive")
    max_total = min(max_total, int(max_speech_tokens))
    min_total = min(min_total, max_total)
    return min_total, max(0, max_total - int(prefix_len))

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

def _uniform_assignment(src_lens, tgt_lens):
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t

def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class SpeechGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        qwen_tts_path = getattr(config, "pretrain_qwen_tts_weights", None)
        if not qwen_tts_path or str(qwen_tts_path).lower() == "none":
            qwen_tts_path = getattr(config, "pretrain_speaker_encoder_weights", None)
        self.talker = Qwen3TTSTalker(
            qwen_tts_path,
            attn_implementation=getattr(
                config,
                "speech_attn_implementation",
                getattr(
                    config,
                    "_attn_implementation",
                    getattr(config, "attn_implementation", "flash_attention_2"),
                ),
            ),
            decode_attn_implementation=getattr(
                config,
                "speech_decode_attn_implementation",
                getattr(config, "decode_attn_implementation", None),
            ),
        )

        config.speech_gen_hidden_size = self.talker.hidden_size
        self.unit_vocab_size = int(self.talker.config.code_predictor_config.vocab_size)
        self.main_eos_token_id = self.talker.codec_eos_token_id
        self.num_code_groups = int(getattr(config, "num_code_groups", 16))
        self.code_predictor_loss_weight = float(getattr(config, "code_predictor_loss_weight", 1.0))
        if self.num_code_groups < 2:
            raise ValueError("SpeechGenerator now expects multi-codebook audio tokens. Set num_code_groups >= 2.")
        if self.num_code_groups != self.talker.num_code_groups:
            raise ValueError(f"num_code_groups={self.num_code_groups} does not match Qwen3-TTS talker groups={self.talker.num_code_groups}")
        self.n_dims = self.talker.hidden_size
        self.llm_input_size = self.n_dims
        self.llm_output_size = self.n_dims
        self.criterion_ce = LabelSmoothingLoss(
            size=self.talker.vocab_size,
            padding_idx=IGNORE_INDEX,
            smoothing=0,
            normalize_length=True,
        )

        speaker_encoder_path = getattr(config, "pretrain_speaker_encoder_weights", None) or qwen_tts_path
        self.roleplay_speaker_encoder = (
            Qwen3TTSSpeakerEmbedding(speaker_encoder_path)
            if speaker_encoder_path and str(speaker_encoder_path).lower() != "none"
            else None
        )
        speaker_embedding_size = (
            self.roleplay_speaker_encoder.embedding_dim
            if self.roleplay_speaker_encoder is not None
            else int(getattr(config, "speaker_encoder_hidden_size", 2048))
        )
        self.roleplay_speaker_proj = (
            nn.Identity()
            if speaker_embedding_size == self.llm_input_size
            else nn.Linear(speaker_embedding_size, self.llm_input_size)
        )

        modules = [
            nn.Linear(config.hidden_size, self.n_dims * 4),
            nn.GELU(),
            nn.Linear(self.n_dims * 4, self.n_dims * 4),
        ]
        
        self.input_proj = nn.Sequential(*modules)
        self.fusion = TQGF(self.n_dims)
        self.freeze_external_encoders()

    def freeze_external_encoders(self):
        if self.roleplay_speaker_encoder is not None:
            self.roleplay_speaker_encoder.eval()
            for param in self.roleplay_speaker_encoder.parameters():
                param.requires_grad = False

    def _prepare_speech_token_batch(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        if speech_tokens.dim() != 3:
            raise ValueError(f"speech_tokens should be multi-codebook [B, T, G], got {tuple(speech_tokens.shape)}")
        if speech_tokens.shape[-1] < 2:
            raise ValueError(f"speech_tokens should contain multiple codebooks, got {speech_tokens.shape[-1]}")
        return speech_tokens

    def _speech_frame_mask(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        return speech_tokens.ne(IGNORE_INDEX).any(dim=-1)

    def _embed_codebook_frames(self, speech_codes: torch.Tensor) -> torch.Tensor:
        speech_codes = speech_codes.masked_fill(speech_codes.eq(IGNORE_INDEX), 0)
        return self.talker.embed_codec_frames(speech_codes)

    def _embed_single_codebook(self, codebook_index: int, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.masked_fill(tokens.eq(IGNORE_INDEX), 0)
        if codebook_index == 0:
            return self.talker.embed_first_code(tokens)
        return self.talker.code_predictor.get_input_embeddings()[codebook_index - 1](
            tokens.clamp(0, self.unit_vocab_size - 1).long()
        )

    def _format_output_tokens(self, out_tokens):
        return " ".join([",".join(str(v) for v in frame) for frame in out_tokens])

    def _first_codebook_tensor(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        if speech_tokens.dim() == 3:
            return speech_tokens[..., 0]
        return speech_tokens

    def prepare_roleplay_embedding(
        self,
        ref_audio_waveform=None,
        ref_audio_waveform_lengths=None,
        has_ref_audio=None,
    ):
        if ref_audio_waveform is None:
            return None

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        ref_audio_waveform = ref_audio_waveform.to(device=device)
        if ref_audio_waveform_lengths is not None:
            ref_audio_waveform_lengths = ref_audio_waveform_lengths.to(device=device)
        batch_size = ref_audio_waveform.shape[0]

        if has_ref_audio is None:
            has_ref_audio = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_ref_audio = has_ref_audio.to(device=device).bool()

        if not has_ref_audio.any().item():
            return None
        if self.roleplay_speaker_encoder is None:
            raise ValueError("pretrain_speaker_encoder_weights is required when role-play ref audio is provided.")

        self.roleplay_speaker_encoder.to(device=device)
        speaker_embedding = self.roleplay_speaker_encoder(
            ref_audio_waveform,
            ref_audio_waveform_lengths,
        ).to(device=device, dtype=dtype)
        speaker_embedding = self.roleplay_speaker_proj(speaker_embedding)
        embeddings = []

        for batch_idx in range(batch_size):
            if has_ref_audio[batch_idx].item():
                embeddings.append(speaker_embedding[batch_idx:batch_idx + 1])
            else:
                embeddings.append(torch.zeros(0, self.llm_input_size, device=device, dtype=dtype))

        max_len = max(x.shape[0] for x in embeddings)
        if max_len == 0:
            return None

        padded = torch.zeros(batch_size, max_len, self.llm_input_size, device=device, dtype=dtype)
        for batch_idx, cur_embedding in enumerate(embeddings):
            if cur_embedding.shape[0] > 0:
                padded[batch_idx, :cur_embedding.shape[0]] = cur_embedding
        return padded

    def _predict_residual_codebooks(
        self,
        main_hidden,
        first_code_id,
        first_code_embed,
        *,
        do_sample,
        top_k,
        top_p,
        temperature,
        use_kv_cache,
        sampling_generator=None,
    ):
        first_code = torch.as_tensor(
            first_code_id,
            dtype=torch.long,
            device=main_hidden.device,
        ).reshape(1)
        frame_codes = [first_code]
        prefix = [main_hidden, first_code_embed]
        predictor_cache = None
        next_predictor_input = None
        for codebook_index in range(1, self.num_code_groups):
            if use_kv_cache:
                predictor_input = (
                    torch.cat(prefix, dim=1)
                    if predictor_cache is None
                    else next_predictor_input
                )
                logits, predictor_cache = (
                    self.talker.residual_logits_for_prefix(
                        predictor_input,
                        codebook_index,
                        past_key_values=predictor_cache,
                        use_cache=True,
                    )
                )
            else:
                logits = self.talker.residual_logits_for_prefix(
                    torch.cat(prefix, dim=1),
                    codebook_index,
                )
            logp = logits.log_softmax(dim=-1)
            next_id = self.sampling_ids(
                logp.squeeze(dim=0),
                [],
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=1.0,
                ignore_eos=True,
                main_codebook=False,
                repetition_aware=False,
                sampling_generator=sampling_generator,
            )
            frame_codes.append(next_id)
            next_embed = self._embed_single_codebook(
                codebook_index,
                next_id.view(1, 1),
            )
            if use_kv_cache:
                next_predictor_input = next_embed
            else:
                prefix.append(next_embed)

        # One device synchronization per RVQ frame, instead of one for each
        # of the 15 residual codebooks. Residual sampling uses neither
        # repetition penalties nor repetition-aware resampling, so retaining
        # the intermediate IDs on device is mathematically equivalent.
        return torch.cat(frame_codes).detach().cpu().tolist()


    def _sample_scores(
        self,
        scores,
        decoded_tokens,
        *,
        do_sample,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        repetition_aware,
        sampling_generator=None,
    ):
        scores = scores.float().clone()
        if repetition_penalty <= 0:
            raise ValueError("speech repetition_penalty must be positive")
        if repetition_penalty != 1.0 and decoded_tokens:
            history = torch.as_tensor(
                sorted(set(int(token) for token in decoded_tokens)),
                dtype=torch.long,
                device=scores.device,
            )
            history = history[(history >= 0) & (history < scores.shape[-1])]
            if history.numel():
                values = scores[history]
                scores[history] = torch.where(
                    values < 0,
                    values * repetition_penalty,
                    values / repetition_penalty,
                )
        if not do_sample:
            return scores.argmax(dim=-1).reshape(1)
        if temperature <= 0:
            raise ValueError("speech sampling temperature must be positive")
        scores = scores / float(temperature)
        unfiltered_scores = scores.clone()
        if int(top_k) > 0 and int(top_k) < scores.numel():
            cutoff = torch.topk(scores, int(top_k)).values[-1]
            scores = scores.masked_fill(scores < cutoff, -torch.inf)
        if 0 < float(top_p) < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
            remove = (
                sorted_probabilities.cumsum(dim=-1)
                - sorted_probabilities
                > float(top_p)
            )
            sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
            filtered = torch.full_like(scores, -torch.inf)
            filtered.scatter_(0, sorted_indices, sorted_scores)
            scores = filtered
        probabilities = torch.softmax(scores, dim=-1)
        candidate = torch.multinomial(
            probabilities,
            1,
            generator=sampling_generator,
        )
        if repetition_aware and decoded_tokens:
            window = decoded_tokens[-10:]
            repetitions = sum(int(token) == int(candidate.item()) for token in window)
            if repetitions / 10.0 > 0.1:
                candidate = torch.multinomial(
                    torch.softmax(unfiltered_scores, dim=-1),
                    1,
                    generator=sampling_generator,
                )
        return candidate

    def sampling_ids(
        self,
        weighted_scores,
        decoded_tokens,
        *,
        do_sample,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        ignore_eos,
        main_codebook,
        repetition_aware,
        sampling_generator=None,
    ):
        tp_group = getattr(self, "distributed_tp_group", None)
        tp_src_rank = getattr(self, "distributed_tp_src_rank", None)
        synchronized_tp = (
            tp_group is not None
            and tp_src_rank is not None
            and dist.is_available()
            and dist.is_initialized()
        )

        # Sampling independently on tensor-parallel ranks is unsafe: tiny
        # numerical differences can make one rank emit EOS while peers enter
        # the next TP collective, causing a permanent deadlock. Only the TP
        # leader samples; every codebook token is then broadcast to its peers.
        if not synchronized_tp or dist.get_rank() == int(tp_src_rank):
            scores = weighted_scores.float().clone()
            eos_token_id = int(
                getattr(self, "main_eos_token_id", self.unit_vocab_size)
            )
            if main_codebook:
                original_eos = (
                    scores[eos_token_id].clone()
                    if 0 <= eos_token_id < scores.shape[-1]
                    else None
                )
                scores[self.unit_vocab_size :] = -torch.inf
                if not ignore_eos and original_eos is not None:
                    scores[eos_token_id] = original_eos
            elif ignore_eos and 0 <= eos_token_id < scores.shape[-1]:
                scores[eos_token_id] = -torch.inf
            top_ids = self._sample_scores(
                scores,
                decoded_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                repetition_aware=repetition_aware,
                sampling_generator=sampling_generator,
            )
            top_ids = top_ids.reshape(1).to(
                device=weighted_scores.device,
                dtype=torch.long,
            )
        else:
            top_ids = torch.empty(1, dtype=torch.long, device=weighted_scores.device)

        if synchronized_tp:
            dist.broadcast(
                top_ids,
                src=int(tp_src_rank),
                group=tp_group,
            )
        return top_ids

    def forward(self, text_reps, text_labels, speech_tokens, text_tokens=None, embedding=None):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        text_reps = text_reps.to(device=device, dtype=dtype)
        text_labels = text_labels.to(device=device)
        speech_tokens = speech_tokens.to(device=device)
        speech_tokens = self._prepare_speech_token_batch(speech_tokens)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device=device)
        
        # 处理text representations
        tgt_text_reps = [text_rep[text_label.ne(IGNORE_INDEX)] for text_rep, text_label in zip(text_reps, text_labels)]
        text_reps_lens = torch.LongTensor([len(rep)*4 for rep in tgt_text_reps]).to(device)

        tgt_text_reps_padding_mask = ~lengths_to_padding_mask(text_reps_lens)
        
        tgt_text_reps = torch.nn.utils.rnn.pad_sequence(tgt_text_reps, batch_first=True)
        tgt_text_reps = self.input_proj(tgt_text_reps)
        tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        
        # 计算token长度
        text_token_lens = text_tokens.ne(IGNORE_INDEX).long().sum(dim=-1)
        speech_frame_mask = self._speech_frame_mask(speech_tokens)
        speech_token_lens = speech_frame_mask.long().sum(dim=-1)
        
        # 预处理有效tokens
        valid_text_tokens = [x[x.ne(IGNORE_INDEX)] for x in text_tokens]
        valid_speech_tokens = [x[mask] for x, mask in zip(speech_tokens, speech_frame_mask)]
        
        # 获取 embeddings: Qwen3-TTS text embeddings -> text projection, then fuse with Ex-Omni thinker hidden.
        text = torch.nn.utils.rnn.pad_sequence(valid_text_tokens, batch_first=True, padding_value=0)
        text_embeddings = self.talker.get_text_embeddings()(text)
        text_embeddings = self.talker.text_projection(text_embeddings)
        text_embeddings = self.fusion(tgt_text_reps, text_embeddings, tgt_text_reps_padding_mask)
        
        speech = torch.nn.utils.rnn.pad_sequence(valid_speech_tokens, batch_first=True, padding_value=IGNORE_INDEX)
        speech_embeddings = self._embed_codebook_frames(speech)
        
        codec_bos_emb = self.talker.get_input_embeddings()(
            torch.tensor([self.talker.codec_bos_id], dtype=torch.long, device=device)
        ).reshape(-1)
        codec_eos_emb = self.talker.get_input_embeddings()(
            torch.tensor([self.talker.codec_eos_token_id], dtype=torch.long, device=device)
        ).reshape(-1)
        
        # 构建输入和标签
        lm_inputs = []
        lm_targets = []
        speech_label_starts = []
        
        for i, (text_len, speech_len) in enumerate(zip(text_token_lens, speech_token_lens)):
            text_len = int(text_len.item())
            speech_len = int(speech_len.item())
            if embedding is not None:
                roleplay_embedding = embedding[i]
                roleplay_embedding = roleplay_embedding[roleplay_embedding.abs().sum(dim=-1).ne(0)]
            else:
                roleplay_embedding = text_embeddings[i][:0]
            input_seq = torch.cat([
                roleplay_embedding,
                text_embeddings[i][:text_len],
                codec_bos_emb.unsqueeze(0),
                speech_embeddings[i][:speech_len],
                codec_eos_emb.unsqueeze(0),
            ])
            lm_inputs.append(input_seq)
            
            roleplay_len = roleplay_embedding.shape[0]
            prefix_len = roleplay_len + text_len + 1
            speech_label_starts.append(prefix_len)
            target_seq = torch.tensor(
                [IGNORE_INDEX] * prefix_len +
                valid_speech_tokens[i][:speech_len, 0].tolist() + 
                [self.talker.codec_eos_token_id],
                dtype=torch.long,
                device=speech_tokens.device
            )
            lm_targets.append(target_seq)
        
        # 批量填充
        max_len = max(x.shape[0] for x in lm_inputs)
        batch_size = len(lm_inputs)
        embed_dim = lm_inputs[0].shape[1]
        
        # 一次性创建所有张量
        input_embeds = torch.zeros((batch_size, max_len, embed_dim), 
                                dtype=lm_inputs[0].dtype, device=lm_inputs[0].device)
        labels = torch.full((batch_size, max_len), IGNORE_INDEX, 
                        dtype=torch.long, device=speech_tokens.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=speech_tokens.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=speech_tokens.device)
        
        # 填充数据
        for i, (input_seq, target_seq) in enumerate(zip(lm_inputs, lm_targets)):
            seq_len = input_seq.shape[0]
            input_embeds[i, :seq_len] = input_seq
            labels[i, :seq_len] = target_seq
            attention_mask[i, :seq_len] = True
            position_ids[i, :seq_len] = torch.arange(seq_len, device=speech_tokens.device)
        
        last_hidden, logits = self.talker.forward_main(
            input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        main_loss = self.criterion_ce(logits[:, :-1], labels[:, 1:], 'none')
        residual_loss = torch.zeros_like(main_loss)
        if self.num_code_groups > 1 and speech.shape[-1] > 1:
            max_speech_len = speech.shape[1]
            residual_hidden = torch.zeros(
                (batch_size, max_speech_len, last_hidden.shape[-1]),
                dtype=last_hidden.dtype,
                device=last_hidden.device,
            )
            for i, (label_start, speech_len) in enumerate(zip(speech_label_starts, speech_token_lens)):
                speech_len = int(speech_len.item())
                if speech_len > 0:
                    pred_start = int(label_start.item() if torch.is_tensor(label_start) else label_start) - 1
                    residual_hidden[i, :speech_len] = last_hidden[i, pred_start:pred_start + speech_len]

            flat_hidden = []
            flat_codec = []
            flat_owner = []
            for batch_idx, speech_len in enumerate(speech_token_lens):
                speech_len = int(speech_len.item())
                if speech_len > 0:
                    flat_hidden.append(residual_hidden[batch_idx, :speech_len])
                    flat_codec.append(speech[batch_idx, :speech_len])
                    flat_owner.append(torch.full((speech_len,), batch_idx, dtype=torch.long, device=device))
            if flat_hidden:
                flat_hidden = torch.cat(flat_hidden, dim=0)
                flat_codec = torch.cat(flat_codec, dim=0)
                flat_owner = torch.cat(flat_owner, dim=0)
                clean_codec = flat_codec.masked_fill(flat_codec.eq(IGNORE_INDEX), 0).clamp(0, self.unit_vocab_size - 1).long()
                residual_logits = self.talker.residual_logits(flat_hidden, clean_codec)
                residual_targets = flat_codec[:, 1:self.num_code_groups]
                token_loss = F.cross_entropy(
                    residual_logits.reshape(-1, self.unit_vocab_size).float(),
                    residual_targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).view(flat_codec.shape[0], -1)
                valid = residual_targets.ne(IGNORE_INDEX)
                sample_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)
                sample_count = torch.zeros(batch_size, device=device, dtype=torch.float32)
                sample_loss.scatter_add_(0, flat_owner, token_loss.sum(dim=1))
                sample_count.scatter_add_(0, flat_owner, valid.sum(dim=1).float())
                residual_loss = (sample_loss / sample_count.clamp(min=1.0)).to(main_loss.dtype)
        loss = main_loss + self.code_predictor_loss_weight * residual_loss
        
        return SpeechGeneratorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=(last_hidden,),
            attentions=None,
            main_loss=main_loss,
            residual_loss=residual_loss,
        )

    def predict(self, 
            tgt_text_reps, 
            text,
            embedding=None,
            max_token_text_ratio=20,
            min_token_text_ratio=2,
            max_speech_tokens=750,
            prefix_speech_tokens=None,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            return_hidden_states=True,  # 新增参数
            stop_event=None,
            unit_chunk_size=0,
            unit_chunk_callback=None,
            use_kv_cache=True,
            residual_use_kv_cache=None,
            sampling_generator=None,
            **kwargs):

        if residual_use_kv_cache is None:
            residual_use_kv_cache = use_kv_cache

        # 1. 处理文本表示
        tgt_text_reps = self.input_proj(tgt_text_reps)
        tgt_text_reps = rearrange(tgt_text_reps, 'b n (d1 d2) -> b (n d2) d1', d2=4)
        
        # drop the eos token
        text = text[:, :-1]
        text_len = text.size(1)      
        text_embedding = self.talker.get_text_embeddings()(text)
        text_embedding = self.talker.text_projection(text_embedding)
        text_embedding = self.fusion(tgt_text_reps, text_embedding, None)

        # 2. encode role-play reference embedding
        if embedding is None:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text_embedding.dtype, device=text_embedding.device)
        else:
            embedding = embedding.to(device=text_embedding.device, dtype=text_embedding.dtype)
            embedding = embedding[:, embedding.abs().sum(dim=-1).ne(0).squeeze(0)] if embedding.shape[0] == 1 else embedding

        # 3. concat llm_input (这就是初始prompt)
        codec_bos_emb = self.talker.get_input_embeddings()(
            torch.tensor([self.talker.codec_bos_id], dtype=torch.long, device=text.device)
        ).reshape(1, 1, -1)

        context_input = torch.concat([embedding, text_embedding, codec_bos_emb], dim=1)
        prefix_main_tokens = []
        prefix_len = 0
        if prefix_speech_tokens is not None:
            prefix_speech_tokens = torch.as_tensor(
                prefix_speech_tokens,
                dtype=torch.long,
                device=text.device,
            )
            if prefix_speech_tokens.dim() == 2:
                prefix_speech_tokens = prefix_speech_tokens.unsqueeze(0)
            prefix_speech_tokens = self._prepare_speech_token_batch(
                prefix_speech_tokens
            )
            if prefix_speech_tokens.shape[0] != 1:
                raise ValueError("speech prefix requires batch_size=1")
            prefix_len = int(prefix_speech_tokens.shape[1])
            if prefix_len:
                context_input = torch.cat(
                    [
                        context_input,
                        self._embed_codebook_frames(prefix_speech_tokens),
                    ],
                    dim=1,
                )
                prefix_main_tokens = (
                    prefix_speech_tokens[0, :, 0].detach().cpu().tolist()
                )

        # 4. cal min/max_length
        min_total_len, max_new_len = _speech_generation_limits(
            text_len,
            prefix_len,
            min_token_text_ratio=min_token_text_ratio,
            max_token_text_ratio=max_token_text_ratio,
            max_speech_tokens=max_speech_tokens,
        )

        # 5. step by step decode
        out_tokens = []
        all_hidden_states = []  # 收集所有步骤的hidden states
        generated_embeds = []
        emitted_tokens = 0
        past_key_values = None
        if use_kv_cache:
            y_pred, logits, past_key_values = self.talker.forward_main(
                context_input,
                use_cache=True,
            )

        def emit_ready_tokens(*, final=False):
            nonlocal emitted_tokens
            if unit_chunk_callback is None:
                return
            chunk_size = int(unit_chunk_size)
            if chunk_size <= 0:
                raise ValueError(
                    "unit_chunk_size must be positive when "
                    "unit_chunk_callback is provided"
                )
            while len(out_tokens) - emitted_tokens >= chunk_size:
                right = emitted_tokens + chunk_size
                unit_chunk_callback(out_tokens[emitted_tokens:right])
                emitted_tokens = right
            if final and emitted_tokens < len(out_tokens):
                unit_chunk_callback(out_tokens[emitted_tokens:])
                emitted_tokens = len(out_tokens)
        
        for i in range(max_new_len):
            if stop_event is not None and stop_event.is_set():
                break
            if not use_kv_cache:
                lm_input = (
                    torch.cat([context_input] + generated_embeds, dim=1)
                    if generated_embeds
                    else context_input
                )
                y_pred, logits = self.talker.forward_main(lm_input)
            y_pred = y_pred[:, -1:, :]
            
            # 保存hidden states
            if return_hidden_states:
                all_hidden_states.append(y_pred)
            
            logp = logits[:, -1].log_softmax(dim=-1)
            decoded_main_tokens = prefix_main_tokens + [
                frame[0] if isinstance(frame, (list, tuple)) else frame
                for frame in out_tokens
            ]
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0), 
                decoded_main_tokens, 
                do_sample=bool(do_sample),
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                ignore_eos=prefix_len + i < min_total_len,
                main_codebook=True,
                repetition_aware=True,
                sampling_generator=sampling_generator,
            ).item()
            
            if top_ids == self.talker.codec_eos_token_id:
                break
            if top_ids >= self.unit_vocab_size:
                continue

            first_code_embed = self._embed_single_codebook(
                0,
                torch.tensor([[top_ids]], dtype=torch.long, device=text.device),
            )
            frame_codes = self._predict_residual_codebooks(
                y_pred[:, -1:, :],
                top_ids,
                first_code_embed,
                do_sample=bool(do_sample),
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                use_kv_cache=bool(residual_use_kv_cache),
                sampling_generator=sampling_generator,
            )

            out_tokens.append(frame_codes)
            generated_embed = self._embed_codebook_frames(
                torch.tensor(
                    frame_codes,
                    dtype=torch.long,
                    device=text.device,
                ).view(1, 1, -1)
            )
            generated_embeds.append(generated_embed)
            emit_ready_tokens()
            if use_kv_cache and i + 1 < max_new_len:
                y_pred, logits, past_key_values = self.talker.forward_main(
                    generated_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        emit_ready_tokens(final=True)
        if return_hidden_states and len(all_hidden_states) > 0:
            hidden_states = torch.cat(all_hidden_states, dim=1)
            
            if hidden_states.device != tgt_text_reps.device:
                hidden_states = hidden_states.to(device=tgt_text_reps.device,dtype=tgt_text_reps.dtype)
            
            return out_tokens, hidden_states
        else:
            return self._format_output_tokens(out_tokens), out_tokens
