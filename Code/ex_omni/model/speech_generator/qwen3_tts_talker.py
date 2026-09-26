import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from ex_omni.model.attention import (
    FLASH_ATTENTION_3,
    FLASH_ATTENTION_2,
    SDPA,
    flash_attention_module,
    resolve_attn_implementation,
    resolve_decode_attn_implementation,
)


@torch.compiler.disable
def _flash_attention_eager(
    query_states, key_states, value_states, *,
    implementation, decode_implementation, training, dropout_p, scaling,
):
    """Run the external FA kernel outside Dynamo while compiling the surrounding Talker."""
    flash_module = flash_attention_module(implementation)
    if (query_states.shape[1] == 1 and not training and
            decode_implementation in {FLASH_ATTENTION_3, FLASH_ATTENTION_2}):
        decode_module = flash_attention_module(decode_implementation)
        flash_attn_with_kvcache = getattr(decode_module, "flash_attn_with_kvcache", None)
        if flash_attn_with_kvcache is not None:
            return flash_attn_with_kvcache(
                query_states, key_states, value_states,
                cache_seqlens=int(key_states.shape[1]),
                softmax_scale=scaling, causal=True,
            )
    flash_kwargs = {"softmax_scale": scaling, "causal": True}
    if implementation == FLASH_ATTENTION_2:
        flash_kwargs["dropout_p"] = dropout_p if training else 0.0
    return flash_module.flash_attn_func(query_states, key_states, value_states, **flash_kwargs)


def _to_namespace(values):
    if isinstance(values, SimpleNamespace):
        return values
    if isinstance(values, dict):
        return SimpleNamespace(**{key: _to_namespace(value) if isinstance(value, dict) else value for key, value in values.items()})
    return values


def _activation(name):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


def _resolve_safetensors(model_path: str) -> str:
    if not model_path or str(model_path).lower() == "none":
        raise ValueError("pretrain_qwen_tts_weights must point to Qwen3-TTS model weights.")
    model_path = os.path.expanduser(model_path)
    candidates = [model_path] if os.path.isfile(model_path) else [os.path.join(model_path, "model.safetensors")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find Qwen3-TTS model.safetensors under {model_path}")


def _normalize_attn_implementation(value):
    return resolve_attn_implementation(value)


def _normalize_decode_attn_implementation(value):
    return resolve_decode_attn_implementation(value)


def _causal_mask(
    batch_size,
    query_len,
    device,
    dtype,
    *,
    key_len=None,
    past_len=0,
):
    key_len = int(key_len if key_len is not None else query_len)
    query_positions = torch.arange(
        int(past_len),
        int(past_len) + int(query_len),
        device=device,
    ).view(-1, 1)
    key_positions = torch.arange(key_len, device=device).view(1, -1)
    blocked = key_positions > query_positions
    mask = torch.zeros(
        query_len,
        key_len,
        device=device,
        dtype=dtype,
    ).masked_fill(blocked, torch.finfo(dtype).min)
    return mask.view(1, 1, query_len, key_len).expand(
        batch_size,
        1,
        query_len,
        key_len,
    )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False):
    if mrope_interleaved:
        def apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
        sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(1)
    else:
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Qwen3TTSRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3TTSRotaryEmbedding(nn.Module):
    def __init__(self, config, multimodal=False):
        super().__init__()
        self.multimodal = multimodal
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rope_theta = float(getattr(config, "rope_theta", 1000000.0))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        if self.multimodal:
            if position_ids.dim() == 2:
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            pos = position_ids[:, :, None, :].float()
            freqs = (inv_freq.to(x.device) @ pos.to(x.device)).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Qwen3TTSMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _activation(config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TTSAttention(nn.Module):
    def __init__(self, config, layer_idx, multimodal=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.multimodal = multimodal
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.attn_implementation = _normalize_attn_implementation(
            getattr(config, "_attn_implementation", getattr(config, "attn_implementation", FLASH_ATTENTION_2))
        )
        self.decode_attn_implementation = (
            _normalize_decode_attn_implementation(
                getattr(config, "decode_attn_implementation", None)
            )
        )
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        *,
        past_key_value=None,
        use_cache=False,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        if self.multimodal:
            rope_scaling = getattr(self.config, "rope_scaling", None) or {}
            if isinstance(rope_scaling, SimpleNamespace):
                rope_scaling = vars(rope_scaling)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                rope_scaling.get("mrope_section", [24, 20, 20]),
                rope_scaling.get("interleaved", False),
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)
        present_key_value = (key_states, value_states) if use_cache else None
        if (
            self.decode_attn_implementation == SDPA
            and query_states.shape[-2] == 1
            and not self.training
        ):
            attn_output = F.scaled_dot_product_attention(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=self.num_key_value_groups > 1,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            output = self.o_proj(attn_output.reshape(*input_shape, -1))
            return (output, present_key_value) if use_cache else output
        if (
            self.attn_implementation in {
                FLASH_ATTENTION_3,
                FLASH_ATTENTION_2,
            }
            and query_states.is_cuda
            and query_states.dtype in (torch.float16, torch.bfloat16)
            and (
                self.attn_implementation == FLASH_ATTENTION_2
                or not self.training
                or self.attention_dropout == 0.0
            )
        ):
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = _flash_attention_eager(
                query_states, key_states, value_states,
                implementation=self.attn_implementation,
                decode_implementation=self.decode_attn_implementation,
                training=self.training,
                dropout_p=self.attention_dropout,
                scaling=self.scaling,
            )
            output = self.o_proj(attn_output.reshape(*input_shape, -1))
            return (output, present_key_value) if use_cache else output
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if self.attn_implementation == "sdpa":
            attn_output = F.scaled_dot_product_attention(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            output = self.o_proj(attn_output.reshape(*input_shape, -1))
            return (output, present_key_value) if use_cache else output
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        output = self.o_proj(attn_output.reshape(*input_shape, -1))
        return (output, present_key_value) if use_cache else output


class Qwen3TTSDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, multimodal=False):
        super().__init__()
        self.self_attn = Qwen3TTSAttention(config, layer_idx, multimodal=multimodal)
        self.mlp = Qwen3TTSMLP(config)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        *,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_output
        else:
            present_key_value = None
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, present_key_value) if use_cache else hidden_states


class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, act, bias=False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = _activation(act)

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3TTSTalkerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx, multimodal=True) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config, multimodal=True)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)

    def get_input_embeddings(self):
        return self.codec_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        *,
        past_key_values=None,
        use_cache=False,
    ):
        batch_size, seq_len, _ = inputs_embeds.shape
        if past_key_values is not None:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    "past_key_values must contain one entry per Talker layer"
                )
            past_len = int(past_key_values[0][0].shape[-2])
        else:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        key_len = past_len + seq_len
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                key_len,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        elif attention_mask.shape[-1] != key_len:
            raise ValueError(
                "attention_mask length must equal cached plus current sequence "
                f"length ({attention_mask.shape[-1]} != {key_len})"
            )
        if position_ids is None:
            pos = torch.arange(
                past_len,
                key_len,
                device=inputs_embeds.device,
            ).view(1, -1).expand(batch_size, -1)
            position_ids = pos.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        hidden_states = inputs_embeds
        causal = _causal_mask(
            batch_size,
            seq_len,
            inputs_embeds.device,
            inputs_embeds.dtype,
            key_len=key_len,
            past_len=past_len,
        )
        padding_mask = (~attention_mask.bool()).view(
            batch_size,
            1,
            1,
            key_len,
        )
        causal = causal.masked_fill(
            padding_mask,
            torch.finfo(inputs_embeds.dtype).min,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            layer_output = layer(
                hidden_states,
                causal,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer_output
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states


class Qwen3TTSTalkerCodePredictorModel(nn.Module):
    def __init__(self, config, embedding_dim):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx, multimodal=False) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config, multimodal=False)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self):
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds,
        *,
        past_key_values=None,
        use_cache=False,
    ):
        batch_size, seq_len, _ = inputs_embeds.shape
        if past_key_values is not None:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    "past_key_values must contain one entry per code predictor layer"
                )
            past_len = int(past_key_values[0][0].shape[-2])
        else:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        key_len = past_len + seq_len
        hidden_states = inputs_embeds
        attention_mask = _causal_mask(
            batch_size,
            seq_len,
            inputs_embeds.device,
            inputs_embeds.dtype,
            key_len=key_len,
            past_len=past_len,
        )
        position_ids = torch.arange(
            past_len,
            key_len,
            device=inputs_embeds.device,
        ).view(1, -1).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        present_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            layer_output = layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                hidden_states = layer_output
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(present_key_values)
        return hidden_states


class Qwen3TTSTalkerCodePredictor(nn.Module):
    def __init__(self, config, talker_config):
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward_finetune(self, inputs_embeds):
        hidden_states = self.model(self.small_to_mtp_projection(inputs_embeds))
        logits = [self.lm_head[i - 1](hidden_states[:, i]) for i in range(1, self.config.num_code_groups)]
        return torch.stack(logits, dim=1)

    def logits_for_prefix(
        self,
        inputs_embeds,
        codebook_index,
        *,
        past_key_values=None,
        use_cache=False,
    ):
        model_output = self.model(
            self.small_to_mtp_projection(inputs_embeds),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            hidden_states, present_key_values = model_output
            return (
                self.lm_head[codebook_index - 1](hidden_states[:, -1]),
                present_key_values,
            )
        return self.lm_head[codebook_index - 1](model_output[:, -1])


class Qwen3TTSTalker(nn.Module):
    def __init__(
        self,
        model_path: str,
        attn_implementation: str | None = None,
        decode_attn_implementation: str | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(f"Could not find Qwen3-TTS config.json under {model_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            root_config = json.load(f)
        self.config = _to_namespace(root_config["talker_config"])
        self.config.code_predictor_config = _to_namespace(root_config["talker_config"]["code_predictor_config"])
        attn_implementation = _normalize_attn_implementation(attn_implementation)
        decode_attn_implementation = _normalize_decode_attn_implementation(
            decode_attn_implementation
        )
        self.config.attn_implementation = attn_implementation
        self.config.decode_attn_implementation = decode_attn_implementation
        self.config._attn_implementation = attn_implementation
        self.config.code_predictor_config.attn_implementation = attn_implementation
        self.config.code_predictor_config._attn_implementation = attn_implementation
        self.config.code_predictor_config.decode_attn_implementation = (
            decode_attn_implementation
        )
        if not hasattr(self.config.code_predictor_config, "layer_types") or self.config.code_predictor_config.layer_types is None:
            self.config.code_predictor_config.layer_types = ["full_attention"] * self.config.code_predictor_config.num_hidden_layers
        self.model = Qwen3TTSTalkerModel(self.config)
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            self.config.text_hidden_size,
            self.config.text_hidden_size,
            self.config.hidden_size,
            self.config.hidden_act,
            bias=True,
        )
        self.codec_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.code_predictor = Qwen3TTSTalkerCodePredictor(self.config.code_predictor_config, self.config)
        self._load_weights(model_path)

    @property
    def hidden_size(self):
        return int(self.config.hidden_size)

    @property
    def vocab_size(self):
        return int(self.config.vocab_size)

    @property
    def num_code_groups(self):
        return int(self.config.num_code_groups)

    @property
    def codec_eos_token_id(self):
        return int(self.config.codec_eos_token_id)

    @property
    def codec_bos_id(self):
        return int(self.config.codec_bos_id)

    def _load_weights(self, model_path):
        state_dict = {}
        prefix = "talker."
        with safe_open(_resolve_safetensors(model_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix):]] = handle.get_tensor(key)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = []
        if unexpected or [key for key in missing if key not in allowed_missing]:
            raise RuntimeError(f"Failed to load Qwen3-TTS talker. missing={missing}, unexpected={unexpected}")

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_text_embeddings(self):
        return self.model.get_text_embeddings()

    def embed_text_tokens(self, text_tokens):
        return self.text_projection(self.get_text_embeddings()(text_tokens))

    def embed_first_code(self, code_ids):
        clean = code_ids.clamp(0, self.vocab_size - 1).long()
        return self.get_input_embeddings()(clean)

    def embed_codec_frames(self, codec_ids):
        if codec_ids.dim() == 2:
            codec_ids = codec_ids.unsqueeze(0)
        embeds = self.embed_first_code(codec_ids[..., 0])
        active_groups = min(codec_ids.shape[-1], self.num_code_groups)
        for codebook_index in range(1, active_groups):
            clean = codec_ids[..., codebook_index].clamp(0, self.config.code_predictor_config.vocab_size - 1).long()
            embeds = embeds + self.code_predictor.get_input_embeddings()[codebook_index - 1](clean)
        return embeds

    def forward_main(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        *,
        past_key_values=None,
        use_cache=False,
    ):
        model_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            hidden_states, present_key_values = model_output
            return (
                hidden_states,
                self.codec_head(hidden_states),
                present_key_values,
            )
        hidden_states = model_output
        return hidden_states, self.codec_head(hidden_states)

    def residual_logits(self, talker_hidden_states, codec_ids):
        if codec_ids.dim() != 2:
            raise ValueError(f"codec_ids should be [N, G], got {tuple(codec_ids.shape)}")
        prefix = [talker_hidden_states.unsqueeze(1), self.embed_first_code(codec_ids[:, :1])]
        for codebook_index in range(1, self.num_code_groups - 1):
            prefix.append(self.code_predictor.get_input_embeddings()[codebook_index - 1](codec_ids[:, codebook_index:codebook_index + 1]))
        return self.code_predictor.forward_finetune(torch.cat(prefix, dim=1))

    def residual_logits_for_prefix(
        self,
        prefix_embeds,
        codebook_index,
        *,
        past_key_values=None,
        use_cache=False,
    ):
        return self.code_predictor.logits_for_prefix(
            prefix_embeds,
            codebook_index,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
