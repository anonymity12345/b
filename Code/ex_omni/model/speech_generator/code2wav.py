import contextlib
import io
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors import safe_open
except Exception:
    safe_load_file = None
    safe_open = None


@dataclass
class Qwen3OmniCode2WavConfig:
    codebook_size: int = 2048
    hidden_size: int = 1024
    max_position_embeddings: int = 8000
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    attention_bias: bool = False
    sliding_window: int = 72
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    num_hidden_layers: int = 8
    num_quantizers: int = 16
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    sample_rate: int = 24000

    @classmethod
    def from_dict(cls, config_dict):
        values = dict(config_dict or {})
        if "rope_parameters" in values and values["rope_parameters"] is not None:
            values["rope_theta"] = values["rope_parameters"].get("rope_theta", values.get("rope_theta", 10000.0))
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in values.items() if key in allowed}
        for key in ("upsample_rates", "upsampling_ratios"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @property
    def layer_types(self):
        return ["sliding_attention"] * self.num_hidden_layers


def _get_activation(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def make_sliding_causal_mask(batch_size, query_len, key_len, sliding_window, device, dtype):
    q_pos = torch.arange(key_len - query_len, key_len, device=device)[:, None]
    k_pos = torch.arange(key_len, device=device)[None, :]
    masked = k_pos > q_pos
    if sliding_window is not None:
        masked = masked | (k_pos <= (q_pos - sliding_window))
    mask = torch.zeros((query_len, key_len), device=device, dtype=dtype)
    mask = mask.masked_fill(masked, torch.finfo(dtype).min)
    return mask.view(1, 1, query_len, key_len).expand(batch_size, 1, query_len, key_len)


class Qwen3OmniRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class Qwen3OmniCode2WavAttention(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(self, hidden_states, position_embeddings, attention_mask):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class Qwen3OmniCode2WavMlp(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _get_activation(config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3OmniCode2WavRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3OmniCode2WavLayerScale(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig):
        super().__init__()
        self.scale = nn.Parameter(torch.full((config.hidden_size,), config.layer_scale_initial_scale))

    def forward(self, x):
        return self.scale * x


class Qwen3OmniCode2WavTransformerLayer(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3OmniCode2WavAttention(config, layer_idx)
        self.mlp = Qwen3OmniCode2WavMlp(config)
        self.input_layernorm = Qwen3OmniCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3OmniCode2WavLayerScale(config)
        self.mlp_layer_scale = Qwen3OmniCode2WavLayerScale(config)

    def forward(self, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.mlp_layer_scale(hidden_states)


class Qwen3OmniCode2WavTransformerModel(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3OmniCode2WavTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3OmniCode2WavRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniRotaryEmbedding(config=config)

    def forward(self, inputs_embeds):
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = make_sliding_causal_mask(
            batch_size=batch_size,
            query_len=seq_len,
            key_len=seq_len,
            sliding_window=self.config.sliding_window,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_embeddings)
        return self.norm(hidden_states)


class Qwen3OmniCausalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state):
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return int(ideal_length - length)

    def forward(self, hidden_state):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class Qwen3OmniCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = self.left_pad

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        return hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad].contiguous()


class Qwen3OmniConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3OmniCausalConvNet(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).permute(0, 2, 1)
        hidden_states = self.pwconv2(self.act(self.pwconv1(self.norm(hidden_states))))
        hidden_states = (self.gamma * hidden_states).permute(0, 2, 1)
        return residual + hidden_states


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(hidden_states * alpha), 2)


class Qwen3OmniCode2WavDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3OmniCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3OmniCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        return hidden_state + self.conv2(self.act2(self.conv1(self.act1(hidden_state))))


class Qwen3OmniCode2WavDecoderBlock(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig, layer_idx):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]
        block = [SnakeBeta(in_dim), Qwen3OmniCausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate)]
        for dilation in (1, 3, 9):
            block.append(Qwen3OmniCode2WavDecoderResidualUnit(out_dim, dilation))
        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden


class Qwen3OmniCode2Wav(nn.Module):
    def __init__(self, config: Qwen3OmniCode2WavConfig):
        super().__init__()
        self.config = config
        self.total_upsample = int(np.prod(config.upsample_rates + config.upsampling_ratios))
        self.pre_transformer = Qwen3OmniCode2WavTransformerModel(config)
        self.code_embedding = nn.Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        self.register_buffer(
            "code_offset",
            torch.arange(config.num_quantizers).view(1, -1, 1) * config.codebook_size,
            persistent=False,
        )
        self.upsample = nn.ModuleList(
            nn.ModuleList([Qwen3OmniCausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                           Qwen3OmniConvNeXtBlock(config.hidden_size)])
            for factor in config.upsampling_ratios
        )
        decoder = [Qwen3OmniCausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3OmniCode2WavDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [SnakeBeta(output_dim), Qwen3OmniCausalConvNet(output_dim, 1, 7)]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} codebooks, got {codes.shape[1]}")
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        hidden = self.pre_transformer(inputs_embeds=hidden).permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


def load_code2wav_config(model_path: str, config_path: Optional[str] = None) -> Qwen3OmniCode2WavConfig:
    path = config_path or os.path.join(model_path, "config.json")
    with open(path, "r") as f:
        raw_config = json.load(f)
    code2wav_config = raw_config.get("code2wav_config", raw_config)
    return Qwen3OmniCode2WavConfig.from_dict(code2wav_config)


def _load_state_dict(model_path: str):
    if os.path.isfile(model_path):
        files = [model_path]
    else:
        files = [
            os.path.join(model_path, name)
            for name in os.listdir(model_path)
            if name.endswith((".safetensors", ".bin", ".pth", ".pt"))
        ]
    state_dict = {}
    for path in sorted(files):
        if path.endswith(".safetensors"):
            if safe_open is None:
                raise ImportError("safetensors is required to load .safetensors code2wav weights")
            part = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("code2wav."):
                        part[key] = f.get_tensor(key)
                    elif not key.startswith(("thinker.", "talker.")):
                        part[key] = f.get_tensor(key)
        else:
            part = torch.load(path, map_location="cpu")
            if isinstance(part, dict) and "state_dict" in part:
                part = part["state_dict"]
        state_dict.update(part)
    return state_dict


def load_code2wav(model_path: str, config_path: Optional[str] = None, device="cuda", dtype=None):
    config = load_code2wav_config(model_path, config_path=config_path)
    model = Qwen3OmniCode2Wav(config)
    state_dict = _load_state_dict(model_path)
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("code2wav."):
            filtered[key[len("code2wav.") :]] = value
        elif not key.startswith(("thinker.", "talker.")):
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if len(unexpected) > 0:
        print(f"Unexpected code2wav keys: {unexpected[:20]}")
    if len(missing) > 0:
        print(f"Missing code2wav keys: {missing[:20]}")
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval()


def _slice_streaming_waveform(
    waveform: torch.Tensor,
    *,
    code_seq_len: int,
    total_upsample: int,
    left_context_size: int,
) -> torch.Tensor:
    """Apply Qwen3-Omni's context trim with causal-tail compensation."""
    nominal_length = int(code_seq_len) * int(total_upsample)
    tail = max(0, nominal_length - int(waveform.shape[-1]))
    start = max(0, int(left_context_size) * int(total_upsample) - tail)
    return waveform[..., start:nominal_length]


class Qwen3OmniWaveformStream:
    """Stateful 16-codebook waveform decoder following Qwen async-chunk."""

    def __init__(
        self,
        decoder,
        *,
        initial_chunk_size: int = 6,
        chunk_size: int = 6,
        left_context_size: int | None = None,
    ) -> None:
        if initial_chunk_size <= 0 or chunk_size <= 0:
            raise ValueError("waveform chunk sizes must be positive")
        if left_context_size is None:
            decoder_config = getattr(
                getattr(decoder.model, "decoder", None), "config", None
            )
            left_context_size = int(
                getattr(decoder_config, "sliding_window", 72)
            )
        if left_context_size < 0:
            raise ValueError("waveform left context must be non-negative")
        self.decoder = decoder
        self.initial_chunk_size = int(initial_chunk_size)
        self.chunk_size = int(chunk_size)
        self.left_context_size = int(left_context_size)
        self.sample_rate = int(decoder.sample_rate)
        self._pending = np.empty((0, decoder.num_quantizers), dtype=np.int64)
        self._context = np.empty((0, decoder.num_quantizers), dtype=np.int64)
        self._started = False
        self._closed = False
        self._start_sample = 0

    def _normalize(self, codes) -> np.ndarray:
        parsed = self.decoder.parse_codes(codes)
        if parsed.shape[0] != 1:
            raise ValueError("waveform streaming supports batch_size=1")
        return np.ascontiguousarray(
            parsed[0].detach().cpu().numpy(), dtype=np.int64
        )

    def _decode(self, current: np.ndarray, *, final: bool) -> dict:
        context_size = int(self._context.shape[0])
        window = np.concatenate([self._context, current], axis=0)
        waveform = self.decoder.streaming_inference(
            window,
            left_context_size=context_size,
        )
        samples = np.ascontiguousarray(
            waveform.detach().cpu().float().numpy().reshape(-1),
            dtype=np.float32,
        )
        payload = {
            "waveform": samples,
            "sample_rate": self.sample_rate,
            "start_sample": self._start_sample,
            "units": int(current.shape[0]),
            "final": bool(final),
        }
        self._start_sample += int(samples.shape[0])
        history = np.concatenate([self._context, current], axis=0)
        self._context = np.ascontiguousarray(
            history[-self.left_context_size :]
            if self.left_context_size
            else history[:0],
            dtype=np.int64,
        )
        self._started = True
        return payload

    def push_units(self, codes) -> list[dict]:
        if self._closed:
            raise RuntimeError("waveform stream is already finished")
        values = self._normalize(codes)
        if not len(values):
            return []
        self._pending = np.concatenate([self._pending, values], axis=0)
        chunks = []
        while True:
            target = self.chunk_size if self._started else self.initial_chunk_size
            if len(self._pending) < target:
                break
            current, self._pending = (
                self._pending[:target],
                self._pending[target:],
            )
            chunks.append(self._decode(current, final=False))
        return chunks

    def finish(self) -> list[dict]:
        if self._closed:
            return []
        self._closed = True
        if len(self._pending):
            current = self._pending
            self._pending = self._pending[:0]
            return [self._decode(current, final=True)]
        return [
            {
                "waveform": np.empty((0,), dtype=np.float32),
                "sample_rate": self.sample_rate,
                "start_sample": self._start_sample,
                "units": 0,
                "final": True,
            }
        ]


def _import_qwen3_tts_tokenizer():
    import inspect
    import transformers.masking_utils as masking_utils
    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils.generic as generic

    original_causal_mask = masking_utils.create_causal_mask
    original_sliding_mask = masking_utils.create_sliding_window_causal_mask
    mask_parameters = inspect.signature(original_causal_mask).parameters
    needs_mask_compat = (
        "inputs_embeds" in mask_parameters
        and "input_embeds" not in mask_parameters
    )

    def mask_compat(original):
        def create_mask(
            config,
            input_embeds=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            position_ids=None,
            cache_position=None,
            **kwargs,
        ):
            del cache_position
            values = inputs_embeds if inputs_embeds is not None else input_embeds
            return original(
                config=config,
                inputs_embeds=values,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )

        return create_mask

    if needs_mask_compat:
        masking_utils.create_causal_mask = mask_compat(original_causal_mask)
        masking_utils.create_sliding_window_causal_mask = mask_compat(
            original_sliding_mask
        )

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def default_rope_parameters(
            config, device=None, seq_len=None, layer_type=None
        ):
            del seq_len, layer_type
            base = config.rope_theta
            partial = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = (
                getattr(config, "head_dim", None)
                or config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float
                    )
                    / dim
                )
            )
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
            default_rope_parameters
        )

    original = generic.check_model_inputs
    parameters = list(inspect.signature(original).parameters.values())
    needs_transformers5_compat = bool(parameters) and (
        parameters[0].name == "func"
        and parameters[0].default is inspect.Parameter.empty
    )
    if needs_transformers5_compat:
        def check_model_inputs_compat(func=None):
            replacement = generic.merge_with_config_defaults
            return replacement if func is None else replacement(func)

        generic.check_model_inputs = check_model_inputs_compat
    captured_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout):
            from qwen_tts import Qwen3TTSTokenizer
    finally:
        generic.check_model_inputs = original
        masking_utils.create_causal_mask = original_causal_mask
        masking_utils.create_sliding_window_causal_mask = (
            original_sliding_mask
        )
    for line in captured_stdout.getvalue().splitlines():
        if line.strip() in {"", "********"}:
            continue
        if "[ERROR] `cache_position`" in line:
            continue
        if "Warning: flash-attn is not installed" in line:
            continue
        print(line)
    return Qwen3TTSTokenizer


class Qwen3OmniCode2WavDecoder:
    """Compatibility wrapper around the official Qwen3-TTS 12 Hz tokenizer decoder."""

    expects_multicodebook = True

    def __init__(self, model_path: str, config_path: Optional[str] = None, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype or torch.float32
        if config_path is not None:
            # The official tokenizer reads config.json from model_path. Keeping
            # this argument in the wrapper preserves the old public API.
            print("Qwen3OmniCode2WavDecoder: code2wav_config_path is ignored by the official tokenizer decoder")
        try:
            Qwen3TTSTokenizer = _import_qwen3_tts_tokenizer()
        except Exception as exc:
            raise ImportError(
                "The official WAV decoder requires qwen-tts and its runtime dependencies "
                "(including the SoX executable)."
            ) from exc

        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_path,
            device_map=device,
            dtype=self.dtype,
        )
        self.model = self.tokenizer.model
        self.sample_rate = int(self.tokenizer.get_output_sample_rate())
        self.num_quantizers = int(
            getattr(self.model.config, "encoder_valid_num_quantizers", 16)
        )

    def parse_codes(self, codes):
        if isinstance(codes, str):
            frames = []
            for frame in codes.strip().split():
                parts = [
                    part.strip().strip("[]")
                    for part in frame.replace("|", ",").replace(";", ",").split(",")
                    if part.strip().strip("[]")
                ]
                frames.append([int(part) for part in parts])
            codes = torch.tensor(frames, dtype=torch.long)
        elif not torch.is_tensor(codes):
            codes = torch.tensor(codes, dtype=torch.long)

        if codes.dim() == 2:
            # Official decode expects [B, T, G].
            codes = codes.unsqueeze(0)
        elif codes.dim() == 3 and codes.shape[1] == self.num_quantizers and codes.shape[2] != self.num_quantizers:
            # Accept [B, G, T] input by transposing the codebook dimension.
            codes = codes.transpose(1, 2)
        elif codes.dim() != 3:
            raise ValueError(f"Expected codes as [T,G] or [B,T,G]/[B,G,T], got {tuple(codes.shape)}")

        if codes.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"Expected {self.num_quantizers} codebooks, got shape {tuple(codes.shape)}"
            )

        return codes.to(dtype=torch.long)

    @torch.inference_mode()
    def offline_inference(self, codes):
        codes = self.parse_codes(codes)
        wavs, sample_rate = self.tokenizer.decode({"audio_codes": codes})
        if int(sample_rate) != self.sample_rate:
            raise ValueError(f"Unexpected decoder sample rate: {sample_rate} != {self.sample_rate}")
        if not wavs:
            return torch.zeros(1, 0, dtype=torch.float32)
        return torch.from_numpy(np.asarray(wavs[0], dtype=np.float32)).reshape(1, -1)

    @torch.inference_mode()
    def streaming_inference(self, codes, *, left_context_size: int = 0):
        codes = self.parse_codes(codes).to(self.tokenizer.device)
        decoder = getattr(self.model, "decoder", None)
        if decoder is None or not hasattr(decoder, "total_upsample"):
            raise RuntimeError(
                "the configured Qwen3-TTS tokenizer has no streaming decoder"
            )
        code_seq_len = int(codes.shape[1])
        waveform = decoder(codes.transpose(1, 2))
        waveform = _slice_streaming_waveform(
            waveform,
            code_seq_len=code_seq_len,
            total_upsample=int(decoder.total_upsample),
            left_context_size=int(left_context_size),
        )
        return waveform[0, 0].detach().cpu().float()

    def start_stream(
        self,
        *,
        initial_chunk_size: int = 6,
        chunk_size: int = 6,
        left_context_size: int | None = None,
    ) -> Qwen3OmniWaveformStream:
        return Qwen3OmniWaveformStream(
            self,
            initial_chunk_size=initial_chunk_size,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )

    @torch.inference_mode()
    def iter_chunked_inference(self, codes, chunk_size=300, left_context_size=25):
        # The official 12 Hz decoder already performs its own context-aware
        # chunked decode. Do not split codes again at this wrapper layer.
        yield self.offline_inference(codes)

    @torch.inference_mode()
    def chunked_inference(self, codes, chunk_size=300, left_context_size=25):
        return self.offline_inference(codes)
