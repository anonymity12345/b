import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import lru_cache
from typing import Tuple, Optional
from einops import rearrange
from ..utils.io_utils import hash_state_dict_keys
from .audio_pack import AudioPack
from .attention_ops import video_flash_attention
from ex_omni.model.attention import (
    FLASH_ATTENTION_3,
    FLASH_ATTENTION_2,
    flash_attention_function,
    resolve_attn_implementation,
)

from ..utils import args_config
from ..streaming.causal_attention import (
    make_3d_positions,
    prepare_streaming_attention_metadata,
    prepare_cache_update_selection,
    streaming_attention,
    update_compressed_cache,
    update_full_cache,
)
from ..streaming.state import LayerKVCache
from ex_omni.distributed.sequence_parallel import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
def _runtime_args():
    return args_config.args


@lru_cache(maxsize=1)
def _attention_backend():
    runtime_args = _runtime_args()
    requested = (
        getattr(runtime_args, "attn_implementation", FLASH_ATTENTION_2)
        if runtime_args is not None
        else FLASH_ATTENTION_2
    )
    # Runtime config has already resolved hardware availability. Avoid tracing
    # optional-package imports / CUDA capability probes inside every DiT layer.
    if torch.compiler.is_compiling():
        return requested
    return resolve_attn_implementation(requested, warn_on_fallback=False)


def prepare_audio_condition(audio_emb, audio_proj, audio_cond_projs):
    if audio_emb is None:
        return None
    audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
    audio_emb = torch.cat(
        [audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb],
        dim=2,
    )
    audio_emb = audio_proj(audio_emb)
    return torch.stack(
        [audio_cond_proj(audio_emb) for audio_cond_proj in audio_cond_projs],
        dim=1,
    )
    
    
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
):
    backend = _attention_backend()
    can_flash = (
        not compatibility_mode
        and backend in {FLASH_ATTENTION_3, FLASH_ATTENTION_2}
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    if can_flash:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = video_flash_attention(q, k, v, backend)
        return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    freqs = torch.outer(
        torch.arange(end, device=freqs.device, dtype=torch.float32),
        freqs,
    )
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    pairs = x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    if torch.compiler.is_compiling():
        # Inductor cannot fuse complex multiplication. The equivalent real
        # formula fuses casts, RoPE and output packing into one pointwise kernel.
        real, imag = pairs.unbind(-1)
        cosine, sine = freqs.real, freqs.imag
        x_out = torch.stack((real * cosine - imag * sine,
                             real * sine + imag * cosine), dim=-1).flatten(2)
    else:
        x_out = torch.view_as_real(torch.view_as_complex(pairs) * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(
        self,
        x,
        freqs,
        streaming_cache=None,
        streaming_positions=None,
        streaming_global_positions=None,
        streaming_config=None,
        streaming_base_freqs=None,
        streaming_tokens_per_frame=1,
        update_streaming_cache=False,
        streaming_compressed=False,
        streaming_attention_metadata=None,
    ):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        if streaming_config is None:
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            x = self.attn(q, k, v)
            return self.o(x)

        if streaming_positions is None or streaming_base_freqs is None:
            raise ValueError("Streaming attention requires positions and base RoPE frequencies.")
        runtime_args = _runtime_args()
        sp_size = (
            int(getattr(runtime_args, "sp_size", 1))
            if runtime_args is not None
            else 1
        )
        current_positions = streaming_positions
        head_parallel = sp_size > 1 and getattr(runtime_args, "sequence_parallel_mode", "gather") == "ulysses"
        attention_heads = self.num_heads
        local_query_length = q.shape[1]
        if sp_size > 1:
            distributed_sp_size = int(get_sequence_parallel_world_size())
            if distributed_sp_size != sp_size:
                raise RuntimeError(
                    "Configured sp_size does not match the initialized sequence "
                    f"parallel group ({sp_size} != {distributed_sp_size})."
                )
            if streaming_global_positions is None:
                raise ValueError(
                    "Streaming sequence parallelism requires global current positions."
                )
            current_positions = streaming_global_positions
            # Gather K/V together; padded queries must never become keys or
            # persistent history. Spatial grids need not divide the GPU count.
            if head_parallel:
                qkv = get_sp_group().sequence_to_heads(torch.stack((q, k, v), dim=2), num_heads=self.num_heads)
                q, k, v = qkv[:, :current_positions.shape[0]].unbind(2)
                attention_heads = q.shape[-1] // self.head_dim
                streaming_positions = current_positions
            else:
                kv = get_sp_group().all_gather(torch.cat((k, v), dim=-1), dim=1)
                k, v = kv[:, :current_positions.shape[0]].chunk(2, dim=-1)
        if streaming_cache is not None and streaming_cache.current_key_output is not None:
            streaming_cache.current_key_output.copy_(k)
        if streaming_attention_metadata is None:
            streaming_attention_metadata = prepare_streaming_attention_metadata(
                streaming_cache, streaming_positions, current_positions,
                streaming_base_freqs, streaming_config, compressed=streaming_compressed,
            )
        query_freqs, key_freqs, full_indices, compressed_indices = streaming_attention_metadata[:4]
        reuse_history = (
            streaming_cache is not None and not update_streaming_cache
            and not self.training and not torch.is_grad_enabled()
            # Captured graph states carry only prepacked K/V. They have no
            # sink_key to rebuild history, including during torch.compile.
            and (not torch.compiler.is_compiling() or streaming_cache.sink_key is None)
            and getattr(_runtime_args(), "cache_rotated_history", False)
        )
        prepared = streaming_cache.prepared_attention if reuse_history else None
        cache_seqlens = None
        if prepared is not None and prepared[0] is key_freqs:
            _, rotated_key, all_value = prepared[:3]
            current_key = rope_apply(k, key_freqs[-k.shape[1]:], attention_heads)
            if len(prepared) == 5:
                cache_seqlens, current_indices = prepared[3:]
                rotated_key.index_copy_(1, current_indices, current_key)
                all_value.index_copy_(1, current_indices, v)
            else:
                rotated_key[:, -k.shape[1]:].copy_(current_key)
                all_value[:, -v.shape[1]:].copy_(v)
        else:
            key_parts, value_parts = [], []
            full_tokens = full_indices.numel() if full_indices is not None else 0
            compressed_tokens = compressed_indices.numel() if compressed_indices is not None else 0
            sink_tokens = key_freqs.shape[0] - k.shape[1] - full_tokens - compressed_tokens
            if sink_tokens:
                key_parts.append(streaming_cache.sink_key.to(k))
                value_parts.append(streaming_cache.sink_value.to(v))
            if compressed_tokens:
                key_parts.append(streaming_cache.compressed_key.index_select(1, compressed_indices).to(k))
                value_parts.append(streaming_cache.compressed_value.index_select(1, compressed_indices).to(v))
            if full_tokens:
                key_parts.append(streaming_cache.key.index_select(1, full_indices).to(k))
                value_parts.append(streaming_cache.value.index_select(1, full_indices).to(v))
            key_parts.append(k)
            value_parts.append(v)
            rotated_key = rope_apply(torch.cat(key_parts, dim=1), key_freqs, attention_heads)
            all_value = torch.cat(value_parts, dim=1)
            if reuse_history:
                streaming_cache.prepared_attention = (key_freqs, rotated_key, all_value)
            elif streaming_cache is not None:
                streaming_cache.prepared_attention = None
        output = streaming_attention(
            rope_apply(q, query_freqs, attention_heads), rotated_key, all_value,
            attention_heads, attn_implementation=_attention_backend(),
            cache_seqlens=cache_seqlens,
        )
        next_cache = streaming_cache
        if update_streaming_cache:
            update = update_compressed_cache if streaming_compressed else update_full_cache
            next_cache = update(streaming_cache, k, v, current_positions, streaming_config,
                                selection=streaming_attention_metadata[4] if len(streaming_attention_metadata) > 4 else None)
        if head_parallel:
            output = F.pad(output, (0, 0, 0, sp_size * local_query_length - output.shape[1]))
            output = get_sp_group().heads_to_sequence(output, num_heads=self.num_heads, head_dim=self.head_dim)
        return self.o(output), next_cache


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, prepared_kv=None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        if prepared_kv is None:
            k = self.norm_k(self.k(ctx))
            v = self.v(ctx)
        else:
            k, v = prepared_kv
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        streaming_cache=None,
        streaming_positions=None,
        streaming_global_positions=None,
        streaming_config=None,
        streaming_base_freqs=None,
        streaming_tokens_per_frame=1,
        update_streaming_cache=False,
        streaming_compressed=False,
        streaming_attention_metadata=None,
        prepared_cross_attention=None,
    ):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attention_output = self.self_attn(
            input_x,
            freqs,
            streaming_cache=streaming_cache,
            streaming_positions=streaming_positions,
            streaming_global_positions=streaming_global_positions,
            streaming_config=streaming_config,
            streaming_base_freqs=streaming_base_freqs,
            streaming_tokens_per_frame=streaming_tokens_per_frame,
            update_streaming_cache=update_streaming_cache,
            streaming_compressed=streaming_compressed,
            streaming_attention_metadata=streaming_attention_metadata,
        )
        next_cache = None
        if streaming_config is not None:
            attention_output, next_cache = attention_output
        x = self.gate(x, gate_msa, attention_output)
        x = x + self.cross_attn(self.norm3(x), context, prepared_kv=prepared_cross_attention)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if streaming_config is not None:
            return x, next_cache
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x



class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        audio_hidden_size: int=32,
    ):
        super().__init__()
        runtime_args = _runtime_args()
        model_config = getattr(runtime_args, "model_config", None) if runtime_args is not None else None
        if model_config is not None:
            in_dim = int(model_config.get("in_dim", in_dim))
            audio_hidden_size = int(model_config.get("audio_hidden_size", audio_hidden_size))
        self.dim = dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
            # nn.LayerNorm(dim)
        self.enable_far_compression = False
        self.compressed_patch_size = (1, 4, 4)
        self.far_patch_embedding = None
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.delta_time_embedding = None
        self.deltatime_type = None
        self.register_buffer(
            "flowmap_gate",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=False,
        )
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

        self.use_audio = bool(getattr(runtime_args, "use_audio", False)) if runtime_args is not None else False
        if self.use_audio:
            audio_input_dim = 10752
            audio_out_dim = dim
            self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
            self.audio_cond_projs = nn.ModuleList()
            for d in range(num_layers // 2 - 1):
                l = nn.Linear(audio_hidden_size, audio_out_dim)
                self.audio_cond_projs.append(l)      

    def setup_far_conditioning(
        self,
        *,
        compressed_patch_size=(1, 4, 4),
    ):
        """Attach the FAR compressed patch embed before Streaming weights load."""
        compressed_patch_size = tuple(int(value) for value in compressed_patch_size)
        if len(compressed_patch_size) != 3:
            raise ValueError("compressed_patch_size must have three dimensions.")
        if self.far_patch_embedding is None:
            self.far_patch_embedding = nn.Conv3d(
                self.patch_embedding.in_channels,
                self.dim,
                kernel_size=compressed_patch_size,
                stride=compressed_patch_size,
            ).to(
                device=self.patch_embedding.weight.device,
                dtype=self.patch_embedding.weight.dtype,
            )
            self.initialize_far_patch_embedding()
        elif tuple(self.far_patch_embedding.kernel_size) != compressed_patch_size:
            raise ValueError("FAR compressed patch size is already configured differently.")
        self.compressed_patch_size = compressed_patch_size
        self.enable_far_compression = True
        return self

    @torch.no_grad()
    def initialize_far_patch_embedding(self):
        if self.far_patch_embedding is None:
            return
        weight = self.patch_embedding.weight.detach().float()
        flattened = weight.reshape(-1, 1, *self.patch_size)
        resized = F.interpolate(
            flattened,
            size=self.far_patch_embedding.kernel_size,
            mode="trilinear",
            align_corners=False,
        )
        self.far_patch_embedding.weight.copy_(
            resized.reshape_as(self.far_patch_embedding.weight).to(
                self.far_patch_embedding.weight
            )
        )
        if self.patch_embedding.bias is not None:
            self.far_patch_embedding.bias.copy_(
                self.patch_embedding.bias.to(self.far_patch_embedding.bias)
            )

    def setup_flowmap_conditioning(
        self,
        *,
        gate: float = 0.25,
        deltatime_type: str = "r",
    ):
        """Attach the secondary time embedding while preserving loaded weights."""
        if deltatime_type not in {"r", "t-r"}:
            raise ValueError("deltatime_type must be 'r' or 't-r'.")
        if not 0.0 <= float(gate) <= 1.0:
            raise ValueError("flowmap gate must be in [0, 1].")
        if self.delta_time_embedding is None:
            self.delta_time_embedding = copy.deepcopy(self.time_embedding)
        self.deltatime_type = deltatime_type
        self.flowmap_gate.fill_(float(gate))
        return self

    @torch.compiler.disable
    def _prepare_inference_text_context(self, context, state):
        cached = state.text_context_cache
        if cached is not None and cached[0] is context:
            return cached[1], cached[2]
        embedded = self.text_embedding(context)
        projections = tuple((block.cross_attn.norm_k(block.cross_attn.k(embedded)),
                             block.cross_attn.v(embedded)) for block in self.blocks)
        state.text_context_cache = (context, embedded, projections)
        return embedded, projections

    def prepare_inference_conditions(self, context, audio_emb=None):
        """Project fixed full-sequence conditions once for one denoising window."""
        if self.training or torch.is_grad_enabled() or self.has_image_input:
            raise ValueError("Static conditions require inference without image-text context")
        embedded = self.text_embedding(context)
        projections = tuple(
            (block.cross_attn.norm_k(block.cross_attn.k(embedded)),
             block.cross_attn.v(embedded)) for block in self.blocks
        )
        result = {"prepared_text_context": (embedded, projections)}
        if self.use_audio:
            result["prepared_audio_condition"] = prepare_audio_condition(
                audio_emb, self.audio_proj, self.audio_cond_projs
            )
        return result

    def time_conditioning(
        self,
        timestep: torch.Tensor,
        r_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        if r_timestep is None:
            return time_emb
        if self.delta_time_embedding is None or self.deltatime_type is None:
            raise ValueError(
                "r_timestep requires setup_flowmap_conditioning() before inference."
            )
        delta_timestep = (
            r_timestep
            if self.deltatime_type == "r"
            else timestep - r_timestep
        )
        delta_emb = self.delta_time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, delta_timestep)
        )
        gate = self.flowmap_gate.to(device=time_emb.device, dtype=time_emb.dtype)
        return (1 - gate) * time_emb + gate * delta_emb

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                r_timestep: Optional[torch.Tensor] = None,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                audio_emb: Optional[torch.Tensor] = None,
                use_gradient_checkpointing_offload: bool = False,
                tea_cache = None,
                **kwargs,
                ):
        streaming_state = kwargs.pop("streaming_state", None)
        share_streaming_metadata = bool(kwargs.pop(
            "share_streaming_metadata",
            getattr(_runtime_args(), "share_streaming_metadata", True),
        ))
        streaming_config = kwargs.pop("streaming_config", None)
        streaming_cache_resolution = str(
            kwargs.pop("streaming_cache_resolution", "full")
        )
        if streaming_cache_resolution not in {"full", "compressed"}:
            raise ValueError("streaming_cache_resolution must be 'full' or 'compressed'.")
        streaming_compressed = streaming_cache_resolution == "compressed"
        update_streaming_cache = bool(kwargs.pop("update_streaming_cache", False))
        prepared_audio_condition = kwargs.pop("prepared_audio_condition", None)
        capture_cache_only = bool(kwargs.pop("capture_cache_only", False))
        streaming_enabled = streaming_state is not None or streaming_config is not None
        if streaming_enabled and (streaming_state is None or streaming_config is None):
            raise ValueError("Both streaming_state and streaming_config are required.")
        if streaming_enabled:
            streaming_config.validate()
            streaming_state.ensure_layers(len(self.blocks))
            if tuple(streaming_config.full_patch_size) != tuple(self.patch_size):
                raise ValueError(
                    "Streaming full_patch_size must match the model patch_size."
                )
            if streaming_compressed:
                if not self.enable_far_compression or self.far_patch_embedding is None:
                    raise ValueError("Compressed cache update requires a FAR-enabled model.")
                if tuple(streaming_config.compressed_patch_size) != tuple(
                    self.compressed_patch_size
                ):
                    raise ValueError("Streaming compressed_patch_size must match the model.")

        t = self.time_conditioning(timestep, r_timestep)
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        prepared_cross_attention = None
        prepared_text = kwargs.pop("prepared_text_context", None)
        if prepared_text is not None:
            if self.training or torch.is_grad_enabled() or self.has_image_input or streaming_enabled:
                raise ValueError("Prepared text context is for offline inference only")
            context, prepared_cross_attention = prepared_text
        elif (streaming_enabled and not self.training and not torch.is_grad_enabled()
                and not self.has_image_input
                and getattr(_runtime_args(), "cache_text_context", False)):
            context, prepared_cross_attention = self._prepare_inference_text_context(context, streaming_state)
        else:
            context = self.text_embedding(context)
        input_shape = x.shape

        if self.use_audio:
            if prepared_audio_condition is not None:
                audio_emb = prepared_audio_condition
            else:
                audio_emb = prepare_audio_condition(
                    audio_emb,
                    self.audio_proj,
                    self.audio_cond_projs,
                )

        x = torch.cat([x, y], dim=1)
        x = (
            self.far_patch_embedding(x)
            if streaming_compressed
            else self.patch_embedding(x)
        )
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        streaming_positions = None
        streaming_base_freqs = None
        if streaming_enabled:
            streaming_positions = make_3d_positions(
                streaming_state.global_frame_offset,
                (f, h, w),
                x.device,
            )
            streaming_base_freqs = tuple(
                frequencies.to(device=x.device) for frequencies in self.freqs
            )
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        def create_streaming_custom_forward(
            module,
            streaming_cache,
            local_streaming_positions,
        ):
            def custom_forward(x, context, t_mod, freqs):
                output, _ = module(
                    x,
                    context,
                    t_mod,
                    freqs,
                    streaming_cache=streaming_cache,
                    streaming_positions=local_streaming_positions,
                    streaming_global_positions=streaming_positions,
                    streaming_config=streaming_config,
                    streaming_base_freqs=streaming_base_freqs,
                    streaming_tokens_per_frame=h * w,
                    update_streaming_cache=False,
                    streaming_compressed=streaming_compressed,
                )
                return output
            return custom_forward
        
        runtime_args = _runtime_args()
        sp_size_arg = int(getattr(runtime_args, "sp_size", 1)) if runtime_args is not None else 1
        if streaming_enabled and tea_cache is not None:
            raise ValueError("TeaCache cannot be combined with streaming KV caches.")
        if tea_cache is not None:
            tea_cache_update = tea_cache.check(self, x, t_mod)
        else:
            tea_cache_update = False
        ori_x_len = x.shape[1]
        local_streaming_positions = streaming_positions
        if tea_cache_update:
            x = tea_cache.update(x)
        else:
            if sp_size_arg > 1:
                # Context Parallel
                sp_size = get_sequence_parallel_world_size()
                pad_size = 0
                if ori_x_len % sp_size != 0:
                    pad_size = sp_size - ori_x_len % sp_size
                    x = torch.cat([x, torch.zeros_like(x[:, -1:]).repeat(1, pad_size, 1)], 1)
                sp_rank = get_sequence_parallel_rank()
                x = torch.chunk(x, sp_size, dim=1)[sp_rank]
                if streaming_enabled:
                    padded_positions = streaming_positions
                    if pad_size:
                        padded_positions = torch.cat([
                            streaming_positions,
                            streaming_positions.new_zeros((pad_size, 3)),
                        ], dim=0)
                    local_streaming_positions = torch.chunk(
                        padded_positions,
                        sp_size,
                        dim=0,
                    )[sp_rank]

            attention_metadata = None
            if streaming_enabled and share_streaming_metadata:
                if update_streaming_cache:
                    streaming_state.attention_metadata_cache.clear()
                attention_metadata = prepare_streaming_attention_metadata(
                    streaming_state.layer_caches[0],
                    (streaming_positions if sp_size_arg > 1 and
                     getattr(runtime_args, "sequence_parallel_mode", "gather") == "ulysses"
                     else local_streaming_positions),
                    streaming_positions, streaming_base_freqs, streaming_config,
                    compressed=streaming_compressed,
                    memo=(streaming_state.attention_metadata_cache
                          if not self.training and not update_streaming_cache else None),
                    memo_key=(streaming_state.global_frame_offset, f, h, w, sp_size_arg,
                              streaming_compressed, tuple(vars(streaming_config).values())),
                )
                if update_streaming_cache:
                    attention_metadata += (prepare_cache_update_selection(
                        streaming_state.layer_caches[0], streaming_positions,
                        streaming_config, compressed=streaming_compressed,
                        contiguous_suffix=getattr(runtime_args, "cache_contiguous_history", False)),)

            for layer_i, block in enumerate(self.blocks):
                # audio cond
                if self.use_audio and audio_emb is not None:
                    au_idx = None
                    if (layer_i <= len(self.blocks) // 2 and layer_i > 1): # < len(self.blocks) - 1:
                        au_idx = layer_i - 2
                        audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, h, w, 1)
                        audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
                        if sp_size_arg > 1:
                            if pad_size > 0:    
                                audio_cond_tmp = torch.cat([audio_cond_tmp, torch.zeros_like(audio_cond_tmp[:, -1:]).repeat(1, pad_size, 1)], 1)
                            audio_cond_tmp = torch.chunk(audio_cond_tmp, sp_size, dim=1)[get_sequence_parallel_rank()]
                        x = audio_cond_tmp + x

                if streaming_enabled:
                    streaming_cache = streaming_state.layer_caches[layer_i]
                    if (
                        self.training
                        and use_gradient_checkpointing
                        and not update_streaming_cache
                        and torch.is_grad_enabled()
                    ):
                        checkpointed_forward = create_streaming_custom_forward(
                            block,
                            streaming_cache,
                            local_streaming_positions,
                        )
                        if use_gradient_checkpointing_offload:
                            with torch.autograd.graph.save_on_cpu():
                                x = torch.utils.checkpoint.checkpoint(
                                    checkpointed_forward,
                                    x, context, t_mod, freqs,
                                    use_reentrant=False,
                                )
                        else:
                            x = torch.utils.checkpoint.checkpoint(
                                checkpointed_forward,
                                x, context, t_mod, freqs,
                                use_reentrant=False,
                            )
                    else:
                        x, next_cache = block(
                            x,
                            context,
                            t_mod,
                            freqs,
                            streaming_cache=streaming_cache,
                            streaming_positions=local_streaming_positions,
                            streaming_global_positions=streaming_positions,
                            streaming_config=streaming_config,
                            streaming_base_freqs=streaming_base_freqs,
                            streaming_tokens_per_frame=h * w,
                            update_streaming_cache=update_streaming_cache,
                            streaming_compressed=streaming_compressed,
                            streaming_attention_metadata=attention_metadata,
                            prepared_cross_attention=(prepared_cross_attention[layer_i]
                                                      if prepared_cross_attention is not None else None),
                        )
                        if update_streaming_cache:
                            streaming_state.layer_caches[layer_i] = next_cache
                elif self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = block(
                        x, context, t_mod, freqs,
                        prepared_cross_attention=(prepared_cross_attention[layer_i]
                                                  if prepared_cross_attention is not None else None),
                    )
            if tea_cache is not None:
                x_cache = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
                x_cache = x_cache[:, :ori_x_len]
                tea_cache.store(x_cache)
            if (
                streaming_enabled
                and update_streaming_cache
                and not streaming_compressed
            ):
                streaming_state.global_frame_offset += f

        if capture_cache_only:
            return x.new_empty(0)
        if streaming_compressed and update_streaming_cache:
            return x.new_zeros(
                input_shape[0],
                self.out_dim,
                input_shape[2],
                input_shape[3],
                input_shape[4],
            )

        x = self.head(x, t)
        if sp_size_arg > 1:
            # Context Parallel
            x = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
            x = x[:, :ori_x_len]

        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        runtime_args = _runtime_args()
        if hasattr(runtime_args, "model_config"):
            model_config = runtime_args.model_config
            if model_config is not None:
                config.update(model_config)        
        return state_dict, config
