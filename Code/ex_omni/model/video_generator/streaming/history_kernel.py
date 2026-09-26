"""Fuse history selection, complex64 RoPE and K/V packing for graph inputs."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(FK, FV, CK, CV, SK, SV, FI, CI, FREQ, OK, OV,
          nfull, ncompressed, nsink, ntokens,
          DIM: tl.constexpr, HEAD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = i // (DIM // 2)
    pair = i % (DIM // 2)
    valid = token < ntokens
    sink = valid & (token < nsink)
    comp = valid & (token >= nsink) & (token < nsink + ncompressed)
    full = valid & (token >= nsink + ncompressed) & (token < nsink + ncompressed + nfull)
    ci = tl.load(CI + token - nsink, comp, other=0)
    fi = tl.load(FI + token - nsink - ncompressed, full, other=0)
    co = ci * DIM + pair * 2
    fo = fi * DIM + pair * 2
    so = token * DIM + pair * 2
    kr = tl.load(CK + co, comp, other=0).to(tl.float32) + tl.load(FK + fo, full, other=0).to(tl.float32)
    ki = tl.load(CK + co + 1, comp, other=0).to(tl.float32) + tl.load(FK + fo + 1, full, other=0).to(tl.float32)
    vr = tl.load(CV + co, comp, other=0) + tl.load(FV + fo, full, other=0)
    vi = tl.load(CV + co + 1, comp, other=0) + tl.load(FV + fo + 1, full, other=0)
    kr += tl.load(SK + so, sink, other=0).to(tl.float32)
    ki += tl.load(SK + so + 1, sink, other=0).to(tl.float32)
    vr += tl.load(SV + so, sink, other=0)
    vi += tl.load(SV + so + 1, sink, other=0)
    freq_offset = token * HEAD + (pair % (HEAD // 2)) * 2
    cosine = tl.load(FREQ + freq_offset, valid, other=0)
    sine = tl.load(FREQ + freq_offset + 1, valid, other=0)
    # Match PyTorch CUDA complex64 multiplication: one rounded product plus
    # an explicit FMA. Two separate products can change BF16 rounding and
    # those rare differences accumulate during autoregressive video.
    tl.store(OK + i * 2, tl.fma(kr, cosine, -(ki * sine)), valid)
    tl.store(OK + i * 2 + 1, tl.fma(kr, sine, ki * cosine), valid)
    tl.store(OV + i * 2, vr, valid)
    tl.store(OV + i * 2 + 1, vi, valid)


def pack_history(cache, full_indices, compressed_indices, frequencies, num_heads,
                 *, include_sink=True):
    if cache.key.shape[0] != 1 or not cache.key.is_contiguous():
        raise ValueError('Fused history preparation requires contiguous batch-one caches')
    empty = torch.empty(0, device=cache.key.device, dtype=torch.int64)
    fi = full_indices if full_indices is not None else empty
    ci = compressed_indices if compressed_indices is not None else empty
    ck = cache.compressed_key if ci.numel() else cache.key
    cv = cache.compressed_value if ci.numel() else cache.value
    if cache.sink_key is None or cache.sink_value is None:
        raise ValueError('Fused history requires the prefilled reference sink')
    nsink = cache.sink_key.shape[1] if include_sink else 0
    sk = cache.sink_key
    sv = cache.sink_value
    for tensor in (cache.value, ck, cv, sk, sv):
        if not tensor.is_contiguous():
            raise ValueError('Fused history preparation requires contiguous K/V tensors')
    dim, tokens = cache.key.shape[-1], frequencies.shape[0]
    if nsink + fi.numel() + ci.numel() > tokens:
        raise ValueError('History length exceeds the supplied RoPE frequencies')
    key = cache.key.new_empty(1, tokens, dim)
    value = torch.empty_like(key)
    freqs = torch.view_as_real(frequencies).contiguous()
    _pack[(triton.cdiv(tokens * dim // 2, 256),)](
        cache.key, cache.value, ck, cv, sk, sv, fi, ci, freqs, key, value,
        fi.numel(), ci.numel(), nsink, tokens, DIM=dim, HEAD=dim // num_heads,
        BLOCK=256, enable_fp_fusion=False)
    return key, value
