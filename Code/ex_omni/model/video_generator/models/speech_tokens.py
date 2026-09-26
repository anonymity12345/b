import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open


class QwenSpeechTokenEmbedding(nn.Module):
    """Embed multi-codebook Qwen TTS speech tokens into continuous features."""

    def __init__(self, tokenizer_path: str, num_codebooks: int = 16, codebook_source: str = "decoder"):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.num_codebooks = int(num_codebooks)
        self.codebook_source = codebook_source
        codebooks = _load_qwen_codebooks(tokenizer_path, self.num_codebooks, codebook_source)
        self.register_buffer("codebooks", codebooks, persistent=False)

    @property
    def embedding_dim(self):
        return int(self.codebooks.shape[-1])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = normalize_speech_tokens(tokens, self.num_codebooks).to(self.codebooks.device)
        tokens = tokens.clamp(0, self.codebooks.shape[1] - 1).long()
        pieces = [self.codebooks[idx].index_select(0, tokens[:, idx]) for idx in range(self.num_codebooks)]
        return torch.stack(pieces, dim=0).sum(dim=0)


def load_speech_tokens(path: str) -> torch.Tensor:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        payload = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        key = "tokens" if "tokens" in data else data.files[0]
        payload = data[key]
    elif suffix in (".pt", ".pth"):
        payload = torch.load(path, map_location="cpu")
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as fin:
            payload = json.load(fin)
    else:
        payload = np.loadtxt(path, dtype=np.int64, delimiter="," if suffix == ".csv" else None)
    return _extract_token_tensor(payload)


def is_speech_token_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".npy", ".npz", ".pt", ".pth", ".json", ".txt", ".csv"}


def encode_speech_file_to_tokens(tokenizer_path: str, audio_path: str, device: str = "cuda:0") -> torch.Tensor:
    tokenizer = load_qwen_speech_tokenizer(tokenizer_path, device=device)
    encoded = tokenizer.encode(audio_path)
    return _extract_token_tensor(encoded)


def load_qwen_speech_tokenizer(tokenizer_path: str, device: str = "cuda:0"):
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Please install qwen-tts to encode wav/audio inputs online.") from exc
    from ex_omni.hub import materialize_pretrained_reference

    tokenizer_path = materialize_pretrained_reference(tokenizer_path)
    return Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map=device)


def extract_encoded_speech_tokens(encoded: Any) -> torch.Tensor:
    return _extract_token_tensor(encoded)


def repeat_tokens_to_frames(features: torch.Tensor, num_frames: int, repeat_factor: int) -> torch.Tensor:
    features = features.repeat_interleave(repeat_factor, dim=0)
    return features[:num_frames]


def pad_features_with_zeros(features: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Pad frame features with null conditioning after learned projections."""
    if features.shape[0] >= num_frames:
        return features[:num_frames]
    pad_shape = (num_frames - features.shape[0], *features.shape[1:])
    return torch.cat([features, features.new_zeros(pad_shape)], dim=0)


def normalize_speech_tokens(tokens: torch.Tensor, num_codebooks: int) -> torch.Tensor:
    tokens = tokens.detach().cpu().long()
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(-1)
    if tokens.ndim != 2:
        raise ValueError(f"Speech tokens must be a 2D tensor, got shape {tuple(tokens.shape)}.")
    if tokens.shape[0] == num_codebooks and tokens.shape[1] != num_codebooks:
        tokens = tokens.transpose(0, 1)
    if tokens.shape[1] < num_codebooks:
        raise ValueError(f"Expected at least {num_codebooks} codebooks, got shape {tuple(tokens.shape)}.")
    return tokens[:, :num_codebooks].contiguous()


def _load_qwen_codebooks(tokenizer_path: str, num_codebooks: int, codebook_source: str) -> torch.Tensor:
    model_path = os.path.join(tokenizer_path, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Qwen speech tokenizer weights not found: {model_path}")
    with safe_open(model_path, framework="pt", device="cpu") as fin:
        if codebook_source == "encoder":
            key_pairs = [
                (
                    f"encoder.quantizer.acoustic_residual_vector_quantizer.layers.{idx}.codebook.embed_sum",
                    f"encoder.quantizer.acoustic_residual_vector_quantizer.layers.{idx}.codebook.cluster_usage",
                )
                for idx in range(num_codebooks)
            ]
        elif codebook_source == "decoder":
            key_pairs = [
                (
                    "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
                    "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
                )
            ]
            key_pairs.extend(
                (
                    f"decoder.quantizer.rvq_rest.vq.layers.{idx}._codebook.embedding_sum",
                    f"decoder.quantizer.rvq_rest.vq.layers.{idx}._codebook.cluster_usage",
                )
                for idx in range(num_codebooks - 1)
            )
        else:
            raise ValueError("speech_token_codebook_source must be `decoder` or `encoder`.")

        available_keys = set(fin.keys())
        missing_keys = [
            key
            for key_pair in key_pairs
            for key in key_pair
            if key not in available_keys
        ]
        if missing_keys:
            raise KeyError(f"Missing Qwen codebook tensors in {model_path}: {missing_keys}")

        codebooks = []
        for embedding_key, usage_key in key_pairs:
            embedding_sum = fin.get_tensor(embedding_key).float()
            cluster_usage = fin.get_tensor(usage_key).float()
            if embedding_sum.ndim != 2 or cluster_usage.ndim != 1:
                raise ValueError(
                    f"Invalid Qwen codebook shapes for {embedding_key}: "
                    f"embedding_sum={tuple(embedding_sum.shape)}, "
                    f"cluster_usage={tuple(cluster_usage.shape)}."
                )
            if embedding_sum.shape[0] != cluster_usage.shape[0]:
                raise ValueError(
                    f"Qwen codebook size mismatch for {embedding_key}: "
                    f"{embedding_sum.shape[0]} embeddings vs {cluster_usage.shape[0]} usage values."
                )
            codebook = embedding_sum / cluster_usage.clamp(min=1e-5).unsqueeze(1)
            if not torch.isfinite(codebook).all():
                raise ValueError(f"Qwen codebook contains non-finite values: {embedding_key}")
            codebooks.append(codebook)

        expected_shape = codebooks[0].shape
        if any(codebook.shape != expected_shape for codebook in codebooks[1:]):
            shapes = [tuple(codebook.shape) for codebook in codebooks]
            raise ValueError(f"Qwen codebook shapes must match for summation, got {shapes}.")
        return torch.stack(codebooks, dim=0)


def _extract_token_tensor(payload: Any) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload.cpu()
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(payload)
    if isinstance(payload, dict):
        for key in ("speech_tokens", "tokens", "audio_tokens", "codes", "input_ids"):
            if key in payload:
                return _extract_token_tensor(payload[key])
        for value in payload.values():
            try:
                return _extract_token_tensor(value)
            except (TypeError, ValueError):
                continue
        raise ValueError("Token dict must contain one of: speech_tokens, tokens, audio_tokens, codes, input_ids.")
    if isinstance(payload, (list, tuple)):
        if len(payload) == 1:
            return _extract_token_tensor(payload[0])
        if all(_is_array_like(value) for value in payload):
            arrays = [_to_numpy(value) for value in payload]
            try:
                return torch.from_numpy(np.stack(arrays, axis=0))
            except ValueError:
                return torch.from_numpy(np.asarray(arrays))
        for value in payload:
            try:
                return _extract_token_tensor(value)
            except (TypeError, ValueError):
                continue
    for attr in ("speech_tokens", "tokens", "audio_tokens", "codes", "input_ids"):
        if hasattr(payload, attr):
            return _extract_token_tensor(getattr(payload, attr))
    return torch.tensor(payload)


def _is_array_like(value: Any) -> bool:
    return isinstance(value, (torch.Tensor, np.ndarray, list, tuple)) and not isinstance(value, (str, bytes))


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


