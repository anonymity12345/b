import json
import os

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig


class QwenVisionEncoder(nn.Module):
    """Qwen3-VL vision tower wrapper for role-play reference images."""

    def __init__(self, model_config):
        super().__init__()
        model_name_or_path = getattr(model_config, "pretrain_vision_encoder_weights", None)
        if model_name_or_path in (None, "", "none", "None"):
            raise ValueError("pretrain_vision_encoder_weights must be set when using ref images.")

        self.model_name_or_path = model_name_or_path
        self.visual = self._load_visual_tower(model_name_or_path)

        vision_config = getattr(self.visual, "config", None)
        vision_hidden_size = getattr(
            vision_config,
            "out_hidden_size",
            getattr(model_config, "vision_encoder_hidden_size", getattr(model_config, "hidden_size")),
        )
        self.hidden_size = getattr(model_config, "hidden_size")
        self.projector = nn.Linear(int(vision_hidden_size), self.hidden_size)

    def _load_visual_tower(self, model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        if model_type != "qwen3_vl":
            raise ValueError(
                f"Only Qwen3-VL vision encoder is supported, got model_type={model_type!r} from {model_name_or_path}"
            )
        return self._load_qwen3_vl_visual(model_name_or_path, config)

    def _load_qwen3_vl_visual(self, model_name_or_path, config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        visual = Qwen3VLVisionModel(config.vision_config)
        state_dict = self._load_prefixed_safetensors(model_name_or_path, "model.visual.")
        missing, unexpected = visual.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            raise RuntimeError(f"Unexpected Qwen3-VL visual keys: {unexpected[:10]}")
        if len(missing) > 0:
            raise RuntimeError(f"Missing Qwen3-VL visual keys: {missing[:10]}")
        return visual

    def _load_prefixed_safetensors(self, model_name_or_path, prefix):
        index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                weight_map = json.load(f)["weight_map"]
            shard_files = sorted({filename for key, filename in weight_map.items() if key.startswith(prefix)})
        else:
            shard_files = sorted(filename for filename in os.listdir(model_name_or_path) if filename.endswith(".safetensors"))

        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_name_or_path, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        state_dict[key[len(prefix):]] = f.get_tensor(key)
        if not state_dict:
            raise ValueError(f"No weights with prefix {prefix!r} found in {model_name_or_path}")
        return state_dict

    def reload_pretrained_visual(self, model_name_or_path=None):
        model_name_or_path = model_name_or_path or self.model_name_or_path
        state_dict = self._load_prefixed_safetensors(model_name_or_path, "model.visual.")
        missing, unexpected = self.visual.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            raise RuntimeError(f"Unexpected Qwen3-VL visual keys when reloading: {unexpected[:10]}")
        if len(missing) > 0:
            raise RuntimeError(f"Missing Qwen3-VL visual keys when reloading: {missing[:10]}")
        self.model_name_or_path = model_name_or_path

    @property
    def dtype(self):
        param = next(self.visual.parameters(), None)
        if param is not None:
            return param.dtype
        return next(self.projector.parameters()).dtype

    @staticmethod
    def _ensure_tensors_on_device(module: nn.Module, device: torch.device, dtype=None) -> None:
        """Keep buffers and rotary caches on the same GPU as activations (needed under ZeRO-3)."""
        cache_attrs = ("inv_freq", "cos", "sin", "cos_cached", "sin_cached")
        for submodule in module.modules():
            for name, buffer in list(submodule._buffers.items()):
                if buffer is None:
                    continue
                target_dtype = dtype or buffer.dtype
                if buffer.device != device or buffer.dtype != target_dtype:
                    submodule._buffers[name] = buffer.to(
                        device=device,
                        dtype=target_dtype,
                        non_blocking=True,
                    )
            for attr in cache_attrs:
                if not hasattr(submodule, attr):
                    continue
                value = getattr(submodule, attr)
                if not torch.is_tensor(value):
                    continue
                target_dtype = dtype or value.dtype
                if value.device != device or value.dtype != target_dtype:
                    setattr(
                        submodule,
                        attr,
                        value.to(device=device, dtype=target_dtype, non_blocking=True),
                    )

    def forward(self, pixel_values, image_grid_thw):
        if pixel_values.device.type != "cuda":
            raise RuntimeError(
                f"QwenVisionEncoder expects CUDA inputs, got pixel_values on {pixel_values.device}."
            )

        target_device = pixel_values.device
        dtype = self.dtype
        self._ensure_tensors_on_device(self.visual, target_device, dtype)
        self._ensure_tensors_on_device(self.projector, target_device)

        pixel_values = pixel_values.to(device=target_device, dtype=dtype)
        image_grid_thw = image_grid_thw.to(device=target_device)
        outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(outputs, tuple):
            image_features = outputs[0]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            image_features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            image_features = outputs.last_hidden_state
        else:
            image_features = outputs

        projector_dtype = self.projector.weight.dtype
        image_features = self.projector(
            image_features.to(device=target_device, dtype=projector_dtype)
        )
        return self._split_image_features(image_features, image_grid_thw)

    def _split_image_features(self, image_features, image_grid_thw):
        spatial_merge_size = int(getattr(getattr(self.visual, "config", None), "spatial_merge_size", 2))
        split_sizes = (image_grid_thw.prod(dim=-1) // (spatial_merge_size ** 2)).tolist()
        return list(torch.split(image_features, split_sizes, dim=0))
