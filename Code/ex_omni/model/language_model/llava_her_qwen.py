
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
import logging
import numpy as np
import gc
from dataclasses import dataclass
from safetensors import safe_open
from collections import OrderedDict, defaultdict

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3Model, Qwen3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ex_omni.model.llava_her_arch import LlavaHerMetaForCausalLM
from ex_omni.model.speech_generator.generation import GenerationWithCTC

from ..speech_encoder.builder import build_speech_encoder
from ..speech_projector.builder import build_speech_projector
from ..speech_generator.builder import build_speech_generator as build_ar_speech_generator
from ..vision_encoder.builder import build_vision_encoder
from ex_omni.utils import rank0_print
import time


logger = logging.getLogger("Ex-Omni")


@dataclass
class OmniCausalLMOutputWithPast(CausalLMOutputWithPast):
    llm_loss: Optional[torch.FloatTensor] = None
    speech_loss: Optional[torch.FloatTensor] = None
    speech_main_loss: Optional[torch.FloatTensor] = None
    speech_residual_loss: Optional[torch.FloatTensor] = None
    total_loss: Optional[torch.FloatTensor] = None


class LlavaHerQwenConfig(Qwen3Config):
    model_type = "llava_her_qwen3"


class LlavaHerQwen3Model(Qwen3Model):
    config_class = LlavaHerQwenConfig
    def __init__(self, config: Qwen3Config):
        super(LlavaHerQwen3Model, self).__init__(config)

def load_safetensors_state_dict(folder_path):
    """
    从safetensors文件加载状态字典
    
    Args:
        folder_path: 包含safetensors文件的文件夹路径
        
    Returns:
        OrderedDict: 加载的状态字典
    """
    state_dict = OrderedDict()

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(folder_path, file_name)
            rank0_print(f"Loading {file_path}...")
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    state_dict[key] = tensor

    return state_dict


def iter_safetensors_paths(folder_path):
    # A checkpoint directory can also contain a separate Streaming video adapter.
    # For sharded models, only load files belonging to the dialogue checkpoint.
    index_path = os.path.join(folder_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            file_names = sorted(set(json.load(handle)["weight_map"].values()))
    else:
        file_names = sorted(
            name for name in os.listdir(folder_path) if name.endswith(".safetensors")
        )
    for file_name in file_names:
        yield os.path.join(folder_path, file_name)


def _module_uses_deepspeed_zero3(module: nn.Module) -> bool:
    return any(hasattr(param, "ds_id") for param in module.parameters())


def _load_state_dict_into_zero3_model(module: nn.Module, state_dict, strict: bool = False):
    from deepspeed import zero

    metadata = getattr(state_dict, "_metadata", None)
    state_dict_keys = list(state_dict.keys())
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    def load(module_to_load: nn.Module, prefix: str = ""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        direct_zero3_params = [
            param
            for name, param in module_to_load.named_parameters(recurse=False)
            if prefix + name in state_dict and hasattr(param, "ds_id")
        ]

        if direct_zero3_params:
            with zero.GatheredParameters(direct_zero3_params, modifier_rank=0):
                if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    module_to_load._load_from_state_dict(
                        state_dict,
                        prefix,
                        local_metadata,
                        True,
                        missing_keys,
                        unexpected_keys,
                        error_msgs,
                    )
        else:
            module_to_load._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        for child_name, child in module_to_load._modules.items():
            if child is None:
                continue
            child_prefix = prefix + child_name + "."
            if any(key.startswith(child_prefix) for key in state_dict_keys):
                load(child, child_prefix)

    load(module)

    if error_msgs:
        raise RuntimeError("\n".join(error_msgs))

    if strict:
        module_state_keys = set(module.state_dict().keys())
        loaded_state_keys = set(state_dict_keys)
        missing_keys.extend(sorted(module_state_keys - loaded_state_keys))
        unexpected_keys.extend(sorted(loaded_state_keys - module_state_keys))

    return missing_keys, unexpected_keys


def load_state_dict_maybe_zero3(module: nn.Module, state_dict, strict: bool = False):
    if not _module_uses_deepspeed_zero3(module):
        return module.load_state_dict(state_dict, strict=strict)

    return _load_state_dict_into_zero3_model(module, state_dict, strict=strict)


def load_safetensors_into_model(model, folder_path):
    has_speech_encoder = False
    has_speech_generator = False
    has_speech_token_embedding = False

    for file_path in iter_safetensors_paths(folder_path):
        rank0_print(f"Loading {file_path}...")
        shard_state_dict = OrderedDict()
        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
            has_speech_encoder = has_speech_encoder or any("speech_encoder" in key for key in keys)
            has_speech_generator = has_speech_generator or any("speech_generator" in key for key in keys)
            has_speech_token_embedding = has_speech_token_embedding or any("speech_token_embedding" in key for key in keys)
            for key in keys:
                shard_state_dict[key] = f.get_tensor(key)

        load_state_dict_maybe_zero3(model, shard_state_dict, strict=False)
        del shard_state_dict
        gc.collect()

    return {
        "has_speech_encoder": has_speech_encoder,
        "has_speech_generator": has_speech_generator,
        "has_speech_token_embedding": has_speech_token_embedding,
    }

# 定义一个辅助函数来安全清零
def safe_zero_loss(ref_tensor):
    # 创建一个 0.0
    zero = torch.tensor(0.0, device=ref_tensor.device, dtype=ref_tensor.dtype)
    # 这一步是核心：loss = 0 + 0 * ref_tensor
    # 这样 loss 的值是 0，但它在计算图上依赖于 ref_tensor
    return zero + 0.0 * torch.nan_to_num(ref_tensor, nan=0.0, posinf=0.0, neginf=0.0).sum()


def sanitize_loss(loss):
    if loss is None:
        return None
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


class LlavaHerQwen3ForCausalLM(Qwen3ForCausalLM, LlavaHerMetaForCausalLM):
    config_class = LlavaHerQwenConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if getattr(config, "init_multimodal_modules_in_constructor", False):
            self.initialize_speech_modules()

    def initialize_speech_modules(self):
        needs_vision_encoder = getattr(self.config, "pretrain_vision_encoder_weights", None) not in (None, "", "none", "None")
        if (
            getattr(self.model, "speech_encoder", None) is not None
            and getattr(self.model, "speech_projector", None) is not None
            and getattr(self.model, "speech_generator", None) is not None
            and (not needs_vision_encoder or getattr(self.model, "vision_encoder", None) is not None)
        ):
            return
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype

        speech_encoder = build_speech_encoder(self.config)
        self.model.speech_encoder = speech_encoder.to(device=device, dtype=dtype)
        self.config.speech_encoder_hidden_size = self.model.speech_encoder.hidden_size

        self.model.speech_projector = build_speech_projector(self.config).to(
            device=device,
            dtype=dtype
        )

        self.model.speech_generator = build_ar_speech_generator(self.config).to(
            device=device,
            dtype=dtype
        )

        if needs_vision_encoder:
            self.model.vision_encoder = build_vision_encoder(self.config).to(
                device=device,
                dtype=dtype,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        重写 from_pretrained 方法，自动加载语音模块权重
        """
        # 首先使用父类的 from_pretrained 方法（此时已经包含了所有权重）
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # 初始化语音模块
        model.initialize_speech_modules()
        
        # 从已加载的模型中提取语音模块权重（更高效）
        model.load_existing_weights(
            pretrained_model_name_or_path,
            load_speech_weights=True,
        )

        return model

    @classmethod
    def from_pretrained_hf_auto_full(cls, pretrained_model_name_or_path, *args, **kwargs):
        if args:
            raise TypeError("from_pretrained_hf_auto_full only supports keyword arguments after model path")
        requested_device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        no_split_module_classes = kwargs.pop("no_split_module_classes", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        config.model_name_or_path = pretrained_model_name_or_path
        model = cls.build_full_model_from_pretrained(
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs,
        )
        return cls._dispatch_hf_auto_full_model(
            model,
            requested_device_map=requested_device_map,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
            offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            dtype=torch_dtype,
        )

    @staticmethod
    def _dispatch_hf_auto_full_model(
        model,
        requested_device_map=None,
        max_memory=None,
        no_split_module_classes=None,
        offload_folder=None,
        offload_buffers=False,
        dtype=None,
    ):
        if requested_device_map in (None, "", "cpu"):
            return model
        if requested_device_map == "cuda":
            return model.to("cuda")
        if isinstance(requested_device_map, str) and requested_device_map.startswith("cuda:"):
            return model.to(requested_device_map)
        if isinstance(requested_device_map, dict) and set(requested_device_map.keys()) == {""}:
            target_device = requested_device_map[""]
            return model.to(target_device)

        try:
            from accelerate import dispatch_model, infer_auto_device_map
        except ImportError as exc:
            raise ImportError("load_method='hf_auto_full' with device_map auto requires accelerate.") from exc

        dispatch_kwargs = {}
        if offload_folder is not None:
            dispatch_kwargs["offload_dir"] = offload_folder
        dispatch_kwargs["offload_buffers"] = offload_buffers

        if isinstance(requested_device_map, dict):
            device_map = requested_device_map
        else:
            if requested_device_map not in ("auto", "balanced", "balanced_low_0", "sequential"):
                raise ValueError(f"Unsupported hf_auto_full device_map: {requested_device_map!r}")
            no_split_module_classes = no_split_module_classes or getattr(model, "_no_split_modules", None)
            if no_split_module_classes is None:
                no_split_module_classes = ["Qwen3DecoderLayer", "Qwen3TTSDecoderLayer"]
            elif "Qwen3TTSDecoderLayer" not in no_split_module_classes:
                no_split_module_classes = list(no_split_module_classes) + ["Qwen3TTSDecoderLayer"]
            infer_kwargs = {
                "no_split_module_classes": no_split_module_classes,
            }
            if max_memory is not None:
                infer_kwargs["max_memory"] = max_memory
            if dtype is not None:
                infer_kwargs["dtype"] = dtype
            device_map = infer_auto_device_map(model, **infer_kwargs)
            rank0_print(f"hf_auto_full inferred device_map: {device_map}")

        return dispatch_model(model, device_map=device_map, **dispatch_kwargs)

    @classmethod
    def build_full_model_from_pretrained(cls, config, *args, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        original_dtype = torch.get_default_dtype()
        override_default_dtype = torch_dtype in (torch.float16, torch.bfloat16, torch.float32)
        if override_default_dtype:
            torch.set_default_dtype(torch_dtype)
        try:
            try:
                from transformers import initialization as transformers_init

                no_init_weights = transformers_init.no_init_weights
            except (ImportError, AttributeError):
                from transformers.modeling_utils import no_init_weights

            # Every backbone parameter is immediately replaced by the full
            # safetensors checkpoint below. Skip the otherwise multi-minute
            # random initialization; multimodal modules are built afterwards.
            with no_init_weights():
                model = cls(config)
        finally:
            if override_default_dtype:
                torch.set_default_dtype(original_dtype)

        model.initialize_speech_modules()
        
        rank0_print(f"Model initialized. Loading model parameters from {config.model_name_or_path}")
        load_stats = load_safetensors_into_model(model, config.model_name_or_path)
        if (
            getattr(config, "force_reload_pretrain_vision_visual", False)
            and hasattr(model.model, "vision_encoder")
            and hasattr(model.model.vision_encoder, "reload_pretrained_visual")
        ):
            model.model.vision_encoder.reload_pretrained_visual(config.pretrain_vision_encoder_weights)
            rank0_print(f"Reloaded vision_encoder.visual from {config.pretrain_vision_encoder_weights}")
        if load_stats["has_speech_encoder"] and (not config.load_pretrain_speech_encoder_weights):
            rank0_print(f"Loaded speech encoder from {config.model_name_or_path}")
        else:
            model.model.speech_encoder.reload_pretrained_weights(config.pretrain_speech_encoder_weights)
            rank0_print(f"Loaded speech encoder from {config.pretrain_speech_encoder_weights}")

        if load_stats["has_speech_generator"]:
            rank0_print(f"Loaded speech generator from {config.model_name_or_path}")
        else:
            rank0_print(f"Initialized speech generator from Qwen3-TTS talker: {config.pretrain_qwen_tts_weights}")

        rank0_print(f"Model parameters loaded")

        return model


    def initialize_model_args(self, model_args):
        self.model_args = model_args

    def get_model(self):
        return self.model

    def get_model_args(self):
        try:
            return self.model_args
        except:
            assert False, "Please call function initialize_model_args first."

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        re_tgt_units: Optional[torch.LongTensor] = None,
        text_tokens: Optional[torch.LongTensor] = None,
        speech_text_labels: Optional[torch.LongTensor] = None,
        ref_audio: Optional[torch.FloatTensor] = None,
        ref_audio_lengths: Optional[torch.LongTensor] = None,
        ref_audio_waveform: Optional[torch.FloatTensor] = None,
        ref_audio_waveform_lengths: Optional[torch.LongTensor] = None,
        has_ref_audio: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        has_ref_image: Optional[torch.Tensor] = None,
        speech_chunk_counts: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        task: Optional[str] = None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                speech_text_labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths,
                pixel_values,
                image_grid_thw,
                speech_text_labels,
                speech_chunk_counts=speech_chunk_counts,
            )

        llama_output=super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **kwargs)

        if not getattr(self.config, "inference", False):
            # 创建 mask
            speech_gen_tasks = {"tts", "s2sv"}
            
            speech_gen_mask = torch.tensor(
                [1.0 if t in speech_gen_tasks else 0.0 for t in task],
                device=llama_output.loss.device
            )
            
            llama_loss = sanitize_loss(llama_output.loss)
            total_loss = torch.zeros((), device=llama_loss.device, requires_grad=True)
            llm_loss_to_log = safe_zero_loss(llama_loss)
            speech_loss_to_log = safe_zero_loss(llama_loss)
            speech_main_loss_to_log = safe_zero_loss(llama_loss)
            speech_residual_loss_to_log = safe_zero_loss(llama_loss)
            # 仅在未冻结 backbone 时加入 LLM loss
            if (not self.model_args.freeze_backbone) or (not self.model_args.freeze_speech_projector):
                total_loss = total_loss + llama_loss
                llm_loss_to_log = llama_loss
            
            if not self.model_args.freeze_speech_generator:
                roleplay_embedding = self.build_roleplay_embedding(
                    ref_audio_waveform=ref_audio_waveform,
                    ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                    has_ref_audio=has_ref_audio,
                )
                speech_gen_output = self.get_model().speech_generator(
                    llama_output['hidden_states'][-1],
                    speech_text_labels if speech_text_labels is not None else labels,
                    tgt_units,
                    text_tokens,
                    embedding=roleplay_embedding,
                )
                speech_loss_values = sanitize_loss(speech_gen_output.loss)
                speech_loss = speech_loss_values * speech_gen_mask
                # 避免除以0
                count = speech_gen_mask.sum().clamp(min=1.0)
                speech_loss_to_log = speech_loss.sum() / count
                speech_main_loss = sanitize_loss(speech_gen_output.main_loss) * speech_gen_mask
                speech_residual_loss = sanitize_loss(speech_gen_output.residual_loss) * speech_gen_mask
                speech_main_loss_to_log = speech_main_loss.sum() / count
                speech_residual_loss_to_log = speech_residual_loss.sum() / count
                total_loss = total_loss + speech_loss_to_log
            
            loss = total_loss

        else:
            loss = sanitize_loss(llama_output.loss)
            llm_loss_to_log = loss
            speech_loss_to_log = None
            speech_main_loss_to_log = None
            speech_residual_loss_to_log = None

        return OmniCausalLMOutputWithPast(
            loss=loss,
            logits=llama_output.logits,
            past_key_values=llama_output.past_key_values,
            hidden_states=llama_output.hidden_states,
            attentions=llama_output.attentions,
            llm_loss=llm_loss_to_log,
            speech_loss=speech_loss_to_log,
            speech_main_loss=speech_main_loss_to_log,
            speech_residual_loss=speech_residual_loss_to_log,
            total_loss=loss,
        )

    def build_roleplay_embedding(
        self,
        ref_audio_waveform=None,
        ref_audio_waveform_lengths=None,
        has_ref_audio=None,
    ):
        if ref_audio_waveform is None:
            return None

        speech_generator = self.get_model().speech_generator
        return speech_generator.prepare_roleplay_embedding(
            ref_audio_waveform=ref_audio_waveform,
            ref_audio_waveform_lengths=ref_audio_waveform_lengths,
            has_ref_audio=has_ref_audio,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        streaming_unit_gen=False,
        faster_infer=False,
        text_tokens=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_timing_callback = kwargs.pop(
            "generation_timing_callback", None
        )
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        ref_audio = kwargs.pop("ref_audio", None)
        ref_audio_lengths = kwargs.pop("ref_audio_lengths", None)
        ref_audio_waveform = kwargs.pop("ref_audio_waveform", None)
        ref_audio_waveform_lengths = kwargs.pop("ref_audio_waveform_lengths", None)
        has_ref_audio = kwargs.pop("has_ref_audio", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        has_ref_image = kwargs.pop("has_ref_image", None)
        speech_chunk_counts = kwargs.pop("speech_chunk_counts", None)
        response_open_token_id = kwargs.pop("response_open_token_id", None)
        response_close_token_id = kwargs.pop("response_close_token_id", None)
        output_speech = bool(kwargs.pop("output_speech", True))
        max_speech_tokens = kwargs.pop("max_speech_tokens", 750)
        speech_do_sample = kwargs.pop("speech_do_sample", True)
        speech_top_k = kwargs.pop("speech_top_k", 50)
        speech_top_p = kwargs.pop("speech_top_p", 1.0)
        speech_temperature = kwargs.pop("speech_temperature", 0.9)
        speech_repetition_penalty = kwargs.pop(
            "speech_repetition_penalty", 1.05
        )
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None or pixel_values is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths,
                pixel_values,
                image_grid_thw,
                speech_chunk_counts=speech_chunk_counts,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if faster_infer or not output_speech:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            ), None
        else:
            llm_generate_start = time.perf_counter()
            outputs = GenerationWithCTC.generate(
                self,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                streaming_unit_gen=streaming_unit_gen,
                **kwargs
            )
            if generation_timing_callback is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            llm_generate_seconds = time.perf_counter() - llm_generate_start
            logger.info(
                "Generation phase complete: "
                f"llm_text={llm_generate_seconds:.3f}s, "
                f"sequence_tokens={int(outputs.sequences.shape[-1])}"
            )
            if generation_timing_callback is not None:
                generation_timing_callback(
                    {
                        "phase": "llm_text",
                        "seconds": llm_generate_seconds,
                        "sequence_tokens": int(outputs.sequences.shape[-1]),
                    }
                )

            hidden_states = outputs['hidden_states']
            hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)

            # Speech conditioning is restricted to response tokens. Atomic
            # protocol IDs make selection exact and independent of decoding.
            if response_open_token_id is None or response_close_token_id is None:
                raise ValueError("Atomic response boundary token IDs are required for speech generation")
            sequence_ids = outputs.sequences[0].tolist()
            open_positions = [i for i, token_id in enumerate(sequence_ids) if token_id == response_open_token_id]
            close_positions = [i for i, token_id in enumerate(sequence_ids) if token_id == response_close_token_id]
            if len(open_positions) != 1 or len(close_positions) != 1:
                logger.error(
                    "Strict response-only speech generation skipped: invalid protocol boundaries "
                    f"open={open_positions}, close={close_positions}"
                )
                return outputs.sequences, None
            response_start = open_positions[0] + 1
            response_end = close_positions[0]
            if response_start >= response_end:
                logger.error("Strict response-only speech generation skipped: empty response span")
                return outputs.sequences, None
            if hidden_states.shape[1] != outputs.sequences.shape[1]:
                raise RuntimeError(
                    "Generated hidden states/token alignment mismatch: "
                    f"hidden={hidden_states.shape[1]}, tokens={outputs.sequences.shape[1]}"
                )
            response_hidden_states = hidden_states[:, response_start:response_end, :]
            response_text_ids = outputs.sequences[:, response_start:response_end]

            speech_generator = self.get_model().speech_generator
            target_device = next(speech_generator.parameters()).device
            target_dtype = next(speech_generator.parameters()).dtype

            if response_hidden_states.device != target_device or response_hidden_states.dtype != target_dtype:
                response_hidden_states = response_hidden_states.to(device=target_device, dtype=target_dtype)

            response_text_ids = response_text_ids.to(device=target_device)
            eos_token_id = self.config.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            response_text_ids = torch.cat([
                response_text_ids,
                torch.full(
                    (response_text_ids.shape[0], 1),
                    int(eos_token_id),
                    dtype=response_text_ids.dtype,
                    device=target_device,
                ),
            ], dim=1)

            roleplay_embedding = self.build_roleplay_embedding(
                ref_audio_waveform=ref_audio_waveform,
                ref_audio_waveform_lengths=ref_audio_waveform_lengths,
                has_ref_audio=has_ref_audio,
            )
            speech_generate_start = time.perf_counter()
            logger.info(
                "Generation phase start: speech_generator.predict "
                f"response_tokens={int(response_end - response_start)}, "
                f"full_assistant_tokens={int(outputs.sequences.shape[-1])}"
            )
            predict_result = self.get_model().speech_generator.predict(
                response_hidden_states,
                response_text_ids,
                embedding=roleplay_embedding,
                max_speech_tokens=max_speech_tokens,
                do_sample=speech_do_sample,
                top_k=speech_top_k,
                top_p=speech_top_p,
                temperature=speech_temperature,
                repetition_penalty=speech_repetition_penalty,
                return_hidden_states=False,
            )
            if generation_timing_callback is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            speech_generate_seconds = time.perf_counter() - speech_generate_start
            if isinstance(predict_result, tuple):
                speech_pred_str = predict_result[0]
                speech_frames = len(predict_result[1]) if isinstance(predict_result[1], list) else -1
            else:
                speech_pred_str = predict_result
                speech_frames = len(speech_pred_str) if isinstance(speech_pred_str, list) else -1
            logger.info(
                "Generation phase complete: "
                f"speech_generator={speech_generate_seconds:.3f}s, "
                f"speech_frames={speech_frames}"
            )
            if generation_timing_callback is not None:
                generation_timing_callback(
                    {
                        "phase": "speech_generator",
                        "seconds": speech_generate_seconds,
                        "speech_frames": speech_frames,
                    }
                )

            return outputs.sequences, speech_pred_str

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        if pixel_values is not None:
            inputs['pixel_values'] = pixel_values
            inputs['image_grid_thw'] = image_grid_thw
        return inputs


    def load_existing_weights(self, pretrained_model_path, load_speech_weights=True):
        print(f"Extracting speech weights from loaded model...")
        state_dict = self.load_safetensors_state_dict(pretrained_model_path)

        if load_speech_weights:
            self.load_speech_weights(state_dict)

    def load_speech_weights(self, state_dict):
        """
        从已加载的模型中提取语音模块权重并重新分配
        适用于模型已经通过 from_pretrained 加载的情况
        """
        # 提取语音模块权重
        speech_weights = self._extract_speech_weights_from_state_dict(state_dict)

        # 重新加载权重到对应模块（确保正确映射）
        self._load_speech_module_weights(speech_weights)
    
    def load_safetensors_state_dict(self, folder_path):
        """
        从safetensors文件加载状态字典
        
        Args:
            folder_path: 包含safetensors文件的文件夹路径
            
        Returns:
            OrderedDict: 加载的状态字典
        """
        state_dict = OrderedDict()

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".safetensors"):
                file_path = os.path.join(folder_path, file_name)
                print(f"Loading {file_path}...")
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        state_dict[key] = tensor

        return state_dict
    
    def _extract_speech_weights_from_state_dict(self, full_state_dict):
        """
        从完整的状态字典中提取语音模块权重
        
        Args:
            full_state_dict: 完整的模型状态字典
            
        Returns:
            提取的语音模块权重字典
        """
        speech_weights = {
            'speech_encoder': {},
            'speech_projector': {},
            'speech_generator': {}
        }
        
        # 定义各模块的键名前缀
        speech_module_prefixes = {
            'speech_encoder': ['model.model.speech_encoder.', 'model.speech_encoder.', 'speech_encoder.'],
            'speech_projector': ['model.model.speech_projector.','model.speech_projector.', 'speech_projector.'],
            'speech_generator': ['model.model.speech_generator.','model.speech_generator.', 'speech_generator.']
        }
        
        # 遍历完整状态字典，提取语音模块权重
        for key, value in full_state_dict.items():
            matched = False
            for module_name, prefixes in speech_module_prefixes.items():
                if matched:
                    break
                for prefix in prefixes:
                    if key.startswith(prefix):
                        # 移除前缀，得到模块内的键名
                        module_key = key[len(prefix):]
                        speech_weights[module_name][module_key] = value
                        print(f"✓ Extracted {module_name} weight: {key} -> {module_key}")
                        matched = True
                        break
        
        # 报告提取结果
        for module_name, weights in speech_weights.items():
            if weights:
                print(f"✓ Extracted {len(weights)} weights for {module_name}")
            else:
                print(f"✗ No weights found for {module_name}")
        
        return speech_weights
    
    def _load_speech_module_weights(self, loaded_weights):
        """
        将加载的权重应用到对应的模块
        
        Args:
            loaded_weights: 加载的权重字典
        """
        # 加载 speech_encoder
        if 'speech_encoder' in loaded_weights and hasattr(self.model, 'speech_encoder'):
            try:
                missing_keys, unexpected_keys = load_state_dict_maybe_zero3(
                    self.model.speech_encoder,
                    loaded_weights['speech_encoder'], strict=False)
                if missing_keys:
                    print(f"Missing keys in speech_encoder: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in speech_encoder: {unexpected_keys}")
                print("✓ speech_encoder weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load speech_encoder weights: {e}")
        
        # 加载 speech_projector
        if 'speech_projector' in loaded_weights and hasattr(self.model, 'speech_projector'):
            try:
                missing_keys, unexpected_keys = load_state_dict_maybe_zero3(
                    self.model.speech_projector,
                    loaded_weights['speech_projector'], strict=False)
                if missing_keys:
                    print(f"Missing keys in speech_projector: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in speech_projector: {unexpected_keys}")
                print("✓ speech_projector weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load speech_projector weights: {e}")
        
        if 'speech_generator' in loaded_weights and hasattr(self.model, 'speech_generator'):
            try:
                missing_keys, unexpected_keys = load_state_dict_maybe_zero3(
                    self.model.speech_generator,
                    loaded_weights['speech_generator'], strict=False)
                if missing_keys:
                    print(f"Missing keys in speech_generator: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in speech_generator: {unexpected_keys}")
                print("✓ speech_generator weights loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load speech_generator weights: {e}")

AutoConfig.register("llava_her_qwen3", LlavaHerQwenConfig)
AutoModelForCausalLM.register(LlavaHerQwenConfig, LlavaHerQwen3ForCausalLM)
