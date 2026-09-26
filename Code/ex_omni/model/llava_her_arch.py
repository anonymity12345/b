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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import copy
import torch.distributed as dist

from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector
from .speech_generator.builder import build_speech_generator as build_ctc_speech_generator
from .speech_generator.builder import build_speech_generator as build_ar_speech_generator
from .vision_encoder.builder import build_vision_encoder

from ex_omni.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX, IMAGE_TOKEN_INDEX


class LlavaHerMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().speech_encoder
    
    def get_speech_projector(self):
        return self.get_model().speech_projector

    def get_vision_encoder(self):
        return getattr(self.get_model(), "vision_encoder", None)

    def get_language_input_device(self):
        embed_tokens = self.get_model().embed_tokens
        param = next(embed_tokens.parameters(), None)
        if param is not None:
            return param.device
        return self.device
    
    def encode_speech(self, speech, speech_lengths):
        speech_encoder = self.get_speech_encoder()
        encoder_outs = speech_encoder(speech, speech_lengths=speech_lengths)
        speech_lengths = speech_encoder.get_output_lengths(speech_lengths)

        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            projector_param = next(speech_projector.parameters(), None)
            if projector_param is not None:
                encoder_outs = encoder_outs.to(
                    device=projector_param.device,
                    dtype=projector_param.dtype,
                )
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f'Unknown speech projector: {speech_projector_type}')
        
        target_device = self.get_language_input_device()
        speech_features = [encoder_outs[i, :speech_lengths[i]].to(target_device) for i in range(len(encoder_outs))]
        
        return speech_features

    def encode_images(self, pixel_values, image_grid_thw):
        vision_encoder = self.get_vision_encoder()
        if vision_encoder is None:
            raise ValueError("Vision encoder is not initialized but image inputs were provided.")
        image_features = vision_encoder(pixel_values, image_grid_thw)
        target_device = self.get_language_input_device()
        return [feature.to(target_device) for feature in image_features]


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        speech, speech_lengths, pixel_values=None, image_grid_thw=None, aux_labels=None,
        speech_chunk_counts=None,
    ):
        speech_encoder = self.get_speech_encoder()
        vision_encoder = self.get_vision_encoder()
        has_speech = speech_encoder is not None and speech is not None
        has_image = vision_encoder is not None and pixel_values is not None and image_grid_thw is not None
        if (not has_speech and not has_image) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, aux_labels
        
        speech_features = self.encode_speech(speech, speech_lengths) if has_speech else []
        image_features = self.encode_images(pixel_values, image_grid_thw) if has_image else []

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _aux_labels = aux_labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        if aux_labels is None:
            aux_labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        aux_labels = [cur_aux_labels[cur_attention_mask] for cur_aux_labels, cur_attention_mask in zip(aux_labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_aux_labels = []
        cur_speech_idx = 0
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = int((cur_input_ids == SPEECH_TOKEN_INDEX).sum().item())
            num_image = int((cur_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            sample_speech_chunks = None
            if speech_chunk_counts is not None:
                sample_speech_chunks = int(speech_chunk_counts[batch_idx].item())

            def consume_silent_speech_chunks(count):
                nonlocal cur_speech_idx
                silent_features = []
                for _ in range(max(0, count)):
                    if cur_speech_idx >= len(speech_features):
                        break
                    silent_features.append(speech_features[cur_speech_idx][0:0])
                    cur_speech_idx += 1
                return silent_features

            silent_speech_features = []
            if sample_speech_chunks is None:
                # Use one placeholder chunk when the sample has no <speech> token.
                if num_speech == 0 and has_speech and cur_speech_idx < len(speech_features):
                    silent_speech_features = consume_silent_speech_chunks(1)
            elif num_speech == 0:
                silent_speech_features = consume_silent_speech_chunks(sample_speech_chunks)

            if num_speech == 0 and num_image == 0:
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds_1
                if silent_speech_features:
                    target_device = self.get_language_input_device()
                    silent_speech_features = [
                        feature.to(device=target_device, dtype=cur_input_embeds.dtype)
                        for feature in silent_speech_features
                    ]
                    cur_input_embeds = torch.cat([cur_input_embeds] + silent_speech_features, dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_aux_labels.append(aux_labels[batch_idx])
                continue
            
            modal_token_indices = []
            modal_token_indices += [(idx, "speech") for idx in torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist()]
            modal_token_indices += [(idx, "image") for idx in torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()]
            modal_token_indices = sorted(modal_token_indices, key=lambda item: item[0])
            modal_indices = [-1] + [idx for idx, _ in modal_token_indices] + [cur_input_ids.shape[0]]
            cur_input_ids_nospe = []
            cur_labels = labels[batch_idx]
            cur_aux_labels = aux_labels[batch_idx]
            cur_labels_nospe = []
            cur_aux_labels_nospe = []
            for i in range(len(modal_indices) - 1):
                cur_input_ids_nospe.append(cur_input_ids[modal_indices[i]+1:modal_indices[i+1]])
                cur_labels_nospe.append(cur_labels[modal_indices[i]+1:modal_indices[i+1]])
                cur_aux_labels_nospe.append(cur_aux_labels[modal_indices[i]+1:modal_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospe]

            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospe))
            cur_input_embeds_no_spe = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_aux_labels = []

            for i in range(len(modal_token_indices) + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_spe[i])
                cur_new_labels.append(cur_labels_nospe[i])
                cur_new_aux_labels.append(cur_aux_labels_nospe[i])
                if i < len(modal_token_indices):
                    _, modal_type = modal_token_indices[i]
                    if modal_type == "speech":
                        cur_modal_features = speech_features[cur_speech_idx]
                        cur_speech_idx += 1
                    elif modal_type == "image":
                        cur_modal_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                    else:
                        raise ValueError(f"Unknown multimodal token type: {modal_type}")
                    cur_new_input_embeds.append(cur_modal_features)
                    cur_new_labels.append(torch.full((cur_modal_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)) 
                    cur_new_aux_labels.append(torch.full((cur_modal_features.shape[0],), IGNORE_INDEX, device=cur_aux_labels.device, dtype=cur_aux_labels.dtype))
            
            if sample_speech_chunks is not None:
                silent_speech_features.extend(consume_silent_speech_chunks(sample_speech_chunks - num_speech))
            for cur_silent_features in silent_speech_features:
                cur_new_input_embeds.append(cur_silent_features)
                cur_new_labels.append(torch.full((cur_silent_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_new_aux_labels.append(torch.full((cur_silent_features.shape[0],), IGNORE_INDEX, device=cur_aux_labels.device, dtype=cur_aux_labels.dtype))

            target_device = self.get_language_input_device()
            cur_new_input_embeds = [x.to(target_device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_aux_labels = torch.cat(cur_new_aux_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_aux_labels.append(cur_new_aux_labels)

        # 防御性校验：整个 batch 遍历完成后，指针必须精确消费完 collator 侧
        # flatten 出来的 speech/image 列表，一个不多一个不少。一旦未来新增
        # task/分支破坏了"占位 chunk 也要前进指针"的约定，这里会立刻抛出清晰的
        # 错误，而不是让错位/越界发生在别处、难以定位。
        if has_speech and cur_speech_idx != len(speech_features):
            raise RuntimeError(
                f"speech_features consumption mismatch: consumed {cur_speech_idx}, "
                f"but batch contributed {len(speech_features)}. This usually means some "
                f"sample's placeholder/actual speech chunk was not accounted for."
            )
        if has_image and cur_image_idx != len(image_features):
            raise RuntimeError(
                f"image_features consumption mismatch: consumed {cur_image_idx}, "
                f"but batch contributed {len(image_features)}. This usually means some "
                f"sample's placeholder/actual image chunk was not accounted for."
            )

        def truncate_keep_supervision(embeds, labels, aux_labels, max_length):
            if embeds.shape[0] <= max_length:
                return embeds, labels, aux_labels

            supervised = labels.ne(IGNORE_INDEX)
            if aux_labels is not None:
                supervised = supervised | aux_labels.ne(IGNORE_INDEX)
            supervised_positions = torch.where(supervised)[0]
            if supervised_positions.numel() > 0:
                end = min(int(supervised_positions[-1].item()) + 1, embeds.shape[0])
                start = max(0, end - max_length)
            else:
                start = 0
                end = max_length
            return embeds[start:end], labels[start:end], aux_labels[start:end]

        # Truncate sequences to max length as speech embeddings can make the sequence longer.
        # Keep supervised assistant/response tokens when possible; right truncation can
        # drop the entire response after long speech/image features and make CE loss NaN.
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            truncated = [
                truncate_keep_supervision(embed, label, aux_label, tokenizer_model_max_length)
                for embed, label, aux_label in zip(new_input_embeds, new_labels, new_aux_labels)
            ]
            new_input_embeds = [x[0] for x in truncated]
            new_labels = [x[1] for x in truncated]
            new_aux_labels = [x[2] for x in truncated]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_aux_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_aux_labels[0].dtype, device=new_aux_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_aux_labels) in enumerate(zip(new_input_embeds, new_labels, new_aux_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_aux_labels_padded[i, -cur_len:] = cur_new_aux_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_aux_labels_padded[i, :cur_len] = cur_new_aux_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _aux_labels is None:
            new_aux_labels = None
        else:
            new_aux_labels = new_aux_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_aux_labels
