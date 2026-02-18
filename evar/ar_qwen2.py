"""Wrapper code for:

Qwen2.5-Omni Technical Report

## Reference
- [1] https://arxiv.org/abs/2503.20215
- [2] https://huggingface.co/docs/transformers/model_doc/qwen2_5_omni
"""

from evar.ar_base import BaseAudioRepr
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
try:
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_Qwen2_5(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(cfg.weight_file, trust_remote_code=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            cfg.weight_file, torch_dtype="auto", device_map="cuda")

    def get_audio_features(self, waves):
        # waves: [B, L]
        model = self.model
        processor = self.processor

        inputs = processor(
            text="", 
            audio=[w.cpu().numpy() for w in waves],
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt"
        )
        input_features, feature_attention_mask = inputs["input_features"].to(model.device), inputs["feature_attention_mask"]

        # https://github.com/huggingface/transformers/blob/b2028e775a52bf57ac2b6bd71b49ce61fa3adde6/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1804C5-L1804C28
        # def get_audio_features
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        audio_feat_lengths, audio_output_lengths = model.thinker.audio_tower._get_feat_extract_output_lengths(
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        aftercnn_lens = audio_feat_lengths

        # https://github.com/huggingface/transformers/blob/b2028e775a52bf57ac2b6bd71b49ce61fa3adde6/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L792C1-L843C1
        # def forward(self, input_features, feature_lens=None, aftercnn_lens=None, **kwargs: Unpack[TransformersKwargs]):
        # feature_lens (`torch.LongTensor` of shape `(batch_size,)`):  mel length
        # aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`): mel length after cnn
        self = model.thinker.audio_tower
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.full((chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        return hidden_states_list

    def encode_frames(self, batch_audio):
        hidden_states_list = self.get_audio_features(batch_audio)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.model.thinker.audio_tower.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            #each_audio_states = self.model.thinker.audio_tower.ln_post(each_audio_states)
            #each_audio_states = self.model.thinker.audio_tower.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        features = torch.stack(token_audio_list).to(torch.float32)
        return features.transpose(1, 2) # [B, T', D] -> [B, D, T']

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]


class AR_Qwen2_5_Token(AR_Qwen2_5):
    def encode_frames(self, batch_audio):
        hidden_states_list = self.get_audio_features(batch_audio)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.model.thinker.audio_tower.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.model.thinker.audio_tower.ln_post(each_audio_states)
            each_audio_states = self.model.thinker.audio_tower.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        features = torch.stack(token_audio_list).to(torch.float32)
        return features.transpose(1, 2) # [B, T', D] -> [B, D, T']
