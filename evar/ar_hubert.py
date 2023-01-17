"""Wrapper code for:

HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

## Reference
- [1] https://ai.facebook.com/blog/hubert-self-supervised-representation-learning-for-speech-recognition-generation-and-compression/
- [2] https://huggingface.co/facebook/hubert-large-ls960-ft
- [3] https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modeling_hubert.py
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import logging
import torch
try:
    from transformers import HubertModel, Wav2Vec2Processor
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_Hubert(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h') # instead of cfg.pretrained_model because non-ft models fail. wav2vec2-base-960h should be fine for preprocessing.
        self.backbone = HubertModel.from_pretrained(cfg.pretrained_model)

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        preprocessed = self.processor(batch_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        hidden_states = self.backbone(preprocessed, output_hidden_states=True).hidden_states # [B, T, D]
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else hidden_states
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))
