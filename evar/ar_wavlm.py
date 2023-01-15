"""Wrapper code for:

WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing

## Reference
- [1] https://arxiv.org/abs/2110.13900
- [2] https://huggingface.co/microsoft/wavlm-large
- [3] https://github.com/microsoft/unilm/tree/master/wavlm
- [4] https://github.com/huggingface/transformers/blob/main/src/transformers/models/wavlm/modeling_wavlm.py
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import logging
try:
    from transformers import WavLMModel, Wav2Vec2Processor
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_WavLM(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h') # instead of cfg.pretrained_model because non-ft models fail. wav2vec2-base-960h should be fine for preprocessing.
        self.backbone = WavLMModel.from_pretrained(cfg.pretrained_model)

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        preprocessed = self.processor(batch_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        features = self.backbone(preprocessed)['last_hidden_state'] # [B, T, D]
        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))
