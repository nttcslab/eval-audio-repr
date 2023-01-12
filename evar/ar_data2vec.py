"""Wrapper code for:

Data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language

## Reference
- [1] https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/
- [2] https://huggingface.co/facebook/data2vec-audio-base-960h
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import logging
try:
    from transformers import Data2VecAudioModel
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_Data2Vec(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.backbone = Data2VecAudioModel.from_pretrained(cfg.data2vec_model)

    def encode_frames(self, batch_audio):
        # logits = self.backbone(batch_audio, output_hidden_states=True)['hidden_states'] # [B, T, D]
        logits = self.backbone(batch_audio)['last_hidden_state'] # [B, T, D]
        return logits.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))
