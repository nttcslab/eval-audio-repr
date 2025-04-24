"""Wrapper code for:

Scaling up masked audio encoder learning for general audio classification

## Reference
- [1] https://www.isca-archive.org/interspeech_2024/dinkel24b_interspeech.html
- [2] https://huggingface.co/mispeech/dasheng-base
"""

from evar.ar_base import BaseAudioRepr
import torch
import logging
try:
    from dasheng_model.feature_extraction_dasheng import DashengFeatureExtractor
    from dasheng_model.modeling_dasheng import DashengModel
except:
    logging.error('Install as follows.\n>>> pip install git+https://github.com/jimbozhang/hf_transformers_custom_model_dasheng.git')


class AR_Dasheng(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.preprocessor = DashengFeatureExtractor.from_pretrained(cfg.model_name)
        self.backbone = DashengModel.from_pretrained(cfg.model_name, outputdim=None)

    def encode_frames(self, batch_audio):
        preprocessed = self.preprocessor(audio.cpu(), sampling_rate=16000, return_tensors="pt")
        preprocessed = preprocessed.to(batch_audio.device)
        hidden_states = self.backbone(**preprocessed).hidden_states  # [B, T, D]
        return hidden_states.transpose(1, 2)  # [B, D, T]

    def forward(self, batch_audio):
        preprocessed = self.preprocessor(batch_audio.cpu(), sampling_rate=16000, return_tensors="pt")
        preprocessed = preprocessed.to(batch_audio.device)
        return self.backbone(**preprocessed).logits
