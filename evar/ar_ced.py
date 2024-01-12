"""Wrapper code for:

CED: Consistent ensemble distillation for audio tagging

## Reference
- [1] https://arxiv.org/abs/2308.11957
- [2] https://github.com/RicherMans/ced
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import sys
import logging
import torch
try:
    sys.path.append('external/hf_transformers_custom_model_ced')
    from ced_model.feature_extraction_ced import CedFeatureExtractor
    from ced_model.modeling_ced import CedForAudioClassification
    from transformers.modeling_outputs import SequenceClassifierOutput
except:
    pass  # please install CED


class AR_CED(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        model_path = cfg.weight_file
        self.feature_extractor = CedFeatureExtractor.from_pretrained(model_path)
        self.backbone = CedForAudioClassification.from_pretrained(model_path)

        logging.info(f' Using weight from Hugging Face: {cfg.weight_file}')

    def encode_frames(self, batch_audio):
        inputs = self.feature_extractor(batch_audio.to('cpu'), sampling_rate=16000, return_tensors="pt")
        inputs['input_values'] = inputs['input_values'].to('cuda')
        features = self.backbone(**inputs).hidden_states
        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        features = self.encode_frames(batch_audio)
        return features.mean(-1)

