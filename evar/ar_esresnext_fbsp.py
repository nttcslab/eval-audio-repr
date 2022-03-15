"""Wrapper code for:

ESResNe(X)t-fbsp: Learning Robust Time-Frequency Transformation of Audio

## Reference
- [1] https://arxiv.org/abs/2104.11587
- [2] https://github.com/AndreyGuzhov/ESResNeXt-fbsp
"""

from evar.ar_base import BaseAudioRepr
import torch
import librosa
import numpy as np
import logging
from evar.model_utils import ensure_weights
try:
    from external.esresnext.model.esresnet_fbsp import ESResNeXtFBSP
except:
    logging.info('Make your copy of ESResNeXt-fbsp under external folder. Check Preparing-models.md for the details.')
    class ESResNeXtFBSP:
        pass


class ESResNeXtFBSP_(ESResNeXtFBSP):

    def forward_reduced_featues(self, x, tfm=None):
        x = self._forward_pre_processing(x)
        if tfm is not None:
            x = tfm(x)
        x = self._forward_features(x)
        x = self._forward_reduction(x)
        return x


class AR_ESResNeXtFBSP(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.backbone = ESResNeXtFBSP_(
            **{"n_fft": 2048,
                "hop_length": 561,
                "win_length": 1654,
                "window": "blackmanharris",
                "normalized": True,
                "onesided": True,
                "spec_height": -1,
                "spec_width": -1,
                "num_classes": 527,
                "apply_attention": True,
            }
        )
        ensure_weights('external/ESResNeXtFBSP_AudioSet.pt',
            'https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt')
        self.backbone.load_state_dict(torch.load('external/ESResNeXtFBSP_AudioSet.pt'))

    def encode_frames(self, batch_audio):
        X = self.forward(batch_audio)
        X = X.unsqueeze(1) # Already have temporally pooled, just adding extra frame dimension [B, 2048] -> [B, 1, 2048]
        return X

    def forward(self, batch_audio):
        return self.backbone.forward_reduced_featues(batch_audio * 32767) # [B, 2048]
