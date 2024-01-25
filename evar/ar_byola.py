"""Wrapper code for:

BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation

## Reference
- [1] https://arxiv.org/abs/2103.06695
- [2] https://github.com/nttcslab/byol-a
"""

from evar.ar_base import (BaseAudioRepr, ToLogMelSpec, calculate_norm_stats, normalize_spectrogram, temporal_pooling)
from evar.model_utils import load_pretrained_weights
import logging
try:
    from external.byol_a.byol_a.models import AudioNTT2020Task6, AudioNTT2020Task6X
except Exception as e:
    pass  # logging.info(f'Make your copy of BYOL-A under external folder. Check Preparing-models.md for the details.')


class AR_BYOLA(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)

        self.body = AudioNTT2020Task6(n_mels=cfg.n_mels, d=cfg.feature_d)
        if cfg.weight_file is not None:
            load_pretrained_weights(self.body, cfg.weight_file, model_key='body')

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x) # B,F,T
        x = self.augment_if_training(x)
        x = x.unsqueeze(1)    # -> B,1,F,T
        x = self.body(x)      # -> B,T,D=C*F
        x = x.transpose(1, 2) # -> B,D,T
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x


class AR_BYOLAX(BaseAudioRepr):
    """A BYOL-A variant extended to stack features from all the layers."""
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)

        self.body = AudioNTT2020Task6X(n_mels=cfg.n_mels, d=cfg.feature_d)
        if cfg.weight_file is not None:
            self.body.load_weight(cfg.weight_file, device='cpu')
        self.cfg.feature_d = self.cfg.feature_d * self.body.n_feature_layer

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x) # B,F,T
        x = self.augment_if_training(x)
        x = x.unsqueeze(1)    # -> B,1,F,T
        x = self.body(x, layered=True) # -> B,T,D=C*F*Layer
        x = x.transpose(1, 2) # -> B,D,T
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x

