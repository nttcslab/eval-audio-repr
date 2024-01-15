"""Wrapper code for:

PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition

## Reference
- [1] https://arxiv.org/abs/1912.10211
- [2] https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

from evar.ar_base import BaseAudioRepr
from evar.model_utils import ensure_weights, load_pretrained_weights
import logging
try:
    from evar.cnn14_decoupled import AudioFeatureExtractor, Cnn14_Decoupled
except:
    logging.info('** Install torchlibrosa if you use Cnn14 **')


class AR_Cnn14(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.feature_extractor = AudioFeatureExtractor(n_fft=cfg.n_fft, hop_length=cfg.hop_size, win_length=cfg.window_size,
            sample_rate=cfg.sample_rate, n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max)
        self.body = Cnn14_Decoupled()
        weight_file = 'external/Cnn14_16k_mAP=0.438.pth'
        ensure_weights(weight_file, 'https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth')
        load_pretrained_weights(self.body, weight_file)

    def encode_frames(self, batch_audio):
        x = self.feature_extractor(batch_audio)  # (B, 1, T, F(mel_bins))
        x = self.augment_if_training(x.transpose(-2, -1)).transpose(-2, -1)  # (..., T, F) -> (..., F, T) -augment-> (..., T, F)
        return self.body.encode(x)               # (B, D, T)

    def forward(self, batch_audio):
        frame_embeddings = self.encode_frames(batch_audio)  # (B, D, T)
        return self.body.temporal_pooling(frame_embeddings) # (B, D)
