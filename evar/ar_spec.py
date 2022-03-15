"""Wrapper code for:

Mel-spectrogram and linear spectrogram.
"""

from evar.ar_base import (BaseAudioRepr, ToLogMelSpec,
    calculate_norm_stats, normalize_spectrogram, temporal_pooling)
import nnAudio.Spectrogram


class AR_MelSpec(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        return normalize_spectrogram(self.norm_stats, x)

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return temporal_pooling(self, x)


class ToLogLinSpec(ToLogMelSpec):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.to_spec = nnAudio.Spectrogram.STFT(n_fft=cfg.n_fft, win_length=cfg.window_size,
            freq_bins=None, hop_length=cfg.hop_size, 
            center=True, sr=cfg.sample_rate,
            output_format="Magnitude",
            verbose=False,
        )


class AR_LinSpec(AR_MelSpec):
    def __init__(self, cfg):
        cfg.n_mels = 64 # dummy for making reuse of AR_MelSpec easy
        super().__init__(cfg=cfg)
        self.to_feature = ToLogLinSpec(cfg)
