"""Wrapper code for:

AST: Audio Spectrogram Transformer

## Reference
- [1] https://arxiv.org/abs/2104.01778 Y. Gong, Y.-A. Chung, and J. Glass, “Ast: Audio spectrogram transformer,” arXiv preprint arXiv:2104.01778, 2021.
- [2] https://github.com/YuanGongND/ast
"""

from evar.ar_base import (BaseAudioRepr, calculate_norm_stats)
import torch
import torchaudio

try:
    from external.ast.src.models import ASTModel
except Exception as e:
    pass  # print(f'(For AST users) Make your copy of AST under external folder. Check Preparing-models.md for the details.')


class AST_Feature(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, waveforms):
        def get_one(waveform):
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True,
                sample_frequency=self.cfg.sample_rate, use_energy=False,
                window_type=self.cfg.window, num_mel_bins=self.cfg.n_mels,
                dither=0.0, frame_shift=10)
            return fbank
        device = waveforms.device
        fbanks = torch.stack([get_one(w.unsqueeze(0)) for w in waveforms])
        return fbanks.to(device)


class AR_AST(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = AST_Feature(cfg)
        tdim = self.to_feature(torch.rand(1, cfg.unit_samples)).shape[1]
        self.backbone = ASTModel(label_dim=10, input_tdim=tdim, imagenet_pretrain=True,
                                 audioset_pretrain=True, pretrained_weight=cfg.weight_file)

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        # AST returns a single embeddings for one audio, then simply add time axis.
        return self.forward(batch_audio).unsqueeze(-1) # B,D -> B,D,1

    def forward(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = self.normalize_spectrogram(x)
        x = self.augment_if_training(x)
        x = self.backbone(x)
        return x

    def normalize_spectrogram(self, spectrograms):
        mu, sigma = self.norm_stats
        spectrograms = (spectrograms - mu) / (sigma * 2) # follows the original AudiosetDataset
        return spectrograms
