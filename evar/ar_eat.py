"""Wrapper code for:

EAT: Self-Supervised Pre-Training with Efficient Audio Transformer

## Reference
- [1] https://www.ijcai.org/proceedings/2024/421
- [2] https://huggingface.co/worstchan/EAT-base_epoch30_pretrain
"""

from evar.ar_base import BaseAudioRepr
import torch
import torchaudio
import logging
try:
    from transformers import AutoModel
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_EAT(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = AutoModel.from_pretrained(cfg.weight_file, trust_remote_code=True)

        self.target_length = 1024
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def encode_frames(self, batch_audio):
        def get_one(waveform):
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True,
                sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128,
                dither=0.0, frame_shift=10)
            return fbank
        device = batch_audio.device
        if len(batch_audio.shape) == 1:  # [L] -> [1, L]
            batch_audio = batch_audio.unsqueeze(0)
        mel = torch.stack([get_one(w.unsqueeze(0)) for w in batch_audio]).to(device)
 
        # Pad or truncate
        n_frames = mel.shape[1]
        if n_frames < self.target_length:
            mel = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - n_frames))(mel)
        else:
            mel = mel[:, :self.target_length, :]

        # Normalize
        x = (mel - self.norm_mean) / (self.norm_std * 2)

        # x = self.augment_if_training(x)  # TODO for future complete implementation -- OK for linear evaluation

        features = []
        while x.shape[1] > 0:
            chunk = x[:, :self.target_length, :]
            if chunk.shape[1] < self.target_length:
                chunk = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - chunk.shape[1]))(chunk)
            chunk = chunk.unsqueeze(1)  # (B, 1, T, F)
            feat = self.model.extract_features(chunk)
            features.append(feat)            
            x = x[:, self.target_length:, :]
        features = torch.cat(features, dim=1)

        return features.transpose(1, 2) # [B, T', D] -> [B, D, T']

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]

    def normalize_spectrogram(self, spectrograms):
        mu, sigma = self.norm_mean, self.norm_std
        spectrograms = (spectrograms - mu) / (sigma * 2) # follows the original AudiosetDataset
        return spectrograms
