"""Masked Modeling Duo (M2D) Wrapper for EVAR.

Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input
https://ieeexplore.ieee.org/document/10097236/

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079
"""

from evar.ar_base import BaseAudioRepr, BaseCLAP, calculate_norm_stats, normalize_spectrogram
import torch
import logging

try:
    import sys
    sys.path.append('external/m2d')
    sys.path.append('..')
    from m2d.runtime_audio import RuntimeM2D
except Exception as e:
    pass  # print(f'(For M2D users) Build your EVAR in your M2D folder.')


class AR_M2D(BaseAudioRepr):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg)

        if make_runtime:
            self.runtime = RuntimeM2D(cfg=cfg, weight_file=cfg.weight_file)
            self.runtime.eval()
            self.cfg = self.runtime.cfg

    def precompute(self, device, data_loader):
        if not self.cfg.mean or not self.cfg.std:
            self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)
        else:
            self.norm_stats = [self.cfg.mean, self.cfg.std]
            logging.info(f' using spectrogram norimalization stats: {self.norm_stats}')

    def precompute_lms(self, device, data_loader):
        self.precompute(device, data_loader)
        self.lms_mode = True

    def using_non_last_layer_output(self):
        if self.cfg.output_layers is None: return True
        if len(self.cfg.output_layers) > 1: return True
        return self.cfg.output_layers[0] != -1

    def encode_frames(self, batch_audio):
        x = self.runtime.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)
        x = self.augment_if_training(x)
        features = self.runtime.encode_lms(x, return_layers=self.using_non_last_layer_output())
        # stack layer outputs
        if self.using_non_last_layer_output():
            states_to_stack = [features[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in features]
            features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        if hasattr(self, 'lms_mode'):
            x = self.encode_frames_lms(batch_audio)
        else:
            x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]

    def encode_frames_lms(self, batch_lms):
        x = normalize_spectrogram(self.norm_stats, batch_lms)
        x = self.augment_if_training(x)
        hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]


class AR_M2D_CLAP(AR_M2D, BaseCLAP):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg, make_runtime=make_runtime)

    def encode_audio(self, batch_audio):
        return self.runtime.encode_clap_audio(batch_audio)

    def encode_text(self, batch_text):
        return self.runtime.encode_clap_text(batch_text)

