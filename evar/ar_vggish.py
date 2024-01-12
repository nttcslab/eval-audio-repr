"""Wrapper code for:

CNN Architectures for Large-Scale Audio Classification

## References
- [1] https://research.google/pubs/pub45611/
- [2] VGGish: https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/blob/master/vggish.py
- [3] VGG: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

from evar.ar_base import (BaseAudioRepr, temporal_pooling)
import torch
import numpy as np
import logging
try:
    from external.tcvrick_vggish import vggish
    from external.tcvrick_vggish.audioset import vggish_input
except:
    pass  # logging.error('Make your copy of VGGish under external folder. Check Preparing-models.md for the details.')


class AR_VGGish(BaseAudioRepr):
    def __init__(self, cfg, vggish_class=None):
        super().__init__(cfg=cfg)

        self.vggish = vggish.VGGish() if vggish_class is None else vggish_class()
        weight_file = 'external/pytorch_vggish.pth'
        logging.info(f' using pretrained weight: {weight_file}')
        self.vggish.load_state_dict(torch.load(weight_file))

    def to_audio_features(self, batch_audio):
        # raw audio -> spectrogram
        device = batch_audio.device
        X = [vggish_input.waveform_to_examples(x.cpu().numpy(), self.cfg.sample_rate) for x in batch_audio]
        X = torch.tensor(np.array(X)).float().to(device) # ex.) [256, 7, 96, 64] if fsd50k. [B,Frame,T,F]
        return X

    def encode_frames(self, batch_audio):
        X = self.to_audio_features(batch_audio)
        Xs = [self.vggish(X[:, i:i+1]) for i in range(X.shape[1])]
        X = torch.stack(Xs, dim=2) # [B, D] x Frame -> [B, D, Frame]
        return X

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio)) # [B, D]


class AR_VGGish_4K(AR_VGGish):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # Remove all the layers after the first FC layer.
        self.vggish.fc = torch.nn.Sequential(*list(self.vggish.fc.children())[:-4])
