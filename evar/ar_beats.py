"""Wrapper code for:

BEATs: Audio Pre-Training with Acoustic Tokenizers

## Reference
- [1] https://arxiv.org/abs/2212.09058
- [2] https://github.com/microsoft/unilm/blob/master/beats/README.md
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import sys
import logging
import torch
try:
    sys.path.append('external/unilm/beats')
    from Tokenizers import TokenizersConfig, Tokenizers
    from BEATs import BEATs, BEATsConfig
except:
    pass


class AR_BEATs(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # load the pre-trained checkpoints
        checkpoint = torch.load(cfg.weight_file)
        logging.info(f' Using weight_file: {cfg.weight_file}')

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        self.backbone = BEATs_model.eval()

    def encode_frames(self, batch_audio):
        padding_mask = torch.zeros_like(batch_audio).bool()
        features = self.backbone.extract_features(batch_audio, padding_mask=padding_mask)[0]
        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]


class AR_BEATsTokenizer(BaseAudioRepr):
    """EXPERIMENTAL"""

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # load the pre-trained checkpoints
        checkpoint = torch.load(cfg.weight_file)
        logging.info(f' Using weight_file: {cfg.weight_file}')

        cfg = TokenizersConfig(checkpoint['cfg'])
        BEATs_tokenizer = Tokenizers(cfg)
        BEATs_tokenizer.load_state_dict(checkpoint['model'])
        self.backbone = BEATs_tokenizer.eval()

    def encode_frames(self, batch_audio):
        padding_mask = torch.zeros_like(batch_audio).bool()
        features = self.backbone.extract_labels(batch_audio, padding_mask=padding_mask)
        features = features.reshape(batch_audio.shape[0], -1).unsqueeze(-1)
        return features.to(float) # [B, D, T]

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]
