"""Wrapper code for:

Look, Listen and Learn More: Design Choices for Deep Audio Embeddings

## Reference
- [1] https://arxiv.org/abs/2104.11587
- [2] https://github.com/marl/openl3
- [3] https://github.com/torchopenl3/torchopenl3
"""

from evar.ar_base import (BaseAudioRepr, temporal_pooling)
import torch
import logging
try:
    import torchopenl3
    from torchopenl3.utils import preprocess_audio_batch
except:
    pass  # logging.error('Install toprchopenl3.\n>>> pip install torchopenl3')


class AR_OpenL3(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.openl3_model = torchopenl3.models.load_audio_embedding_model(
            cfg.openl3_input_repr, cfg.openl3_content_type, cfg.feature_d)

    def encode_frames(self, batch_audio):
        frame_embeddings, ts_list = torchopenl3.get_audio_embedding(batch_audio,
            self.cfg.sample_rate, model=self.openl3_model) # -> [B, T, D]
        return frame_embeddings.transpose(1, 2) # -> [B, D, T]

    def forward(self, batch_audio):
        frame_embeddings = self.encode_frames(batch_audio)
        return temporal_pooling(self, frame_embeddings)
