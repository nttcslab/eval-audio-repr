"""Wrapper code for:

Towards Learning a Universal Non-Semantic Representation of Speech

## Reference
- [1] http://arxiv.org/abs/2002.12764
- [2] https://aihub.cloud.google.com/u/0/p/products%2F41239b97-c960-479a-be50-ae7a23ae1561
"""

from evar.ar_base import (BaseAudioRepr, temporal_pooling)
import torch
import logging
try:
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf.executing_eagerly()
    import tensorflow_hub as hub
except:
    pass  # logging.error('Install tensorflow and tensorflow_hub.\n>>> pip install tensorflow tensorflow_hub')


class AR_TRILL(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.model = hub.load(cfg.trill_url)
        self.emb_type = cfg.trill_emb_type

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        x = self.model(samples=tf.convert_to_tensor(batch_audio.cpu().numpy()), sample_rate=16000)[self.emb_type].numpy()
        x = torch.tensor(x.transpose(0, 2, 1)).float().to(device) # transpose: [B,T,D] -> [B,D,T]
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x
