"""Wrapper code for:

Self-supervised Audio Teacher-Student Transformer for Both Clip-level and Frame-level Tasks

## Reference
- [1] https://arxiv.org/abs/2306.04186
- [2] https://github.com/Audio-WestlakeU/audiossl/blob/main/audiossl/methods/atstframe
"""

from evar.ar_base import BaseAudioRepr
import logging
import sys
from einops import rearrange
try:
    sys.path.append('external/audiossl')
    from audiossl.methods.atstframe.embedding import load_model, get_scene_embedding, get_timestamp_embedding
except Exception as e:
    pass  # Please clone audiossl


class AR_ATST_Frame(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.backbone = load_model(cfg.weight_file)
        logging.info(f' Using weight file: {cfg.weight_file}')

    def encode_frames(self, batch_audio):
        batch_audio = batch_audio.unsqueeze(1)  # [B, L] -> [B, 1, L] as described in the README
        x, _ = get_timestamp_embedding(batch_audio, self.backbone)  # -> [B,T,N_BLOCKS*emb_size]
        # no need x = rearrange(x, 'B 1 T N D -> B (N * D) T')
        return x

    def forward(self, batch_audio):
        #import pdb; pdb.set_trace()
        batch_audio = batch_audio.unsqueeze(1)  # [B, L] -> [B, 1, L] as described in the README
        x = get_scene_embedding(batch_audio, self.backbone)  # [B,N_BLOCKS*emb_size]
        return x
