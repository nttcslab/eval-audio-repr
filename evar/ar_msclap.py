"""Wrapper code for:

Natural Language Supervision for General-Purpose Audio Representations

## Reference
- [1] https://arxiv.org/abs/2309.05767
- [2] https://github.com/microsoft/CLAP
"""

from evar.ar_base import BaseCLAP
try:
    from msclap import CLAP
except:
    pass  # please install: pip install msclap


class AR_MSCLAP(BaseCLAP):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        # MS CLAP accepts file name as audio input.
        self.filename_mode = True

        self.backbone = CLAP(version=str(cfg.weight_file), use_cuda=False)

    def encode_frames(self, batch_audio):
        assert False, 'encode_frames for MS CLAP is not supported for now'

    def forward(self, batch_audio):
        audio_embeddings = self.backbone.get_audio_embeddings(batch_audio)
        return audio_embeddings

    def encode_audio(self, batch_audio):
        audio_embeddings = self.forward(batch_audio)
        return audio_embeddings

    def encode_text(self, batch_text):
        text_embeddings = self.backbone.get_text_embeddings(batch_text)
        return text_embeddings

    def compute_similarity(self, text_embs, audio_embs):
        similarity = self.backbone.compute_similarity(audio_embs, text_embs)
        return similarity.T
