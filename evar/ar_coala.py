"""Wrapper code for:

COALA: Co-Aligned Autoencoders for Learning Semantically Enriched Audio Representations

## Note

- FS is 22,000: https://github.com/xavierfav/coala/blob/master/utils.py#L66
- Fixed the original scaler_top_1000.pkl: https://github.com/xavierfav/coala/issues/3

## Reference
- [1] https://arxiv.org/abs/2006.08386
- [2] https://github.com/xavierfav/coala
"""

from evar.ar_base import (BaseAudioRepr, temporal_pooling)
import torch
import librosa
import numpy as np
import logging
try:
    from external.coala.encode import return_loaded_model, scaler
    from external.coala.models_t1000 import AudioEncoder
    from external.coala.utils import pad
except:
    pass  # logging.error('Make your copy of COALA under external folder. Check Preparing-models.md for the details.')


def _compute_spectrogram(audio, sr=22000, n_mels=96):
    """Borrowed from coala/utils.py, removed wav loading to accept raw audio input."""
    # zero pad and compute log mel spec
    try:
        x = pad(audio, sr)
    except ValueError:
        x = audio
    audio_rep = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=512, n_fft=1024, n_mels=n_mels, power=1.)
    audio_rep = np.log(audio_rep + np.finfo(np.float32).eps)
    return audio_rep


def _extract_audio_embedding_chunks(model, audio):
    """Borrowed from coala/encode.py, modified to accept torch tensor raw audio input."""
    with torch.no_grad():
        device = audio.device
        x = _compute_spectrogram(audio.cpu().numpy())
        x_chunks = np.array([scaler.transform(chunk.T) for chunk in 
                librosa.util.frame(np.asfortranarray(x), frame_length=96, hop_length=96, axis=-1).T])
        x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1).to(device)
        embedding_chunks, embedding_d_chunks = model(x_chunks)
        return embedding_chunks, embedding_d_chunks


class AR_COALA(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.model = return_loaded_model(AudioEncoder, 'external/coala/saved_models/dual_ae_c/audio_encoder_epoch_200.pt')

    def encode_frames(self, batch_audio):
        xs = [_extract_audio_embedding_chunks(self.model, x)[0] for x in batch_audio]
        x = torch.stack(xs).transpose(1, 2) # [Frame, D] x B -> [B, Frame, D] -> [B, D, Frame (T)]
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x
