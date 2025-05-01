"""Wrapper code for:

Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking

## Reference
- [1] https://arxiv.org/abs/2406.16148
- [2] https://github.com/evelyn0414/OPERA
"""

from evar.ar_base import BaseAudioRepr, np
import torch
import librosa
import logging

try:
    import sys
    sys.path.append('../../external/OPERA')
    import os
    evar_home = os.getenv('EVAR', '')
    sys.path.append(os.path.join(evar_home, 'external/OPERA'))
    from src.model.models_cola import Cola
    from src.util import _equally_slice_pad_sample, _duplicate_padding
except Exception as e:
    pass  # print(f'(For M2D users) Build your EVAR in your M2D folder.')


def split_pad_sample(sample, desired_length, sample_rate, types='repeat'):
    # Quoted from https://github.com/evelyn0414/OPERA/blob/main/src/util.py
    """
    if the audio sample length > desired_length, then split and pad samples
    else simply pad samples according to pad_types
    * types 'zero'   : simply pad by zeros (zero-padding)
    * types 'repeat' : pad with duplicate on both sides (half-n-half)
    * types 'aug'    : pad with augmented sample on both sides (half-n-half)	
    """
    if types == 'zero':
        return _equally_slice_pad_sample(sample, desired_length, sample_rate)

    output_length = int(desired_length * sample_rate)
    soundclip = sample[0].copy()
    n_samples = len(soundclip)

    output = []
    if n_samples > output_length:
        """
        if sample length > desired_length, slice samples with desired_length then just use them,
        and the last sample is padded according to the padding types
        """
        # frames[j] = x[j * hop_length : j * hop_length + frame_length]
        frames = librosa.util.frame(
            soundclip, frame_length=output_length, hop_length=output_length//2, axis=0)
        for i in range(frames.shape[0]):
            output.append((frames[i], sample[1], sample[2]))

        # get the last sample
        last_id = frames.shape[0] * (output_length//2)
        last_sample = soundclip[last_id:]

        padded = _duplicate_padding(
            soundclip, last_sample, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))
    else:  # only pad
        padded = _duplicate_padding(
            soundclip, soundclip, output_length, sample_rate, types)
        output.append((padded, sample[1], sample[2]))

    return output


def pre_process_audio_mel_t(audio, sample_rate=16000, n_mels=64, f_min=50, f_max=8000, nfft=1024, hop=512):
    # Quoted from https://github.com/evelyn0414/OPERA/blob/main/src/util.py
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    # convert scale to dB from magnitude
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() != S.min():
        mel_db = (S - S.min()) / (S.max() - S.min())
    else:
        mel_db = S
        print("warning in producing spectrogram!")

    return mel_db


def get_entire_signal_librosa(data, input_sec=8, sample_rate=16000, butterworth_filter=None, pad=False, from_cycle=False, yt=None, types='repeat'):
    device = data.device
    # Cut from https://github.com/evelyn0414/OPERA/blob/main/src/util.py
    # Trim leading and trailing silence from an audio signal.
    FRAME_LEN = int(sample_rate / 10)  # 
    HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
    yt, index = librosa.effects.trim(data.cpu().numpy(), frame_length=FRAME_LEN, hop_length=HOP)

    # check audio not too short    
    duration = librosa.get_duration(y=yt, sr=sample_rate)
    if duration < input_sec:
        yt = split_pad_sample([yt, 0,0], input_sec, sample_rate, types)[0][0]
    
    # # visualization for testing the spectrogram parameters
    # plot_melspectrogram(yt.squeeze(), title=filename.replace("/", "-"))
    return torch.tensor(pre_process_audio_mel_t(yt.squeeze(), f_max=8000)).to(device)


class AR_OPERA_CT(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        if 'icbhi_sprs_mode' not in cfg:
            logging.error('\n\n *** The model supports app/ICBHI_SPRT only. Exiting... ***\n')
            exit(-1)
        self.backbone = Cola(encoder="htsat")
        ckpt = torch.load(cfg.weight_file)
        self.backbone.load_state_dict(ckpt["state_dict"], strict=False)

    def encode_frames(self, batch_audio):
        x = get_entire_signal_librosa(batch_audio, input_sec=8) #, input_sec=self.cfg.unit_samples / self.cfg.sample_rate)
        x = self.augment_if_training(x)
        x = x.transpose(-2, -1)  # B,D,T -> B,T,D
        features = self.backbone.extract_feature(x, self.cfg.feature_d)
        return features.unsqueeze(-1) # [B, D] -> [B, D, 1]

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]

