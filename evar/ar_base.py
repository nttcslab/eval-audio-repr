"""Audio Representation Base class and the related utilities.

### How BaseAudioRepr works in the evaluation pipeline

The evaluation pipeline will call BaseAudioRepr functions as follows,
where `ar` is an instance of BaseAudioRepr class.

1. Call `ar.precompute(device, data_loader)` to let the `ar` compute data statistics for some models which need stats.
2. Call `ar.forward(batch_audio)` to let the `ar` convert a batch of audio samples into a batch of embedding vectors.

After converting all audio files to the embeddings, the pipeline will conduct a linear evaluation.

#### `ar.encode_frames(batch_audio)` is not used, what is it for?

This is a template for handling embedding vectors per time frame.

    # Convert batch audio into batch of embeddings per time frame.
    # [B,Samples] -> [B,D,T] where T is Time frames and D is embedding dimensions.
    embedding_frames = encode_frames(batch_audio)
    # Convert embeddings per time frames into a single embedding vector: [B,D,T] -> [B,D]
    embeddings = forward(embedding_frames)
"""

from evar.common import (logging, nn, torch, np, EasyDict)
from evar.utils.calculations import RunningStats
from evar.model_utils import (mean_max_pooling, mean_pooling, max_pooling,
    MLP, initialize_layers, set_layers_trainable, show_layers_trainable)
import nnAudio.Spectrogram


class BaseAudioRepr(nn.Module):
    def __init__(self, cfg, filename_mode=False):
        super().__init__()
        self.cfg = EasyDict(cfg.copy())
        self.aug_fn = None
        self.filename_mode = filename_mode  # True if the repr accepts file name as audio input.

    def precompute(self, device, data_loader):
        """Do precomputation using training data whatever needed,
        normalization statistics for example.
        """
        pass

    def encode_frames(self, batch_audio):
        raise NotImplementedError(f'implement encode_frames() in {self.__class__}')

    def forward(self, batch_audio):
        raise NotImplementedError(f'implement forward() in {self.__class__}')

    def set_augment_tf_feature_fn(self, aug_fn):
        self.aug_fn = aug_fn

    def augment_if_training(self, tf_feature):
        if self.aug_fn is None or not self.training:
            return tf_feature
        return self.aug_fn(tf_feature)


class BaseCLAP(BaseAudioRepr):
    def encode_audio(self, batch_audio):
        raise NotImplementedError(f'implement encode_audio() in {self.__class__}')

    def encode_text(self, batch_text):
        raise NotImplementedError(f'implement encode_audio() in {self.__class__}')

    def compute_similarity(self, text_embs, audio_embs):
        # This implementation follows: https://github.com/microsoft/CLAP/blob/main/msclap/CLAPWrapper.py#L329
        audio_embs = audio_embs / torch.norm(audio_embs, dim=-1, keepdim=True)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        similarity = text_embs @ audio_embs.T
        return similarity


class ToLogMelSpec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.to_spec = nnAudio.Spectrogram.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.window_size,
            hop_length=cfg.hop_size,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def forward(self, batch_audio):
        x = self.to_spec(batch_audio)
        return (x + torch.finfo().eps).log()


def _calculate_stats(device, data_loader, converter, max_samples):
    running_stats = RunningStats()
    sample_count = 0
    for batch_audio, _ in data_loader:
        with torch.no_grad():
            converteds = converter(batch_audio.to(device)).detach().cpu()
        running_stats.put(converteds)
        sample_count += len(batch_audio)
        if sample_count >= max_samples:
            break
    return torch.tensor(running_stats())


def calculate_norm_stats(device, data_loader, converter, max_samples=5000):
    norm_stats = _calculate_stats(device, data_loader, converter, max_samples)
    logging.info(f' using spectrogram norimalization stats: {norm_stats.numpy()}')
    return norm_stats


def calculate_scaling_stats(model, device, data_loader, max_samples=5000):
    model.eval()
    model.scaling_stats = _calculate_stats(device, data_loader, model, max_samples)
    logging.info(f' using scaling stats: {model.scaling_stats.numpy()}')


def normalize_spectrogram(norm_stats, spectrograms):
    mu, sigma = norm_stats
    spectrograms = (spectrograms - mu) / sigma
    return spectrograms


def temporal_pooling(ar, frame_embeddings):
    if ar.cfg.temporal_pooling_type == 'mean':
        return mean_pooling(frame_embeddings)
    elif ar.cfg.temporal_pooling_type == 'max':
        return max_pooling(frame_embeddings)
    elif ar.cfg.temporal_pooling_type == 'mean_max':
        return mean_max_pooling(frame_embeddings)
    assert False, f'Unknown frame pooling type: {ar.cfg.temporal_pooling_type}'
