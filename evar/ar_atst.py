"""Wrapper code for:

ATST: Audio Representation Learning with Teacher-Student Transformer

## Reference
- [1] https://arxiv.org/abs/2204.12076 Xian Li and Xiaofei Li, “ATST: Audio Representation Learning with Teacher-Student Transformer,” arXiv preprint arXiv:2204.12076, 2022.
- [2] https://github.com/Audio-WestlakeU/audiossl
"""

from evar.ar_base import (BaseAudioRepr, calculate_norm_stats)
import torch
import torchaudio
import logging
import sys
try:
    sys.path.append('external/audiossl')
    from audiossl.models.atst.audio_transformer import AST, AST_base
except Exception as e:
    class AST:  # dummy
        pass
    pass  # logging.info(f'Make your copy of AST under external folder. Check Preparing-models.md for the details.')

from torch.nn import functional as F
from torchvision import transforms


class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(CustomAudioTransform):
    def __init__(self,min,max):
        self.min=min
        self.max=max
    def __call__(self,input):
        min_,max_ = None,None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_)/(max_- min_) *2. - 1.
        return input


class CentralCrop(CustomAudioTransform):
    def __init__(self, size:int, pad:bool=True):
        self.size = size
        self.pad = pad

    def __call__(self, signal):

        if signal.shape[1] < self.size :
            if self.pad:
                signal = F.pad(signal, (0, self.size-signal.shape[1]))
            return signal

        start = (signal.shape[1] - self.size) // 2
        return signal[:, start: start + self.size]


class FinetuneTrainTransform:
    def __init__(self,sr=16000,max_len=12):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64).to('cuda')
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        self.sr=sr
        self.len=len

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )

        self.global_transform = transforms.Compose(
                                [CentralCrop(int(sr*max_len),pad=False),
                                self.mel_feature,
                                ]
                                )

    def __call__(self,input):
        output=self.global_transform(input)
        return output


class PretrainedEncoderPLModule(torch.nn.Module):
    def __init__(self,
                 pretrained_encoder: AST,
                 chunk_len: float, # 6.0
                 n_blocks: int, # 12
                 avgpool:bool = True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.chunk_len = int(chunk_len*16000/160 + 1)
        self.n_blocks = n_blocks
        self.avgpool = avgpool
        if avgpool:
            self.embed_dim = self.encoder.embed_dim*2*n_blocks
        else:
            self.embed_dim = self.encoder.embed_dim*n_blocks

    def forward(self, x):
        length = torch.tensor([x.shape[-1] for _ in range(x.shape[0])]).to(x.device)
        x = self.encoder.get_intermediate_layers_chunks(x,
                                                        length,
                                                        self.n_blocks,
                                                        self.chunk_len,
                                                        avgpool=self.avgpool)
        return x


def load_pretrained_weights(model, pretrained_weights, checkpoint_key="teacher"):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
 

class ATST_Feature(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tfm = FinetuneTrainTransform()

    def forward(self, waveforms):
        device = waveforms.device
        # import pdb; pdb.set_trace()
        fbanks = torch.stack([self.tfm(w.unsqueeze(0)) for w in waveforms])
        return fbanks.to(device)


class AR_ATST(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ATST_Feature(cfg)
        self.base = AST_base()
        load_pretrained_weights(self.base, cfg.weight_file)
        self.backbone = PretrainedEncoderPLModule(self.base, chunk_len=6.0, n_blocks=cfg.n_blocks)

    def encode_frames(self, batch_audio):
        # AST returns a single embeddings for one audio, then simply add time axis.
        return self.forward(batch_audio).unsqueeze(-1) # B,D -> B,D,1

    def forward(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = self.augment_if_training(x)
        x = self.backbone(x)
        return x
