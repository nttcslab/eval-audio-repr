"""Wrapper code for:

WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research

## Reference
- [1] https://arxiv.org/abs/2303.17395
- [2] https://github.com/XinhaoMei/WavCaps
"""

from evar.ar_base import BaseCLAP
import sys
import torch
try:
    sys.path.append('external/WavCaps')
    from retrieval.models.ase_model import ASE
except:
    pass  # please install WavCaps


class AR_WavCaps(BaseCLAP):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        cp = torch.load(cfg.weight_file)
        config = cp["config"]
        config['audio_encoder_args']['pretrained'] = False
        model = ASE(config)
        model.load_state_dict(cp["model"], strict=False)
        self.backbone = model

    def encode_frames(self, batch_audio):
        assert False, 'encode_frames for MS CLAP is not supported for now'

    def forward(self, batch_audio):
        # Split long audio into pieces and average the features.
        features, clip_samples = [], 32000 * 10
        for chunk_index in range((batch_audio.shape[-1] + clip_samples - 1) // clip_samples):
            chunk = batch_audio[:, chunk_index*clip_samples:(chunk_index + 1)*clip_samples]
            if chunk.shape[-1] < clip_samples:  # from https://github.com/XinhaoMei/WavCaps/blob/master/retrieval/zero_shot_classification.py
                pad_length = clip_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, [0, pad_length], "constant", 0.0)
            features.append(self.backbone.encode_audio(chunk))
        features = torch.stack(features)
        features = torch.mean(features, dim=0)
        return features

    def encode_audio(self, batch_audio):
        audio_embeddings = self.forward(batch_audio)
        return audio_embeddings

    def encode_text(self, batch_text):
        text_input = self.backbone.text_encoder.tokenizer(batch_text,
                                    padding='longest',
                                    truncation=True,
                                    max_length=30,
                                    return_tensors="pt").to(self.backbone.text_encoder.device)
        text_feats = self.backbone.text_encoder.text_encoder(input_ids=text_input.input_ids,
                                    attention_mask=text_input.attention_mask)[0]
        text_feats = self.backbone.text_proj(text_feats[:, 0, :])
        return text_feats
