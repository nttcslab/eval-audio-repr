"""Wrapper code for:

wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

## Reference
- [1] https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
- [2] https://huggingface.co/facebook/wav2vec2-large-960h-lv60
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import torch
import logging
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_Wav2Vec2Logit(BaseAudioRepr):
    """Wav2Vec2.0 logits from LM output.
    https://huggingface.co/facebook/wav2vec2-large-960h-lv60
    """
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec_model)
        self.backbone = Wav2Vec2ForCTC.from_pretrained(cfg.wav2vec_model)

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        preprocessed = self.processor(batch_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        logits = self.backbone(preprocessed).logits # [B, T, D]
        return logits.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))


class AR_Wav2Vec2Context(AR_Wav2Vec2Logit):
    """Wav2Vec2.0 context network.
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1529
    """

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        preprocessed = self.processor(batch_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        features = self.backbone.wav2vec2(preprocessed, output_hidden_states=True).hidden_states # [B, T, D]
        hidden_states = self.backbone(preprocessed, output_hidden_states=True).hidden_states # [B, T, D]
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else hidden_states
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, D, T]


class AR_Wav2Vec2Feature(AR_Wav2Vec2Logit):
    """Wav2Vec2.0 feature encoder."""

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        preprocessed = self.processor(batch_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        features = self.backbone.wav2vec2.feature_extractor(preprocessed) # [B, D, T]
        features = features.transpose(1, 2) # -> [B, T, D]
        return features.transpose(1, 2) # [B, D, T]
