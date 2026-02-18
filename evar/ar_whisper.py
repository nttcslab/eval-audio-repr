"""Wrapper code for Whisper:

Robust Speech Recognition via Large-Scale Weak Supervision

## Reference
- [1] https://openai.com/index/whisper/
- [2] https://huggingface.co/openai/whisper-large-v3
- [3] https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import logging
import torch
try:
    from transformers import WhisperModel, AutoProcessor
except:
    logging.error('Install transformers.\n>>> pip install transformers')

import torch
import math


class AR_Whisper(BaseAudioRepr):
    """
    Whisper wrapper for eval-audio-repr (evar).
    This implementation handles arbitrary audio lengths by chunking and 
    slicing the encoder outputs to match the actual input duration.
    """

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # Load the model as a base WhisperModel (Encoder + Decoder)
        # We only use the encoder part for feature extraction.
        self.model = WhisperModel.from_pretrained(cfg.pretrained_model)
        self.processor = AutoProcessor.from_pretrained(cfg.pretrained_model)

        self.sample_rate = 16000
        self.num_features = self.model.config.d_model  # hidden_size

        # Whisper encoder downsamples the input by a factor of 2.
        # Original spectrogram is 10ms per frame -> Output is 20ms per frame.
        # 16000 Hz * 0.020 s = 320 samples per latent frame.
        self.samples_per_frame = 320

    def encode_frames(self, batch_audio):
        device = batch_audio.device

        """Extract features from the encoder.
        Args:
            x (torch.Tensor): Input audio waveform of shape (batch_size, samples).
        Returns:
            torch.Tensor: Latent representations of shape (batch_size, frames, hidden_size).
        """
        _, total_samples = batch_audio.shape
        chunk_samples = 30 * self.sample_rate  # Whisper's fixed window (480,000 samples)

        all_batch_latents = []

        # Process audio in 30-second chunks
        for i in range(0, total_samples, chunk_samples):
            chunk = batch_audio[:, i : i + chunk_samples]
            actual_samples = chunk.shape[1]

            # Convert to Mel-spectrogram and pad/trim to 30s
            # Note: processor expects numpy or list on CPU
            input_features = self.processor(
                chunk.cpu().numpy(), 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_features.to(device, dtype=self.model.dtype)

            with torch.no_grad():
                # Extract hidden states from the encoder
                encoder_outputs = self.model.encoder(input_features)
                latent = encoder_outputs.last_hidden_state  # (batch, 1500, hidden_size)

            # Calculate the number of valid tokens for this chunk
            # Each token corresponds to 320 samples of the input audio.
            valid_tokens = math.ceil(actual_samples / self.samples_per_frame)

            # Slice the latent to remove padding frames
            if valid_tokens < latent.size(1):
                latent = latent[:, :valid_tokens, :]

            all_batch_latents.append(latent)

        # Concatenate all chunks along the time dimension
        features = torch.cat(all_batch_latents, dim=1)

        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))
