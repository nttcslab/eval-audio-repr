"""Wrapper code for OWSM families:

Open Whisper-style Speech Models (OWSM, pronounced as “awesome”)

## Reference
- [1] https://www.wavlab.org/activities/2024/owsm/
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling, np
import logging
import torch
import math
try:
    from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
    from espnet2.bin.s2t_inference import Speech2Text
    import torch.nn.functional as F
    from espnet2.torch_utils.device_funcs import to_device
except:
    logging.error('Install ESPnet2.\n>>> pip install git+https://github.com/espnet/espnet@7bcb169291f5d4a9b1fd00f8bfe554de84e50024\n>>> pip install espnet_model_zoo')


def call_s2t(
    self,
    speech: Union[torch.Tensor, np.ndarray],
    text_prev: Optional[Union[torch.Tensor, np.ndarray, str, List]] = None,
    lang_sym: Optional[str] = None,
    task_sym: Optional[str] = None,
    predict_time: Optional[bool] = None,
):
    """Using inference code for a single utterance, borrowed from:
    https://github.com/espnet/espnet/blob/4bb3c8f8ac6cb111fca8021a91b275aa63f0fa57/espnet2/bin/s2t_inference.py#L383
    """

    lang_sym = lang_sym if lang_sym is not None else self.lang_sym
    task_sym = task_sym if task_sym is not None else self.task_sym
    predict_time = predict_time if predict_time is not None else self.predict_time

    lang_id = self.converter.token2id[lang_sym]
    task_id = self.converter.token2id[task_sym]
    notime_id = self.converter.token2id[self.preprocessor_conf["notime_symbol"]]

    # Prepare hyp_primer
    hyp_primer = [self.s2t_model.sos, lang_id, task_id]
    if not predict_time:
        hyp_primer.append(notime_id)

    if text_prev is not None:
        if isinstance(text_prev, str):
            text_prev = self.converter.tokens2ids(
                self.tokenizer.text2tokens(text_prev)
            )
        else:
            text_prev = text_prev.tolist()

        # Check if text_prev is valid
        if self.s2t_model.na in text_prev:
            text_prev = None

    if text_prev is not None:
        hyp_primer = [self.s2t_model.sop] + text_prev + hyp_primer

    self.beam_search.set_hyp_primer(hyp_primer)

    # Preapre speech
    if isinstance(speech, np.ndarray):
        speech = torch.tensor(speech)

    # Only support single-channel speech
    if speech.dim() > 1:
        assert (
            speech.dim() == 2 and speech.size(1) == 1
        ), f"speech of size {speech.size()} is not supported"
        speech = speech.squeeze(1)  # (nsamples, 1) --> (nsamples,)

    speech_length = int(
        self.preprocessor_conf["fs"] * self.preprocessor_conf["speech_length"]
    )
    # Pad or trim speech to the fixed length
    if speech.size(-1) >= speech_length:
        speech = speech[:speech_length]
    else:
        speech = F.pad(speech, (0, speech_length - speech.size(-1)))

    # Batchify input
    # speech: (nsamples,) -> (1, nsamples)
    speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
    # lengths: (1,)
    lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    batch = {"speech": speech, "speech_lengths": lengths}
    # logging.info("speech length: " + str(speech.size(1)))

    # a. To device
    batch = to_device(batch, device=self.device)

    # b. Forward Encoder
    enc, enc_olens = self.s2t_model.encode(**batch)

    return enc


class AR_OWSM(BaseAudioRepr):
    """
    This implementation handles arbitrary audio lengths by chunking and 
    slicing the encoder outputs to match the actual input duration.
    """

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.model = Speech2Text.from_pretrained(
            model_tag=cfg.pretrained_model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            beam_size=5,
            ctc_weight=0.0,
            maxlenratio=0.0,
            # below are default values which can be overwritten in __call__
            lang_sym="<eng>",
            task_sym="<asr>",
            predict_time=False,
        )

        self.sample_rate = 16000
        self.num_features = self.model.s2t_model.encoder._output_size

        self.samples_per_frame = 640

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

            latents_in_batch = []
            for b in range(chunk.shape[0]):
                single_wav = chunk[b]  # [Chunk_Samples]
                single_latent = call_s2t(self.model, single_wav)  # [1, Frames, Hidden_Size]
                latents_in_batch.append(single_latent)
            latent = torch.cat(latents_in_batch, dim=0) # [Batch, Frames, Hidden_Size]

            valid_tokens = math.ceil(actual_samples / self.samples_per_frame)
            if valid_tokens < latent.size(1):
                latent = latent[:, :valid_tokens, :]

            all_batch_latents.append(latent)

        # Concatenate all chunks along the time dimension
        features = torch.cat(all_batch_latents, dim=1)

        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))
