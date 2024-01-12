"""Wrapper code for:

HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection

## Reference
- [1] https://arxiv.org/abs/2202.00874
- [2] https://github.com/RetroCirce/HTS-Audio-Transformer
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import sys
import logging
import torch
try:
    sys.path.append('external/htsat')
    from model.htsat import HTSAT_Swin_Transformer
    import config
except:
    pass  # please install HTS-AT


class AR_HTSAT(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # load the pre-trained checkpoints
        checkpoint = torch.load(cfg.weight_file)
        logging.info(f' Using weight_file: {cfg.weight_file}')

        self.backbone = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head)

        states, L = {}, len('sed_model.')
        for k in checkpoint["state_dict"]:
            new_k = k[L:] if k.startswith('sed_model.') else k
            states[new_k] = checkpoint["state_dict"][k]
        self.backbone.load_state_dict(states)
        # cfg = checkpoint['config']

    def encode_frames(self, batch_audio):
        assert False, 'encode_frames for HTS-AT is not supported for now'

    def forward(self, batch_audio):
        # Split long audio into pieces and average the features.
        features = []
        for chunk_index in range((batch_audio.shape[-1] + config.clip_samples - 1) // config.clip_samples):
            chunk = batch_audio[:, chunk_index*config.clip_samples:(chunk_index + 1)*config.clip_samples]
            features.append(self.backbone(chunk, mixup_lambda=None, infer_mode=True)['latent_output'])
        features = torch.stack(features)
        features = torch.mean(features, dim=0)
        return features