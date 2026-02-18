"""Masked Modeling Duo (M2D) Wrapper for EVAR.

Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input
https://ieeexplore.ieee.org/document/10097236/

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079
"""

from evar.ar_base import BaseAudioRepr, BaseCLAP, calculate_norm_stats, normalize_spectrogram
import torch
import logging

try:
    import sys
    import os
    evar_home = os.getenv('EVAR', '')
    sys.path.append(os.path.join(evar_home, 'external/m2d'))
    # for backward compatibility
    sys.path.append('..')
    sys.path.append('../m2d')  # for running in external/xxx
    from m2d.runtime_audio import RuntimeM2D
except Exception as e:
    pass  # print(f'(For M2D users) Build your EVAR in your M2D folder.')


class AR_M2D(BaseAudioRepr):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg)

        if make_runtime:
            self.runtime = RuntimeM2D(cfg=cfg, weight_file=cfg.weight_file)
            self.runtime.eval()
            self.cfg = self.runtime.cfg

    def precompute(self, device, data_loader):
        if not self.cfg.mean or not self.cfg.std:
            self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)
        else:
            self.norm_stats = [self.cfg.mean, self.cfg.std]
            logging.info(f' using spectrogram norimalization stats: {self.norm_stats}')

    def precompute_lms(self, device, data_loader):
        self.precompute(device, data_loader)
        self.lms_mode = True

    def using_non_last_layer_output(self):
        if self.cfg.output_layers is None: return True
        if len(self.cfg.output_layers) > 1: return True
        return self.cfg.output_layers[0] != -1

    def to_norm_aug_lms(self, batch_audio): # Convert raw audio to normalized and augmented (only if self.training) LMS
        x = self.runtime.to_normalized_spec(batch_audio)
        x = self.augment_if_training(x)
        return x

    def encode_frames(self, batch_audio):
        x = self.to_norm_aug_lms(batch_audio)
        features = self.runtime.encode_lms(x, return_layers=self.using_non_last_layer_output())
        # stack layer outputs
        if self.using_non_last_layer_output():
            states_to_stack = [features[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in features]
            features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        if hasattr(self, 'lms_mode'):
            x = self.encode_frames_lms(batch_audio)
        else:
            x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]

    def encode_frames_lms(self, batch_lms):
        x = normalize_spectrogram(self.norm_stats, batch_lms)
        x = self.augment_if_training(x)
        hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]


class AR_M2D_CLAP(AR_M2D, BaseCLAP):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg, make_runtime=make_runtime)

    def encode_audio(self, batch_audio):
        x = self.encode_frames(batch_audio).transpose(1, 2)
        return self.runtime.project_audio(x)

    def encode_text(self, batch_text):
        return self.runtime.encode_clap_text(batch_text)


class AR_M2D_CLAP_PORTABLE(BaseCLAP):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg)
        from examples.portable_m2d import PortableM2D

        if make_runtime:
            self.runtime = PortableM2D(cfg=cfg, weight_file=cfg.weight_file)
            self.runtime.eval()
            self.cfg = self.runtime.cfg

    def precompute_lms(self, device, data_loader):
        self.precompute(device, data_loader)
        self.lms_mode = True

    def encode_frames(self, batch_audio):
        x = self.runtime.to_normalized_feature(batch_audio)
        x = self.augment_if_training(x)
        features = self.runtime.encode_lms(x)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        if hasattr(self, 'lms_mode'):
            x = self.encode_frames_lms(batch_audio)
        else:
            x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]

    def encode_frames_lms(self, batch_lms):
        x = normalize_spectrogram(self.norm_stats, batch_lms)
        x = self.augment_if_training(x)
        hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def encode_audio(self, batch_audio):
        x = self.encode_frames(batch_audio).transpose(1, 2)
        return self.runtime.project_audio(x)

    def encode_text(self, batch_text):
        return self.runtime.encode_clap_text(batch_text)


##### For ablation study on the paper:
#   "What Do Neurons Listen To? A Neuron-level Dissection of a General-purpose Audio Model"
#   Takao Kawamura, Daisuke Niizumi, and Nobutaka Ono
#   arXiv link: https://arxiv.org/abs/2602.15307

def get_activation_steering_hook(neuron_indices, gain=0.0):
    """
    Creates a forward hook to adjust the response of specific neurons.
    
    Args:
        neuron_indices (list or torch.Tensor): Indices of the target neurons to be modified.
        gain (float): Scaling factor applied to the output. 
                      - 0.0: Complete ablation (suppression).
                      - 0.0 < gain < 1.0: Attenuation (weakening).
                      - gain > 1.0: Amplification (strengthening).
    """
    def hook(module, input, output):
        # Ensure we are working with the actual activation tensor.
        # If output is a tuple (common in some models), we take the first element.
        target_output = output[0] if isinstance(output, tuple) else output
        
        # Create a mask to avoid potential in-place modification issues
        # and ensure the change propagates to the next layer.
        device = target_output.device
        mask = torch.ones(target_output.shape[-1], device=device)
        mask[neuron_indices] = gain
        
        # Apply the mask across the last dimension
        new_output = target_output * mask
        
        # Return the modified tensor in the same format as the original output
        return (new_output,) if isinstance(output, tuple) else new_output

    return hook


class AR_M2D_Steering_Neurons(AR_M2D):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg)

        layer_map = defaultdict(list)
        for l, n in cfg.int_neurons:
            layer_map[l].append(n)

        for name, module in self.runtime.backbone.named_modules():
            for layer_idx, neurons in layer_map.items():
                if name.endswith(f".{layer_idx}.mlp.act"):
                    handle = module.register_forward_hook(get_activation_steering_hook(neurons, cfg.int_gain))
                    print(f"Steering {name} neurons: {neurons} with a gain: {cfg.int_gain}")
