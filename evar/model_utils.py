"""Model utilities.
"""

import logging
from pathlib import Path
import torch
from torch import nn


def ensure_weights(filename, url):
    """Ensures thar `filename` exists, or download from the `url`"""

    if not Path(filename).is_file():
        import urllib.request
        logging.info(f'Downloading {url} as {filename} ...')
        urllib.request.urlretrieve(url, filename)


def load_pretrained_weights(model, pathname, model_key='model', strict=True):
    state_dict = torch.load(pathname)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model' in state_dict:
        state_dict = state_dict['model']
    children = sorted([n + '.' for n, _ in model.named_children()])

    # 'model.xxx' -> 'xxx"
    weights = {}
    for k in state_dict:
        weights[k[len(model_key)+1:] if k.startswith(model_key+'.') else k] = state_dict[k]
    state_dict = weights

    # model's parameter only
    def find_model_prm(k):
        for name in children:
            if name in k: # ex) "conv_block1" in "model.conv_block1.conv1.weight"
                return k
        return None

    weights = {}
    for k in state_dict:
        if find_model_prm(k) is None: continue
        weights[k] = state_dict[k]

    logging.info(f' using network pretrained weight: {Path(pathname).name}')
    print(list(weights.keys()))
    logging.info(str(model.load_state_dict(weights, strict=strict)))
    return sorted(list(weights.keys()))


def set_layers_trainable(layer, trainable=False):
    for n, p in layer.named_parameters():
        p.requires_grad = trainable


def show_layers_trainable(layer, show_all_trainable=True):
    total_params = sum(p.numel() for p in layer.parameters())
    total_trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params:,} (trainable {total_trainable_params:,})')
    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
    frozen = [n for n, p in layer.named_parameters() if not p.requires_grad]
    print('Trainable parameters:', trainable if show_all_trainable else f'{trainable[:10]} ...')
    print('Others are frozen such as:', frozen[:3], '...' if len(frozen) >= 3 else '')


def initialize_layers(layer):
    # initialize all childrens first.
    for l in layer.children():
        initialize_layers(l)

    # initialize only linaer
    if type(layer) != nn.Linear:
        return

    # Thanks to https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/models.py#L10
    logging.debug(f' initialize {layer}.weight')
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            logging.debug(f' initialize {layer}.bias')
            layer.bias.data.fill_(0.)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, hidden_dropout=0.5, mean=0.0, std=0.01, bias=0.):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        fcs = []
        for l, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            if l > 0:
                fcs.append(nn.Dropout(hidden_dropout))
            linear = nn.Linear(in_size, out_size)
            nn.init.normal_(linear.weight, mean=mean, std=std)
            nn.init.constant_(linear.bias, bias)
            fcs.append(linear)
            fcs.append(nn.ReLU())
        self.mlp = nn.Sequential(*fcs[:-1])

    def forward(self, x):
        out = self.mlp(x)
        return out


def mean_max_pooling(frame_embeddings, dim=-1):
    assert len(frame_embeddings.shape) == 3 # Batch,Feature Dimension,Time
    (x1, _) = torch.max(frame_embeddings, dim=dim)
    x2 = torch.mean(frame_embeddings, dim=dim)
    x = x1 + x2
    return x


def mean_pooling(frame_embeddings, dim=-1):
    assert len(frame_embeddings.shape) == 3 # Batch,Feature Dimension,Time
    x2 = torch.mean(frame_embeddings, dim=dim)
    return x2


def max_pooling(frame_embeddings, dim=-1):
    assert len(frame_embeddings.shape) == 3 # Batch,Feature Dimension,Time
    (x1, _) = torch.max(frame_embeddings, dim=dim)
    return x1
