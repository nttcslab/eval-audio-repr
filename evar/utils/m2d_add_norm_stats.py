"""A small utility for M2D fine-tuned by EVAR.
This utility adds a parameter "module.ar.runtime.backbone.norm_stats" to a checkpoint file with constant normalization statistic values [-7.1, 4.2].
These values are the dataset average and standard deviation when pre-trained on AudioSet with M2D.

Usage: python [this script] [source checkpoint file] [output checkpoint file]
"""

import torch
import sys

src_file = sys.argv[1]
dest_file = sys.argv[2]

checkpoint = torch.load(src_file, map_location='cpu')
if 'module.ar.runtime.backbone.cls_token' not in checkpoint:
    print(f'{src_file} is not a fine-tuned checkpoint; no "module.ar.runtime.backbone.cls_token".')
    exit(1)

checkpoint['module.ar.runtime.backbone.norm_stats'] = torch.tensor([-7.1, 4.2])
torch.save(checkpoint, dest_file)
print(f'Saved {dest_file} with an additional parameter "module.ar.runtime.backbone.norm_stats".')
