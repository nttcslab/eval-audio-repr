"""Main solver program for ICBHI 2017 and SPRSound evaluation.
"""

import sys
sys.path.append('../..')
import os
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from torchinfo import summary
from dataset import ICBHI, SPRS
from ce import train_ce
import logging
logging.basicConfig(level=logging.INFO)

import evar
from evar.model_utils import set_layers_trainable
from lineareval import make_cfg
from finetune import AudioFineuneAug
from timm.models.vision_transformer import Block

# from hybrid import train_supconce
from args import args
assert args.method == 'sl'

if not(os.path.isfile(os.path.join(args.datapath, args.metadata))):
    raise(IOError(f"CSV file {args.metadata} does not exist in {args.datapath}"))

METHOD = args.method
if args.dataset == 'ICBHI': #for cross entropy
    DEFAULT_NUM_CLASSES = 4 
elif args.dataset == 'SPRS':
    DEFAULT_NUM_CLASSES = 7
from dataset import DESIRED_DURATION


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, hidden_dropout=0.5, mean=0.0, std=0.01, bias=0.):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(input_size, affine=False)
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
        x = torch.mean(x, dim=-2) # [B, T, D] -> [B, D]
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        out = self.mlp(x)
        return out

class TfmEncHead(torch.nn.Module):
    def __init__(self, embed_dim=768, output_dim=3, depth=2, heads=1, mlp_ratio=1):
        # grid_size, pos_embed
        super().__init__()
        self.sem_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sem_blocks = nn.ModuleList([
            Block(embed_dim, heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        self.linear = nn.Linear(embed_dim, output_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # append sem token
        sem_token = self.sem_token  # + self.pos_embed[:, :1, :]
        sem_tokens = sem_token.expand(x.shape[0], -1, -1)
        x = torch.cat((sem_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.sem_blocks:
            x = blk(x)
        x = x[:, 0, :]  # use semantic token only
        x = self.norm(x)

        return self.linear(x)


class TaskNetwork(torch.nn.Module):
    def __init__(self, cfg, ar, num_classes=4, head='mlp', freeze_body=False):
        super().__init__()
        set_layers_trainable(ar, trainable=(not freeze_body))
        self.ar = ar
        if head == 'mlp':
            self.head = MLP(input_size=cfg.feature_d, hidden_sizes=(128,), output_size=num_classes, mean=0.0, std=0.01, bias=0.)
        else:
            self.head = TfmEncHead(embed_dim=cfg.feature_d, output_dim=num_classes, depth=4, heads=1, mlp_ratio=1)

    def forward(self, audio):
        x = self.ar.encode_frames(audio).transpose(1, 2) # [B, D, T] -> [B, T, D]
        return self.head(x)


# Dataset and dataloaders
if args.dataset == 'ICBHI':
    train_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='train', device=args.device, samplerate=args.samplerate, pad_type=args.pad, meta_label=args.metalabel)
    val_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='test', device=args.device, samplerate=args.samplerate, pad_type=args.pad, meta_label=args.metalabel)
elif args.dataset == 'SPRS':
    train_ds = SPRS(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='train', device="cpu", samplerate=args.samplerate, pad_type=args.pad, meta_label=args.metalabel)
    if args.appmode == 'intra':
        val_ds = SPRS(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='intra_test', device="cpu", samplerate=args.samplerate, pad_type=args.pad, meta_label=args.metalabel)
    elif args.appmode == 'inter':
        val_ds = SPRS(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='inter_test', device="cpu", samplerate=args.samplerate, pad_type=args.pad, meta_label=args.metalabel)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)


cfg, n_folds, balanced = make_cfg(args.config, task='circor1', options=f'weight_file={args.weightspath}', extras={}, abs_unit_sec=DESIRED_DURATION)
cfg.ft_freq_mask = args.freqmask  # For SpecAugment
cfg.ft_time_mask = args.timemask  # For SpecAugment
cfg.flat_features = (args.head != 'mlp')
cfg.icbhi_sprs_mode = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Make a fresh model
ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
# Norm stats
if hasattr(train_loader, 'lms_mode') and train_loader.lms_mode:
    ar.precompute_lms(device, train_loader)
else:
    ar.precompute(device, train_loader)
if hasattr(ar, 'norm_stats'): print(' Normalization statistics:', ar.norm_stats)
# Augmentation
aug_fn = AudioFineuneAug(cfg.ft_freq_mask, cfg.ft_time_mask)
ar.set_augment_tf_feature_fn(aug_fn)
# Complete network
model = TaskNetwork(cfg, ar, num_classes=DEFAULT_NUM_CLASSES, head=args.head, freeze_body=args.freeze_body).to(device)
# print(' TaskNetwork', task_model)
s = summary(model, device=args.device)
nparams = s.trainable_params

### Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6) 

if args.dataset == 'ICBHI':
    if args.noweights:
        criterion_ce = nn.CrossEntropyLoss()  
    else:
        weights = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32) #N_COUNT, C_COUNT, W_COUNT, B_COUNT = 2063, 1215, 501, 363 for trainset
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        weights = weights.to(args.device)    
        criterion_ce = nn.CrossEntropyLoss(weight=weights)
else:
    criterion_ce = nn.CrossEntropyLoss()

model_name = (cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]).replace('circor1', args.dataset) + f'-{args.head}-lr{args.lr}-bs{args.bs}'
print(model_name)
print(args, flush=True)
history = train_ce(model, train_loader, val_loader, None, None, criterion_ce, optimizer, args.epochs, scheduler, DEFAULT_NUM_CLASSES, args.split_iter)
del model

report, (best_sp, best_se, best_icbhi_score, best_weight), train_losses, val_losses, train_se_scores, train_sp_scores, train_icbhi_scores, train_acc_scores, val_se_scores, val_sp_scores, val_icbhi_scores, val_acc_scores = history
scores_csv = Path('results')/(str(args.dataset).lower() + '-scores.csv')
scores_csv.parent.mkdir(parents=True, exist_ok=True)

if args.split_iter > 1: model_name += f's{args.split_iter}'
if args.freeze_embed: model_name += 'Z'
if args.adjust_pos: model_name += 'P'
text_all_args = str(dict(mode=model_name, **dict(vars(args))))
report = f'{model_name}: {report}'
print(report)

weight_path = Path('results/checkpoints')
weight_path.mkdir(parents=True, exist_ok=True)
torch.save(best_weight, weight_path/(model_name + '.pth'))

# scores
try:
    dforg = pd.read_csv(scores_csv)
except:
    print(f'Create a new {scores_csv}')
    dforg = pd.DataFrame()
df = pd.DataFrame(dict(model=[model_name], best_sp=[best_sp], best_se=[best_se], best_icbhi_score=[best_icbhi_score], report=[report], args=[text_all_args]))
pd.concat([dforg, df]).to_csv(scores_csv, index=None)

# logs
epoch_logs = dict(train_losses=train_losses, val_losses=val_losses, train_se_scores=train_se_scores, train_sp_scores=train_sp_scores,
         train_icbhi_scores=train_icbhi_scores, train_acc_scores=train_acc_scores, val_se_scores=val_se_scores,
         val_sp_scores=val_sp_scores, val_icbhi_scores=val_icbhi_scores, val_acc_scores=val_acc_scores)
df = pd.DataFrame(epoch_logs)
Path('results/logs').mkdir(parents=True, exist_ok=True)
weight_name = Path(args.weightspath).parent.name + '_' + Path(args.weightspath).stem
df.to_csv(f'results/logs/{weight_name}.csv')

del train_ds; del val_ds
del train_loader; del val_loader
