import os
import argparse

import torch
import numpy as np
from tqdm import tqdm
import wget
import sys
import pandas as pd
import librosa

from benchmark.utils.audio_utils import load_audio, find_audios

sys.path.append(os.environ.get('EVAR', ''))
import evar
from lineareval import make_cfg


def select_args(config):
    args = argparse.Namespace()
    args.accelerator = config.dataset.pre_extract.accelerator
    args.output_dir = config.dataset.pre_extract.output_dir
    args.overwrite = config.dataset.pre_extract.overwrite
    args.audio_dir = config.dataset.pre_extract.audio_dir
    args.n_shard = config.args.n_shard
    args.shard_rank = config.args.shard_rank
    args.keep_folder_structure = config.dataset.pre_extract.keep_folder_structure
    args.evar_config = config.dataset.pre_extract.feature_extractor.pretrain.evar_config
    args.weight = config.dataset.pre_extract.feature_extractor.pretrain.weight
    args.options = config.dataset.pre_extract.feature_extractor.pretrain.options
    return args


class WavDataset(evar.data.BaseRawAudioDataset):
    def __init__(self, cfg, files):
        super().__init__(cfg.unit_samples, tfms=None, random_crop=False, return_filename=cfg.return_filename)
        self.cfg = cfg
        self.df = pd.DataFrame({'file_name': files})
        self.cfg.task_data = 'dummy'

    def __len__(self):
        return len(self.df)

    def get_audio(self, index):
        filename = self.df.file_name.values[index]
        wav, sr = librosa.load(filename, sr=self.cfg.sample_rate, mono=True)
        wav = torch.tensor(wav).to(torch.float32)
        return wav

    def __getitem__(self, index):
        wav = self.get_audio(index)
        return wav


def collate_trunc_wav(original_batch):
    # truncate all items to the size of the shortest item
    truncated = []
    shortest = min([b.shape[-1] for b in original_batch])
    for item in original_batch:
        l = item.shape[-1]
        if l > shortest:
            i = np.random.randint(l - shortest)
            item = item[..., i:i+shortest]
        truncated.append(item)
    return torch.stack(truncated)


def main(config):
    args = select_args(config)

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = find_audios(args.audio_dir)
    print(f'Found {len(audio_files)} audio files')

    if args.n_shard > 1:
        print(f'processing shard {args.shard_rank} of {args.n_shard}')
        audio_files.sort() # make sure no intersetction
        audio_files = audio_files[args.shard_rank * len(audio_files) // args.n_shard : (args.shard_rank + 1) * len(audio_files) // args.n_shard]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options = f'weight_file={args.weight},' + ('' if args.options is None else args.options)
    cfg, n_folds, balanced = make_cfg(args.evar_config, 'as20k', options, extras={}, abs_unit_sec=10)  # as20k is a dummy task, 10s is a dummy input unit second
    model = eval('evar.'+cfg.audio_repr)(cfg).to(device)

    batch_size = 32  # TODO make it flexible
    dataset = WavDataset(cfg, np.random.default_rng().choice(audio_files, min(len(audio_files), 1000), replace=False))  # choose random 1000< samples for calculating statistics
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_trunc_wav)
    print(f'Precomputing using the audio representation: {cfg.id} for {len(dataset)} files ({len(data_loader)} batches)')
    model.precompute(device, data_loader)

    print(f'Extracting features using {cfg.id}')
    for audio_file in tqdm(audio_files):
        # load audio
        try:
            waveform = load_audio(
                audio_file,
                target_sr=config.dataset.pre_extract.feature_extractor.pretrain.target_sr,
                is_mono=True,
                is_normalize=False,
                crop_to_length_in_sec=None,
            )
        except Exception as e:
            print(f"skip audio {audio_file} because of {e}")
            continue
        
        # extract features
        #waveform = waveform.squeeze().cpu().numpy()
        with torch.no_grad():
            embeddings = model(waveform.to(device)) # [dims]
        # reshape to [1, 1, dims]
        out = embeddings.reshape(1, 1, -1).cpu().detach().numpy()
        
        # save to npy
        if args.keep_folder_structure:
            output_file = os.path.join(
                args.output_dir,
                os.path.relpath(audio_file, args.audio_dir)+'.npy',
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            output_file = os.path.join(
                args.output_dir,
                os.path.basename(audio_file)+'.npy',
            )
        if not args.overwrite:
            assert not os.path.exists(output_file), f"{output_file} exists"
        np.save(output_file, out)
