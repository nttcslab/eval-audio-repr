"""Audio to text and text to audio retrieval.

## Before you run

Download your copy of utils.py from WavCaps repository and install modules:

    wget https://raw.githubusercontent.com/XinhaoMei/WavCaps/master/retrieval/tools/utils.py -O evar/utils/wavcaps_utils.py
    pip install wandb loguru

## Examples

    python retr_a2t_t2a.py config/wavcaps.yaml clotho
    python retr_a2t_t2a.py config/wavcaps.yaml audiocaps
    python retr_a2t_t2a.py config/wavcaps.yaml ja_audiocaps

## Usage

NAME
    retr_a2t_t2a.py

SYNOPSIS
    retr_a2t_t2a.py CONFIG_FILE TASK <flags>

POSITIONAL ARGUMENTS
    CONFIG_FILE
    TASK

FLAGS
    -o, --options=OPTIONS
        Default: ''
"""

from evar.common import (pd, kwarg_cfg,
                         torch, logging, append_to_csv,
                         app_setup_logger, RESULT_DIR)
import fire
from tqdm import tqdm
import librosa

import evar.ar_m2d
from lineareval import make_cfg, short_model_desc
try:
    from evar.utils.wavcaps_utils import a2t, t2a
except:
    print('Download your copy of utils.py from WavCaps repository:')
    print('wget https://raw.githubusercontent.com/XinhaoMei/WavCaps/master/retrieval/tools/utils.py -O evar/utils/wavcaps_utils.py')
    print('pip install wandb loguru')

torch.backends.cudnn.benchmark = True
# Workaround for "RuntimeError: Too many open files. Communication with the workers is no longer possible."
torch.multiprocessing.set_sharing_strategy('file_system')


class AudioCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, random_crop=True, return_filename=False):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.return_filename = return_filename

        # Current split only
        df = pd.read_csv(cfg.task_metadata)
        df = df[df.split == split].reset_index()
        # Group by file name. Typical captioning dataset has 5 captions per an audio.
        df = df.groupby('file_name')['caption'].apply(list).reset_index()
        self.df = df

        self.captions = df.caption.values  # [[caption1, caption2, ...], [next sample's caption1, ...], ...]

    def __len__(self):
        return len(self.df)

    def get_audio(self, index):
        filename = self.cfg.task_data + '/' + self.df.file_name.values[index]
        if self.return_filename:
            return filename
        wav, sr = librosa.load(filename, sr=self.cfg.sample_rate, mono=True)
        wav = torch.tensor(wav).to(torch.float32)
        assert sr == self.cfg.sample_rate, f'Invalid sampling rate: {sr} Hz, expected: {self.cfg.sample_rate} Hz.'
        return wav

    def __getitem__(self, index):
        captions = self.captions[index]
        wav = self.get_audio(index)
        return wav, captions


def to_embeddings(ar, data_loader, device, _id=None, reliable_count=10):
    dataset = data_loader.dataset
    logging.info(f'Getting {_id} embeddings for {len(dataset)} samples from {dataset.split} split ...')

    ar.eval()
    audio_embs, cap_embs = [], []
    for i in tqdm(range(len(dataset)), mininterval=5.0):
        X, y = dataset[i]
        with torch.no_grad():
            X = [X] if ar.cfg.return_filename else X.unsqueeze(0).to(device)
            cur_emb = ar.encode_audio(X)
            audio_embs.append(cur_emb.squeeze(0).detach().cpu())
            cur_emb = ar.encode_text(y)
            cap_embs.extend(cur_emb.detach().cpu())
    audio_embs = torch.vstack(audio_embs).to(torch.float)
    cap_embs = torch.vstack(cap_embs).to(torch.float)
    # Repeat the audio embeddings. audio_embs=[0, 1, 2, ...], cap_embs=[0,0,0,0,0, 1,1,1,1,1, ...] -> audio_embs=[0,0,0,0,0, 1,1,1,1,1, ...]
    n_captions_per_audio = cap_embs.shape[0] // audio_embs.shape[0]
    assert n_captions_per_audio > 0
    if n_captions_per_audio > 0:
        audio_embs = audio_embs.repeat_interleave(n_captions_per_audio, dim=0)

    return audio_embs.numpy(), cap_embs.numpy()


def create_retrieval_data(cfg):
    #train_dataset = AudioCaptionDataset(cfg, 'train', random_crop=True)
    valid_dataset = AudioCaptionDataset(cfg, 'val', random_crop=False, return_filename=cfg.return_filename)
    test_dataset = AudioCaptionDataset(cfg, 'test', random_crop=False, return_filename=cfg.return_filename)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    return valid_loader, test_loader

    
def audio_text_retrieval(config_file, task, options=''):
    cfg, n_folds, _ = make_cfg(config_file, task, options, extras={}, original_data=True)
    cfg.runtime_cfg = kwarg_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    app_setup_logger(cfg, level=logging.INFO) # Add this when debugging deeper: level=logging.DEBUG

    valid_loader, test_loader = create_retrieval_data(cfg)
    ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
    ar.precompute(device, valid_loader)
    ar.eval()
    logging.info(short_model_desc(ar))

    audio_embs, cap_embs = to_embeddings(ar, test_loader, device, _id=cfg.id)
    logging.info(f'Embedding dimensions = audio:{audio_embs.shape}, caption:{cap_embs.shape}')
    results_a2t = a2t(audio_embs, cap_embs)
    results_t2a = t2a(audio_embs, cap_embs)

    split = 'test'
    logging.info('{}: Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                     'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}'.format(split, *results_a2t))
    logging.info('{}: Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                     'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}'.format(split, *results_t2a))
    df = pd.DataFrame({'model': [cfg.id], 'task': [task],
                       'a2tR1': [results_a2t[0]], 'a2tR5': [results_a2t[1]], 'a2tR10': [results_a2t[2]], 'a2tmAP10': [results_a2t[-1]],
                       't2aR1': [results_t2a[0]], 't2aR5': [results_t2a[1]], 't2aR10': [results_t2a[2]], 't2amAP10': [results_t2a[-1]],
                       'weight': [cfg.weight_file]})
    append_to_csv(f'{RESULT_DIR}/retrieval_scores.csv', df)


if __name__ == '__main__':
    fire.Fire(audio_text_retrieval)
