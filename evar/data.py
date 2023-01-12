"""Dataset handlings.

Balanced sampler is supported for multi-label tasks.
"""

from .common import (np, pd, torch, torchaudio, F)
from .sampler import BalancedRandomSampler, InfiniteSampler
import multiprocessing
from sklearn.preprocessing import MultiLabelBinarizer


class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, unit_samples, tfms=None, random_crop=False):
        self.unit_samples = unit_samples
        self.tfms = tfms
        self.random_crop = random_crop

    def __len__(self):
        raise NotImplementedError

    def get_audio(self, index):
        raise NotImplementedError

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        wav = self.get_audio(index) # shape is expected to be (self.unit_samples,)

        # Trim or stuff padding
        l = len(wav)
        if l > self.unit_samples:
            start = np.random.randint(l - self.unit_samples) if self.random_crop else 0
            wav = wav[start:start + self.unit_samples]
        elif l < self.unit_samples:
            wav = F.pad(wav, (0, self.unit_samples - l), mode='constant', value=0)
        wav = wav.to(torch.float)

        # Apply transforms
        if self.tfms is not None:
            wav = self.tfms(wav)

        # Return item
        label = self.get_label(index)
        return wav if label is None else (wav, label)


class WavDataset(BaseRawAudioDataset):
    def __init__(self, cfg, split, holdout_fold=1, always_one_hot=False, random_crop=True, classes=None):
        super().__init__(cfg.unit_samples, tfms=None, random_crop=random_crop)
        self.cfg = cfg
        self.split = split

        df = pd.read_csv(cfg.task_metadata)
        # Multi-fold, leave one out of cross varidation.
        if 'split' not in df.columns:
            assert 'fold' in df.columns, '.csv needs to have either "split" or "fold" column...'
            # Mark split either 'train' or 'test', no 'val' or 'valid' used in this implentation.
            df['split'] = df.fold.apply(lambda f: 'test' if f == holdout_fold else 'train')
        df = df[df.split == split].reset_index()
        self.df = df
        self.multi_label = df.label.map(str).str.contains(',').sum() > 0

        if self.multi_label or always_one_hot:
            # one-hot
            oh_enc = MultiLabelBinarizer()
            self.labels = torch.tensor(oh_enc.fit_transform([str(ls).split(',') for ls in df.label]), dtype=torch.float32)
            self.classes = oh_enc.classes_
        else:
            # single valued gt values
            self.classes = sorted(df.label.unique()) if classes is None else classes
            self.labels = torch.tensor(df.label.map({l: i for i, l in enumerate(self.classes)}).values)

    def __len__(self):
        return len(self.df)

    def get_audio(self, index):
        filename = self.cfg.task_data + '/' + self.df.file_name.values[index]
        wav, sr = torchaudio.load(filename)
        assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {filename} has {sr} Hz.'
        return wav[0]

    def get_label(self, index):
        return self.labels[index]


def create_dataloader(cfg, fold=1, seed=42, batch_size=None, always_one_hot=False, balanced_random=False, pin_memory=True):
    batch_size = batch_size or cfg.batch_size
    train_dataset = WavDataset(cfg, 'train', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True)
    valid_dataset = WavDataset(cfg, 'valid', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True,
        classes=train_dataset.classes)
    test_dataset = WavDataset(cfg, 'test', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)

    train_sampler = BalancedRandomSampler(train_dataset, batch_size, seed) if train_dataset.multi_label else \
        InfiniteSampler(train_dataset, batch_size, seed, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=pin_memory,
                                            num_workers=multiprocessing.cpu_count()) if balanced_random else \
                   torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                            num_workers=multiprocessing.cpu_count())
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=multiprocessing.cpu_count())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=multiprocessing.cpu_count())

    return (train_loader, valid_loader, test_loader, train_dataset.multi_label)
