"""Dataset handlings.

Balanced sampler is supported for multi-label tasks.
"""

from .common import (np, pd, torch, F, Path)
from .sampler import BalancedRandomSampler, InfiniteSampler
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import WeightedRandomSampler
import librosa


class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, unit_samples, tfms=None, random_crop=False, return_filename=False):
        self.unit_samples = unit_samples
        self.tfms = tfms
        self.random_crop = random_crop
        self.return_filename = return_filename

    def __len__(self):
        raise NotImplementedError

    def get_audio(self, index):
        raise NotImplementedError

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        label = self.get_label(index)

        if self.return_filename:
            fn = self.cfg.task_data + '/' + self.df.file_name.values[index]  # requires self.cfg & self.df to be set in advance.
            return fn if label is None else (fn, label)
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
        return wav if label is None else (wav, label)


class WavDataset(BaseRawAudioDataset):
    def __init__(self, cfg, split, holdout_fold=1, always_one_hot=False, random_crop=True, classes=None):
        super().__init__(cfg.unit_samples, tfms=None, random_crop=random_crop, return_filename=cfg.return_filename)
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
        wav, sr = librosa.load(filename, sr=(self.cfg.sample_rate if '/original/' in self.cfg.task_data else None), mono=True)
        wav = torch.tensor(wav).to(torch.float32)
        assert sr == self.cfg.sample_rate, f'Invalid sampling rate: {sr} Hz, expected: {self.cfg.sample_rate} Hz.'
        return wav

    def get_label(self, index):
        return self.labels[index]


class ASSpectrogramDataset(WavDataset):
    """Spectrogram audio dataset class for M2D AudioSet 2M fine-tuning."""

    def __init__(self, cfg, split, always_one_hot=False, random_crop=True, classes=None):
        super().__init__(cfg, split, holdout_fold=1, always_one_hot=always_one_hot, random_crop=random_crop, classes=classes)

        self.df.file_name = self.df.file_name.str.replace('.wav', '.npy', regex=False)
        self.folder = Path(cfg.data_path)
        assert cfg.dur_frames == 1001, 'Set dur_frames=1001 or any number that suits your model.'
        self.crop_frames = cfg.dur_frames
        self.random_crop = random_crop

        print(f'Dataset ({split}) contains {len(self.df)} files without normalizing stats.')

    def get_audio_file(self, filename):
        lms = torch.tensor(np.load(filename))
        return lms

    def get_audio(self, index):
        filename = self.folder/self.df.file_name.values[index]
        return self.get_audio_file(filename)

    def complete_audio(self, lms):
        # Trim or pad
        start = 0
        l = lms.shape[-1]
        if l > self.crop_frames:
            start = int(torch.randint(l - self.crop_frames, (1,))[0]) if self.random_crop else 0
            lms = lms[..., start:start + self.crop_frames]
        elif l < self.crop_frames:
            pad_param = []
            for i in range(len(lms.shape)):
                pad_param += [0, self.crop_frames - l] if i == 0 else [0, 0]
            lms = F.pad(lms, pad_param, mode='constant', value=0)
        self.last_crop_start = start
        lms = lms.to(torch.float)

        return lms

    def __getitem__(self, index):
        lms = self.get_audio(index)
        lms = self.complete_audio(lms)
        # Return item
        label = self.get_label(index)
        return lms if label is None else (lms, label)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(crop_frames={self.crop_frames}, random_crop={self.random_crop}, '
        return format_string


def create_as_dataloader(cfg, batch_size, always_one_hot=False, balanced_random=False, pin_memory=True, num_workers=8):
    batch_size = batch_size or cfg.batch_size
    train_dataset = ASSpectrogramDataset(cfg, 'train', always_one_hot=always_one_hot, random_crop=True)
    valid_dataset = ASSpectrogramDataset(cfg, 'valid', always_one_hot=always_one_hot, random_crop=True,
        classes=train_dataset.classes)
    test_dataset = ASSpectrogramDataset(cfg, 'test', always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)

    weights = pd.read_csv('evar/metadata/weight_as.csv').weight.values
    train_sampler = WeightedRandomSampler(weights, num_samples=200000, replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=pin_memory,
                                            num_workers=num_workers) if balanced_random else \
                   torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    train_loader.lms_mode = True

    return (train_loader, valid_loader, test_loader, train_dataset.multi_label)


def create_dataloader(cfg, fold=1, seed=42, batch_size=None, always_one_hot=False, balanced_random=False, pin_memory=True, num_workers=8, always_wav=False):
    if Path(cfg.task_metadata).stem == 'as' and not always_wav:
        return create_as_dataloader(cfg, batch_size=batch_size, always_one_hot=always_one_hot, balanced_random=balanced_random, pin_memory=pin_memory, num_workers=num_workers)

    batch_size = batch_size or cfg.batch_size
    train_dataset = WavDataset(cfg, 'train', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True)
    valid_dataset = WavDataset(cfg, 'valid', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True,
        classes=train_dataset.classes)
    test_dataset = WavDataset(cfg, 'test', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)

    train_sampler = BalancedRandomSampler(train_dataset, batch_size, seed) if train_dataset.multi_label else \
        InfiniteSampler(train_dataset, batch_size, seed, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=pin_memory,
                                            num_workers=num_workers) if balanced_random else \
                   torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)

    return (train_loader, valid_loader, test_loader, train_dataset.multi_label)
