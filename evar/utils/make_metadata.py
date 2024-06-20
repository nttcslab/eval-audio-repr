"""Metadata maker for downstream tasks.

This program will create metadata files for tasks.
A metadata file is a .csv which contains file_name, label, split columns, and rows for all the audio samples
that belong to the task dataset.

Usage:
    python -m utils.preprocess_ds nsynth /path/to/nsynth
    python -m utils.preprocess_ds spcv2 /path/to/speech_commands_v0.02
    python -m utils.preprocess_ds spcv1 /path/to/speech_commands_v0.01
    python -m utils.preprocess_ds surge /path/to/surge
    python -m utils.preprocess_ds us8k /path/to/UrbanSound8K
    python -m utils.preprocess_ds vc1 /path/to/VoxCeleb1
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import requests
import fire


def flatten_list(lists):
    return list(chain.from_iterable(lists))


BASE = 'evar/metadata'


# UrbanSound8K https://urbansounddataset.weebly.com/urbansound8k.html

def convert_us8k_metadata(root):
    US8K = Path(root)
    df = pd.read_csv(US8K/f'metadata/UrbanSound8K.csv')
    df['file_name'] = df.fold.map(lambda x: f'audio/fold{x}/') + df.slice_file_name

    re_df = pd.DataFrame(df['class'].values, index=df.file_name, columns=['label'])
    re_df['fold'] = df.fold.values
    re_df.to_csv(f'{BASE}/us8k.csv')

    # test
    df = pd.read_csv(f'{BASE}/us8k.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 10
    assert len(df) == 8732
    assert np.all([fold in [1,2,3,4,5,6,7,8,9,10] for fold in df.fold.values])
    print(f'Created {BASE}/us8k.csv - test passed')


def us8k(root):
    convert_us8k_metadata(root)


# ESC-50 https://github.com/karolpiczak/ESC-50

def convert_esc50_metadata(root):
    root = Path(root)
    df = pd.read_csv(root/'meta/esc50.csv')
    repl_map = {'filename': 'file_name', 'category': 'label'}
    df.columns = [repl_map[c] if c in repl_map else c for c in df.columns]
    df.file_name = 'audio/' + df.file_name
    df.to_csv(f'{BASE}/esc50.csv', index=None)

    # test
    df = pd.read_csv(f'{BASE}/esc50.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 50
    assert len(df) == 2000
    assert np.all([fold in [1,2,3,4,5] for fold in df.fold.values])
    print(f'{BASE}/esc50.csv - test passed')


def esc50(root):
    convert_esc50_metadata(root)


# GTZAN
# Thanks to https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/gtzan/gtzan.py
# Thanks to https://github.com/xavierfav/coala
# Splits follow https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split
def convert_gtzan_metadata(root):
    # Make a list of files
    file_labels = [[str(f).replace(root + '/', ''), f.parent.name] for f in sorted(Path(root).glob('*/*.wav'))]
    df = pd.DataFrame(file_labels, columns=['file_name', 'label'])
    # Set splits
    contents = requests.get('https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/test_filtered.txt')
    _files = contents.text.splitlines()
    df.loc[df.file_name.isin(_files), 'split'] = 'test'
    contents = requests.get('https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/valid_filtered.txt')
    _files = contents.text.splitlines()
    df.loc[df.file_name.isin(_files), 'split'] = 'valid'
    contents = requests.get('https://raw.githubusercontent.com/jongpillee/music_dataset_split/master/GTZAN_split/train_filtered.txt')
    _files = contents.text.splitlines()
    df.loc[df.file_name.isin(_files), 'split'] = 'train'
    np.all(df.isna().values) == False
    df.to_csv(f'{BASE}/gtzan.csv', index=None)

    # test
    df = pd.read_csv(f'{BASE}/gtzan.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 10
    assert len(df) == 1000
    print(f'{BASE}/gtzan.csv - test passed')


def gtzan(root):
    convert_gtzan_metadata(root)


# NSynth https://magenta.tensorflow.org/datasets/nsynth

def convert_nsynth_metadata(root, nsynth='nsynth', label_column='instrument_family_str',
    n_samples=305979, files=None, filter_=lambda x: x, label_fn=None):

    def read_meta(root, mode):
        j = json.load(open(f'{root}/nsynth-{mode}/examples.json'))
        loop_indexes = files if files and mode == 'train' else j
        file_names = [f'nsynth-{mode}/audio/{file_id}.wav' for file_id in loop_indexes]
        labels = [j[x][label_column] if label_fn is None else label_fn(j[x]) for x in loop_indexes]
        return pd.DataFrame({'file_name': file_names, 'label': labels, 'split': mode})

    df = pd.concat([read_meta(root, mode) for mode in ['train', 'valid', 'test']], ignore_index=True)
    df = filter_(df)
    df.to_csv(f'{BASE}/{nsynth}.csv')

    df = pd.read_csv(f'{BASE}/{nsynth}.csv')
    assert len(df) == n_samples, f'{len(df)}'
    print(f'Created {nsynth}.csv - test passed')
    print(f'train:valid:test = {sum(df.split == "train")}:{sum(df.split == "valid")}:{sum(df.split == "test")}')


def nsynth(root):
    convert_nsynth_metadata(root)


# FSDnoisy18k http://www.eduardofonseca.net/FSDnoisy18k/

def convert_fsdnoisy18k_metadata(root):
    FSD = Path(root)
    train_df = pd.read_csv(FSD/f'FSDnoisy18k.meta/train.csv')
    # train_df = train_df[train_df.manually_verified != 0]
    # train_df = train_df[train_df.noisy_small == 0]
    test_df = pd.read_csv(FSD/f'FSDnoisy18k.meta/test.csv')
    # fname := split/fname
    train_df['fname'] = 'FSDnoisy18k.audio_train/' + train_df.fname
    test_df['fname'] = 'FSDnoisy18k.audio_test/' + test_df.fname
    # split. train -> train + val
    train_df['split'] = 'train'
    valid_index = np.random.choice(train_df.index.values, int(len(train_df) * 0.1), replace=False)
    train_df.loc[valid_index, 'split'] = 'valid'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    # filename -> file_name
    df.columns = [c if c != 'fname' else 'file_name' for c in df.columns]
    df.to_csv(f'{BASE}/fsdnoisy18k.csv', index=False)
    n_samples = len(df)

    df = pd.read_csv(f'{BASE}/fsdnoisy18k.csv')
    assert len(df) == n_samples, f'{len(df)}'
    print(f'Created fsdnoisy18k.csv - test passed')


def fsdnoisy18k(root):
    convert_fsdnoisy18k_metadata(root)


# FSD50K https://arxiv.org/abs/2010.00475

def convert_fsd50k_multilabel(FSD50K_root):
    FSD = Path(FSD50K_root)
    df = pd.read_csv(FSD/f'FSD50K.ground_truth/dev.csv')
    df['split'] = df['split'].map({'train': 'train', 'val': 'valid'})
    df['file_name'] = df.fname.apply(lambda s: f'FSD50K.dev_audio/{s}.wav')
    dftest = pd.read_csv(FSD/f'FSD50K.ground_truth/eval.csv')
    dftest['split'] = 'test'
    dftest['file_name'] = dftest.fname.apply(lambda s: f'FSD50K.eval_audio/{s}.wav')
    df = pd.concat([df, dftest], ignore_index=True)
    df['label'] = df.labels
    
    df = df[['file_name', 'label', 'split']]
    df.to_csv(f'{BASE}/fsd50k.csv')
    return df


def fsd50k(root):
    convert_fsd50k_multilabel(root)


# Speech Command https://arxiv.org/abs/1804.03209

def convert_spc_metadata(root, version=2):
    ROOT = Path(root)
    files = sorted(ROOT.glob('[a-z]*/*.wav'))
    
    labels = [f.parent.name for f in files]
    file_names = [f'{f.parent.name}/{f.name}' for f in files]
    df = pd.DataFrame({'file_name': file_names, 'label': labels})
    assert len(df) == [64721, 105829][version - 1] # v1, v2
    assert len(set(labels)) == [30, 35][version - 1] # v1, v2
    
    with open(ROOT/'validation_list.txt') as f:
        vals = [l.strip() for l in f.readlines()]
    with open(ROOT/'testing_list.txt') as f:
        tests = [l.strip() for l in f.readlines()]
    assert len(vals) == [6798, 9981][version - 1] # v1, v2
    assert len(tests) == [6835, 11005][version - 1] # v1, v2
    
    df['split'] = 'train'
    df.loc[df.file_name.isin(vals), 'split'] = 'valid'
    df.loc[df.file_name.isin(tests), 'split'] = 'test'
    assert len(df[df.split == 'valid']) == [6798, 9981][version - 1] # v1, v2
    assert len(df[df.split == 'test']) == [6835, 11005][version - 1] # v1, v2
    df.to_csv(f'{BASE}/spcv{version}.csv', index=False)

    # test
    df = pd.read_csv(f'{BASE}/spcv{version}.csv').set_index('file_name')
    assert len(df) == [64721, 105829][version - 1] # v1, v2
    print(f'Created spcv{version}.csv - test passed')


def spcv1(root):
    convert_spc_metadata(root, version=1)


def spcv2(root):
    convert_spc_metadata(root, version=2)


def vc1():
    contents = requests.get('https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt')
    texts = contents.text.splitlines()
    num_files = [text.split(' ') for text in texts]
    splits = [{'1': 'train', '2': 'valid', '3': 'test'}[nf[0]] for nf in num_files]
    files = ['wav/'+nf[1].strip() for nf in num_files]
    labels = [f.split('/')[1] for f in files]
    df = pd.DataFrame({'file_name': files, 'label': labels, 'split': splits})
    df.to_csv(f'{BASE}/vc1.csv', index=None)

    df = pd.read_csv(f'{BASE}/vc1.csv')
    assert len(df) == 153516, f'153516 != {len(df)}'
    assert len(df.label.unique()) == 1251, f'# of labels is not correct.'
    print(f'{BASE}/vc1.csv - test passed')


def surge(root):
    files = sorted(Path(root).glob('*/*.ogg'))
    tone_split_map = {tone: split for tone, split in pd.read_csv('evar/predefined/dataset_surge_tone_splits.csv').values}
    df = pd.DataFrame({'file_name': [str(f).replace(root+'/', '') for f in files],
                    'label': [f.stem for f in files],
                    'split': [tone_split_map[f.parent.name] for f in files]})
    df.to_csv(f'{BASE}/surge.csv')
    print(f'Created surge.csv, # of samples in train/valid/test splits are:')
    print(f' {sum(df.split == "train")} {sum(df.split == "valid")} {sum(df.split == "test")}')


def audiocaps(root, wav_dir='work/original/audiocaps', header=''):
    # Download wav files for AudioCaps: https://github.com/XinhaoMei/ACT?tab=readme-ov-file#set-up-dataset
    wavfiles = sorted([str(f) for f in Path(wav_dir).rglob('**/*.wav')])
    id2file = {Path(f).stem[1:12]:f for f in wavfiles}  # Yxxxxxxxxxxx.wav
    print('# of AudioSet wav files:', len(wavfiles), 'id examples:', list(id2file.keys())[:5])

    all_meta = pd.DataFrame(columns=['ytid', 'file_name', 'caption', 'split']).set_index('ytid')
    for split in ['train', 'test', 'val']:
        df = pd.read_csv(Path(root)/f'{split}.csv')
        print(split, len(df))
        missing = 0
        ids, filenames, captions = [], [], []
        for _id, cap in df[['youtube_id', 'caption']].values:
            if _id in id2file:
                ids.append(_id)
                filenames.append(Path(id2file[_id]).relative_to(wav_dir))
                captions.append(cap)
            else:
                missing += 1
        print(f'{split} -- missing:', missing)
        meta = pd.DataFrame({'ytid': ids, 'file_name': filenames, 'caption': captions})
        meta['split'] = split
        all_meta = pd.concat([all_meta, meta]) if len(all_meta) > 0 else meta
    all_meta.to_csv(f'evar/metadata/{header}audiocaps.csv', index=None)
    print(f'Created evar/metadata/{header}audiocaps.csv with # of samples:', len(all_meta))


def ja_audiocaps(root, wav_dir='work/original/audiocaps'):
    # Set root to ml-audiocaps' ja folder: /path/to/ml-audiocaps/ja
    audiocaps(root, wav_dir=wav_dir, header='ja_')


def clotho(root):
    all_meta = pd.DataFrame(columns=['file_name', 'caption', 'split'])
    for split in ['train', 'test', 'val']:
        csvsplit = {'train': 'development', 'test': 'evaluation', 'val': 'validation'}[split]
        df = pd.read_csv(Path(root)/f'clotho_captions_{csvsplit}.csv')
        dfs = [df[['file_name', f'caption_{i+1}']] for i in range(5)]
        for d in dfs:
            d.columns = ['file_name', f'caption']
        df = pd.concat(dfs).sort_values('file_name')
        df['file_name'] = csvsplit + '/' + df['file_name']
        df['split'] = split
        print(csvsplit, len(df))
        all_meta = pd.concat([all_meta, df]) if len(all_meta) > 0 else df
    all_meta.to_csv('evar/metadata/clotho.csv', index=None)
    print('Created evar/metadata/clotho.csv with # of samples:', len(all_meta))


def __making_dataset_surge_tone_splits(root):
    tones = sorted([d.name for d in Path(root).glob('*')])
    N_tones = len(tones)
    test_tone_indexes = np.random.randint(0, N_tones, size=N_tones // 10) # 10%
    rest_indexes = [i for i in range(N_tones) if i not in test_tone_indexes]
    valid_tone_indexes = np.random.choice(rest_indexes, size=N_tones // 10) # 10%
    train_tone_indexes = [i for i in rest_indexes if i not in valid_tone_indexes]
    print(len(train_tone_indexes), len(valid_tone_indexes), len(test_tone_indexes))
    df = pd.DataFrame({'tone': tones})
    df.loc[train_tone_indexes, 'split'] = 'train'
    df.loc[valid_tone_indexes, 'split'] = 'valid'
    df.loc[test_tone_indexes, 'split'] = 'test'
    df.to_csv('evar/predefined/dataset_surge_tone_splits.csv', index=False)
    # test -> fine, nothing printed.
    for i in range(N_tones):
        if i in train_tone_indexes: continue
        if i in valid_tone_indexes: continue
        if i in test_tone_indexes: continue
        print(i, 'missing')


def __making_voxforge_metadata(url_folders):
    ## CAUTION: following will not work, leaving here for providing the detail.
    N = len(url_folders)
    folders_train = list(np.random.choice(url_folders, size=int(N * 0.7), replace=False))
    rest = [folder for folder in url_folders if folder not in folders_train]
    folders_valid = list(np.random.choice(rest, size=int(N * 0.15), replace=False))
    folders_test = [folder for folder in rest if folder not in folders_valid]

    Ltrn, Lval, Ltest = len(folders_train), len(folders_valid), len(folders_test)
    print(Ltrn, Lval, Ltest, Ltrn + Lval + Ltest)
    # 9685 2075 2077 13837

    for folder in file_folders:
        if folder in folders_train:
            split = 'train'
        elif folder in folders_valid:
            split = 'valid'
        elif folder in folders_test:
            split = 'test'
        else:
            assert False
        splits.append(split)

    ns = np.array(splits)
    Ltrn, Lval, Ltest, L = sum(ns == 'train'), sum(ns == 'valid'), sum(ns == 'test'), len(ns)
    print(f'Train:valid:test = {Ltrn/L:.2f}:{Lval/L:.2f}:{Ltest/L:.2f}, total={Ltrn + Lval + Ltest}')
    # Train:valid:test = 0.69:0.15:0.16, total=176428


def __making_cremad_metadata(not_working_just_a_note):
    ## CAUTION: following will not work, leaving here for providing the detail.
    TFDS_URL = 'https://storage.googleapis.com/tfds-data/manual_checksums/crema_d.txt'

    contents = requests.get(TFDS_URL)
    urls = [line.strip().split()[0] for line in contents.text.splitlines()]
    urls = [url for url in urls if url[-4:] == '.wav'] # wav only, excluding summaryTable.csv

    filenames = [url.split('/')[-1] for url in urls]
    speaker_ids = [file_name.split('_')[0] for file_name in filenames]
    labels = [file_name.split('_')[2] for file_name in filenames]

    print(len(filenames))
    # 7438

    uniq_speakers = list(set(speaker_ids))
    N = len(uniq_speakers)
    speakers_train = list(np.random.choice(uniq_speakers, size=int(N * 0.7), replace=False))
    rest = [sp for sp in uniq_speakers if sp not in speakers_train]
    speakers_valid = list(np.random.choice(rest, size=int(N * 0.1), replace=False))
    speakers_test = [sp for sp in rest if sp not in speakers_valid]

    Ltrn, Lval, Ltest = len(speakers_train), len(speakers_valid), len(speakers_test)
    print(Ltrn, Lval, Ltest, Ltrn + Lval + Ltest)
    # 63 9 19 91

    splits = []
    for sp in speaker_ids:
        if sp in speakers_train:
            split = 'train'
        elif sp in speakers_valid:
            split = 'valid'
        elif sp in speakers_test:
            split = 'test'
        else:
            assert False
        splits.append(split)

    ns = np.array(splits)
    Ltrn, Lval, Ltest, L = sum(ns == 'train'), sum(ns == 'valid'), sum(ns == 'test'), len(ns)
    print(f'Train:valid:test = {Ltrn/L:.2f}:{Lval/L:.2f}:{Ltest/L:.2f}, total={Ltrn + Lval + Ltest}')
    # Train:valid:test = 0.69:0.10:0.21, total=7438


if __name__ == "__main__":
    Path(BASE).mkdir(parents=True, exist_ok=True)
    fire.Fire()
