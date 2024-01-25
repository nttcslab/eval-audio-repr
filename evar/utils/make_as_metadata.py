"""AudioSet metadata maker

As the AudioSet samples become disappearing, we create the CSV metadata files for AudioSet20K and AudioSet,
while corresponding to the samples present within your environment.

To prepare audio files, please follow these steps:
- Download AudioSet samples in the .wav format.
- Convert the sampling rate of the downloaded files to your desired rate.
- Arrange the acquired files in the appropriate subfolders, namely `balanced_train_segments`,
 `eval_segments`, or `unbalanced_train_segments`, located under the "work/16k/as" folder.
  It should be noted that "16k" denotes a sampling rate of 16k Hz, and should be replaced with your rate accordingly.

To create CSV metadata files, please follow these steps:

    cd evar
    python evar/utils/make_as_metadata.py work/16k/as
    cd ..

The `make_as_metadata.py` will create `evar/metadata/as20k.csv` and `evar/metadata/as.csv`.
Please note that "16k" denotes a sampling rate of 16k Hz and should be replaced with your rate accordingly.

Example terminal outputs:

    /lab/byol-a/v2/evar$ python evar/utils/make_as_metadata.py work/16k/as
    Wrote /tmp/eval_segments.csv
    Wrote /tmp/balanced_train_segments.csv
    Wrote /tmp/unbalanced_train_segments.csv
    Official total: 2084320 , exists: 2025310
    The full "as" has 2025310 files.
    The "as20k" has 42118 files out of 2025310

"""
from re import U
import urllib.request
import json
from pathlib import Path
import pandas as pd
import numpy as np
import csv
import fire


def download_segment_csv():
    EVAL_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv'
    BALANCED_TRAIN_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv'
    UNBALANCED_TRAIN_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'
    CLASS_LABEL_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'

    for subset_url in [EVAL_URL, BALANCED_TRAIN_URL, UNBALANCED_TRAIN_URL, CLASS_LABEL_URL]:
        subset_path = '/tmp/' + Path(subset_url).name
        if Path(subset_path).is_file():
            continue
        with open(subset_path, 'w') as f:
            subset_data = urllib.request.urlopen(subset_url).read().decode()
            f.write(subset_data)
            print('Wrote', subset_path)


def gen_weight(train_files_csv, label_file, output_file):
    def make_index_dict(label_csv):
        index_lookup = {}
        with open(label_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            line_count = 0
            for row in csv_reader:
                index_lookup[row['mid']] = row['index']
                line_count += 1
        return index_lookup

    # Following AudioMAE https://github.com/facebookresearch/AudioMAE/blob/main/dataset/audioset/gen_weight.py
    index_dict = make_index_dict(label_file)
    label_count = np.zeros(527)

    df = pd.read_csv(train_files_csv)
    df = df[df.split == 'train']

    for sample in df.label.values:
        sample_labels = sample.split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    label_weight = 1000.0 / (label_count + 0.01)
    #label_weight = 1000.0 / (label_count + 100)

    sample_weight = np.zeros(len(df))
    for i, sample in enumerate(df.label.values):
        sample_labels = sample.split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            # summing up the weight of all appeared classes in the sample, note audioset is multiple-label classification
            sample_weight[i] += label_weight[label_idx]
    sample_weight = np.power(sample_weight, 1.0/1.5)  # making the weights softer
    pd.DataFrame({'file_name': df.file_name.values, 'weight': sample_weight}).to_csv(output_file, index=None)
    print('Saved AudioSet label weight as:', output_file)


def make_metadata(as_dir='work/16k/as'):
    src_folder = Path(as_dir)

    # download the original metadata.
    download_segment_csv()

    # load label maps.
    e_df = pd.read_csv('/tmp/eval_segments.csv', skiprows=2, sep=', ', engine='python')
    e_df['split'] = 'eval_segments'
    b_df = pd.read_csv('/tmp/balanced_train_segments.csv', skiprows=2, sep=', ', engine='python')
    b_df['split'] = 'balanced_train_segments'
    u_df = pd.read_csv('/tmp/unbalanced_train_segments.csv', skiprows=2, sep=', ', engine='python')
    u_df['split'] = 'unbalanced_train_segments'
    df = pd.concat([e_df, b_df, u_df])
    df = df[['# YTID', 'positive_labels', 'split']].copy()
    df.columns = ['ytid', 'label', 'split']

    # clean labels.
    def remove_quotations(s):
        assert s[0] == '"' and s[-1] == '"'
        return s[1:-1]
    df.label = df.label.apply(lambda s: remove_quotations(s))

    # mark existing samples.
    files = list(src_folder.glob('*/**/*.wav'))
    _splits = [str(f.parent).replace(as_dir, '').split('/')[1] for f in files]
    _ids = [f.stem[:11] for f in files]
    filenames = [str(f).replace(as_dir+'/', '') for f in files]
    id2split = {_id: _split for _id, _split in zip(_ids, _splits)}
    id2file = {_id: _file for _id, _file in zip(_ids, filenames)}
    df['exists'] = [(ytid in id2split) and (id2split[ytid] == split) for ytid, split in df[['ytid', 'split']].values]
    print('Official total:', len(df), ', exists:', df.exists.sum())
    # filter unavailable samples.
    df = df[df.exists].copy()
    df['file_name'] = [id2file[ytid] for ytid in df.ytid.values]
    # asserting ytids are unique...

    ## as
    d = df.copy()
    print('The full "as" has', len(d), 'files.')
    d['split'] = [{'balanced_train_segments': 'train', 'unbalanced_train_segments': 'train', 'eval_segments': 'test'}[s] for s in d.split.values]
    d[['file_name', 'label', 'split']].to_csv('evar/metadata/as.csv', index=None)

    ## as weight
    gen_weight('evar/metadata/as.csv', '/tmp/class_labels_indices.csv', 'evar/metadata/weight_as.csv')

    ### as20k
    d = df[df.split.isin(['eval_segments', 'balanced_train_segments'])].copy()
    print('The "as20k" has', len(d), 'files out of', len(df))
    d['split'] = [{'balanced_train_segments': 'train', 'eval_segments': 'test'}[s] for s in d.split.values]
    d[['file_name', 'label', 'split']].to_csv('evar/metadata/as20k.csv', index=None)


fire.Fire(make_metadata)
