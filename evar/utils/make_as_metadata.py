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
import fire


def download_segment_csv():
    EVAL_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv'
    BALANCED_TRAIN_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv'
    UNBALANCED_TRAIN_URL = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'

    for subset_url in [EVAL_URL, BALANCED_TRAIN_URL, UNBALANCED_TRAIN_URL]:
        subset_path = '/tmp/' + Path(subset_url).name
        if Path(subset_path).is_file():
            continue
        with open(subset_path, 'w') as f:
            subset_data = urllib.request.urlopen(subset_url).read().decode()
            f.write(subset_data)
            print('Wrote', subset_path)


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

    ### as20k
    d = df[df.split.isin(['eval_segments', 'balanced_train_segments'])].copy()
    print('The "as20k" has', len(d), 'files out of', len(df))
    d['split'] = [{'balanced_train_segments': 'train', 'eval_segments': 'test'}[s] for s in d.split.values]
    d[['file_name', 'label', 'split']].to_csv('evar/metadata/as20k.csv', index=None)


fire.Fire(make_metadata)
