"""
Download CREMA-D dataset to your "to_folder".
This code uses a file list from TFDS and downloads .wav files only.
The definition of labels and data splits is available in evar/metadata/cremad.csv.

Following NOSS [2] split. We assign 70 % of speakers (63) as training, 10 % (9) as validation,
and the remaining 20 % (19) as test splits, with no speaker duplication in multiple splits.

## Usage

'''sh
python download_cremad.py <to_folder>
'''

## Reference

- [1] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova and R. Verma, "CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset," in IEEE Transactions on Affective Computing, vol. 5, no. 4, pp. 377-390, 1 Oct.-Dec. 2014, doi: 10.1109/TAFFC.2014.2336244.
- [2] J. Shor, A. Jansen, R. Maor, O. Lang, O. Tuval, F. d. C. Quitry, M. Tagliasacchi, I. Shavitt, D. Emanuel, and Y. Haviv, “Towards learning a universal non-semantic representation of speech,” in Interspeech, Oct 2020.
- [3] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/crema_d.py
"""

import urllib.request
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import fire


TFDS_URL = 'https://storage.googleapis.com/tfds-data/manual_checksums/crema_d.txt'


def _download_worker(args):
    url, dest_path = args
    filename = url.split('/')[-1]

    if (Path(dest_path)/Path(filename).name).exists():
        print(' skip', Path(filename).stem)
        return

    destfile = f'{dest_path}/{filename}'
    try:
        urllib.request.urlretrieve(url, destfile)
    except:
        print('ERROR to download', url)


def download_extract_cremad(dest_path):
    lines = urllib.request.urlopen(TFDS_URL)
    urls = [line.decode('utf-8').strip().split()[0] for line in lines]
    urls = [url for url in urls if url[-4:] == '.wav'] # wav only, excluding summaryTable.csv

    print('Downloading CREMA-D for', len(urls), 'wav files.')
    Path(dest_path).mkdir(exist_ok=True, parents=True)
    with Pool() as p:
        args = [[url, dest_path] for url in urls]
        shapes = list(tqdm(p.imap(_download_worker, args), total=len(args)))

    print('finished.')


if __name__ == "__main__":
    fire.Fire(download_extract_cremad)
