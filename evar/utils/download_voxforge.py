"""
Download VoxForge dataset to your "to_folder".
This code uses a file list from TFDS, downloads .tgz files, and extract them.
The definition of labels and data splits is available in evar/metadata/voxforge.csv.

Following TFDS implementation for the details.

## Usage

'''sh
python download_voxforge.py <to_folder>
'''

## Reference

- [1] http://www.voxforge.org/
- [2] TFDS: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/voxforge.py

@article{maclean2018voxforge,
    title={Voxforge},
    author={MacLean, Ken},
    journal={Ken MacLean.[Online]. Available: http://www.voxforge.org/home.[Acedido em 2012]},
    year={2018}
}
"""

import urllib.request
import shutil
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import fire


TFDS_URL = 'https://storage.googleapis.com/tfds-data/downloads/voxforge/voxforge_urls.txt'


def _download_extract_worker(args):
    url, filename, dest_path = args

    if (Path(dest_path)/Path(filename).stem).exists():
        #print(' skip', Path(filename).stem)
        #print('.', end='')
        return

    tmpfile = '/tmp/' + filename
    try:
        urllib.request.urlretrieve('http://' + url, tmpfile)
    except:
        print('ERROR to download', url)
        return
    try:
        shutil.unpack_archive(tmpfile, dest_path)
    except:
        print('ERROR to extract', url)

    os.remove(tmpfile)


def download_extract_voxforge(dest_path):
    file = urllib.request.urlopen(TFDS_URL)
    urls = [line.decode('utf-8').strip() for line in file]
    filenames = [url.split('/')[-1] for url in urls]
    assert len(set(filenames)) == len(urls)

    print('Downloading voxforge for', len(urls), 'tgz archives.')
    Path(dest_path).mkdir(exist_ok=True, parents=True)
    with Pool() as p:
        args = [[url, filename, dest_path] for url, filename in zip(urls, filenames)]
        shapes = list(tqdm(p.imap(_download_extract_worker, args), total=len(args)))

    print('finished.')


if __name__ == "__main__":
    fire.Fire(download_extract_voxforge)
