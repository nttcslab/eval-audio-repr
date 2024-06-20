"""Downstream task definitions."""

from evar.common import (pd, np, WORK, METADATA_DIR)


_defs = {
    # folds, unit_sec, data_folder (None if task name is the folder name), balanced training when fine-tining
    'us8k': [10, 4.0, None, False],
    'esc50': [5, 5.0, None, False],
    'fsd50k': [1, 7.6358, None, False], ## Changed to NOT balanced: to make it the same as PaSST.
    'fsdnoisy18k': [1, 8.25, None, False],
    'gtzan': [1, 30.0, None, False],
    'nsynth': [1, 4.0, None, False],
    'cremad': [1, 2.5, None, False],
    'spcv1': [1, 1.0, None, False],
    'spcv2': [1, 1.0, None, False],
    'surge': [1, 4.0, None, False],
    'vc1': [1, 8.2, None, False],
    'voxforge': [1, 5.8, None, False],
    'as20k': [1, 10.0, 'as', False],
    'as': [1, 10.0, 'as', True],
    'audiocaps': [1, 10.0, None, False],
    'ja_audiocaps': [1, 10.0, 'audiocaps', False],
    'clotho': [1, 30.0, None, False],
}

_fs_table = {
    16000: '16k',
    22000: '22k', # Following COALA that uses 22,000 Hz
    32000: '32k',
    44100: '44k',
    48000: '48k',
}

def get_original_folder(task, folder):
    orgs = {
        'us8k': 'UrbanSound8K',
        'esc50': 'ESC-50-master',
        'as20k': 'AudioSet',
        'as': 'AudioSet',
    }
    return orgs[task] if task in orgs else folder


def get_defs(cfg, task, original_data=False):
    """Get task definition parameters.

    Returns:
        pathname (str): Metadata .csv file path.
        wav_folder (str): "work/16k/us8k" for example.
        folds (int): Number of LOOCV folds or 1. 1 means no cross validation.
        unit_sec (float): Unit duration in seconds.
        weighted (bool): True if the training requires a weighted loss calculation.
        balanced (bool): True if the training requires a class-balanced sampling.
    """
    folds, unit_sec, folder, balanced = _defs[task]
    folder = folder or task
    workfolder = f'{WORK}/original/{get_original_folder(task, folder)}' if original_data else f'{WORK}/{_fs_table[cfg.sample_rate]}/{folder}'
    return f'{METADATA_DIR}/{task}.csv', workfolder, folds, unit_sec, balanced
