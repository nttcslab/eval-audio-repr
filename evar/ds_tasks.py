"""Downstream task definitions."""

from evar.common import (pd, np, WORK, METADATA_DIR)


_defs = {
    # folds, unit_sec, activation, data_folder (None if task name is the folder name), balanced training when fine-tining
    'us8k': [10, 4.0, 'softmax', None, False],
    'esc50': [5, 5.0, 'softmax', None, False],
    'fsd50k': [1, 7.6358, 'sigmoid', None, False], ## Changed to NOT balanced: to make it the same as PaSST.
    'fsdnoisy18k': [1, 8.25, 'softmax', None, False],
    'gtzan': [1, 30.0, 'softmax', None, False],
    'nsynth': [1, 4.0, 'softmax', None, False],
    'cremad': [1, 2.5, 'softmax', None, False],
    'spcv1': [1, 1.0, 'softmax', None, False],
    'spcv2': [1, 1.0, 'softmax', None, False],
    'surge': [1, 4.0, 'softmax', None, False],
    'vc1': [1, 8.2, 'softmax', None, False],
    'voxforge': [1, 5.8, 'softmax', None, False],
    'as20k': [1, 10.0, 'sigmoid', 'as', False],
    'as': [1, 10.0, 'sigmoid', 'as', True],
}

_fs_table = {
    16000: '16k',
    22000: '22k', # Following COALA that uses 22,000 Hz
    32000: '32k',
    44100: '44k',
    48000: '48k',
}


def get_defs(cfg, task):
    """Get task definition parameters.

    Returns:
        pathname (str): Metadata .csv file path.
        wav_folder (str): "work/16k/us8k" for example.
        folds (int): Number of LOOCV folds or 1. 1 means no cross validation.
        unit_sec (float): Unit duration in seconds.
        activation (str): Type of activation for the task: softmax for single label, sigmoid for multi-label.
        balanced (bool): True if the training requires a class-balanced sampling.
    """
    folds, unit_sec, activation, folder, balanced = _defs[task]
    folder = folder or task
    return f'{METADATA_DIR}/{task}.csv', f'{WORK}/{_fs_table[cfg.sample_rate]}/{folder}', folds, unit_sec, activation, balanced
