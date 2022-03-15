"""Common imports/constants/small functions."""

from evar.utils import *
import shutil
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from torch.utils.data import DataLoader, Dataset


# Folders
WORK = 'work'
METADATA_DIR = 'evar/metadata'
RESULT_DIR = 'results'
LOG_DIR = 'logs'


# App level utilities
def complete_cfg(cfg, options, no_id=False):
    # Override with options.
    for item in options.split(','):
        if item == '': continue
        keyvalues = item.split('=')
        assert len(keyvalues) == 2, f'An option need one and only one "=" in the option {item} in {options}.'
        key, value = keyvalues
        if re_valuable.match(value):
            value = eval(value)
        else:
            value = keyvalues[1] # use option value string as is
        if key[0] == '+':
            key = key[1:]
            cfg[key] = None
        if key not in cfg.keys():
            raise Exception(f'Cannot find a setting named: {key} of the option {item}')
        cfg[key] = value
    # Set ID.
    if not no_id:
        task = Path(cfg.task_metadata).stem if 'task_metadata' in cfg else ''
        name = cfg.name if 'name' in cfg else str(cfg.audio_repr.split(',')[-1])
        cfg.id = task + '_' + name + '_' + hash_text(str(cfg), L=8)
    return cfg


def kwarg_cfg(**kwargs):
    cfg = EasyDict(kwargs)
    cfg.id = hash_text(str(cfg), L=8)
    return cfg


def app_setup_logger(cfg, level=logging.INFO):
    logpath = Path(LOG_DIR)/cfg.id
    logpath.mkdir(parents=True, exist_ok=True)
    setup_logger(filename=logpath/'log.txt', level=level)
    print('Logging to', logpath/'log.txt')
    logging.info(str(cfg))
    return logpath


def setup_dir(dirs=[]):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
