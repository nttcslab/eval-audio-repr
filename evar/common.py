"""Common imports/constants/small functions."""

from evar.utils import *
import shutil
import re
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


def eval_if_possible(text):
    for pat in [r'\[.*\]', r'\(.*\)']:
        if re.search(pat, text):
            return eval(text)
    if re_valuable.match(text):
        return eval(text)
    return text


def split_camma(text):
    flag = None
    elements = []
    cur = []
    for c in text:
        if flag is not None:
            cur.append(c)
            if flag == '[' and c == ']': flag = None
            if flag == '(' and c == ')': flag = None
            if flag == '"' and c == '"': flag = None
            if flag == "'" and c == "'": flag = None
            continue
        if c in ['[', '(', '"', "'"]:
            cur.append(c)
            flag = c
            continue
        if c == ',':
            elements.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
            elements.append(''.join(cur))
    return elements


# App level utilities
def complete_cfg(cfg, options, no_id=False):
    # Override parameter values with given "options".
    if 'name' not in cfg or not isinstance(cfg['name'], str):
        cfg['name'] = ''
    print(options)
    for item in split_camma(options):
        if item == '': continue
        keyvalues = item.split('=')
        assert len(keyvalues) == 2, f'An option need one and only one "=" in the option {item} in {options}.'
        key, value = keyvalues
        value = eval_if_possible(value)
        if key[0] == '+':
            key = key[1:]
            cfg[key] = None
        if key not in cfg.keys():
            raise Exception(f'Cannot find a setting named: {key} of the option {item}')
        cfg[key] = value
    # Set ID.
    if not no_id:
        task = Path(cfg.task_metadata).stem if 'task_metadata' in cfg else ''
        if 'name' in cfg and len(cfg['name']) > 0:
            name = cfg.name
        elif 'weight_file' in cfg and len(str(cfg['weight_file'])) > 0:
            weight_path = Path(str(cfg['weight_file']))
            parent = weight_path.parent.stem if len(weight_path.parent.stem) > 0 else str(cfg.audio_repr.split(',')[-1])
            name = f'{parent}-{weight_path.stem}'
        else:
            str(cfg.audio_repr.split(',')[-1])
        cfg.id = name + '_' + task + '_' + hash_text(str(cfg), L=8)
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
