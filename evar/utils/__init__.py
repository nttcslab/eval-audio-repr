"""Utilities for EVAR
"""

import os
import sys
from itertools import chain
import subprocess
import re
import logging
from easydict import EasyDict
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
import random
import datetime
import hashlib
import torch
try:
    import pickle5 as pickle
except:
    import pickle


# Regular expression to check string can be converted into variables
# Thanks to -- https://stackoverflow.com/a/385597/6528729
re_valuable = re.compile("""(?x)
   ^
      (               # int|float|double
        [+-]?\ *      # first, match an optional sign *and space*
        (             # then match integers or f.p. mantissas:
            \d+       # start out with a ...
            (
                \.\d* # mantissa of the form a.b or a.
            )?        # ? takes care of integers of the form a
            |\.\d+     # mantissa of the form .b
        )
        ([eE][+-]?\d+)?  # finally, optionally match an exponent
      )
      |(              # bool
        False|True
      )
   $""")


def run_command(cmd_line):
    print('>>>', ' '.join(cmd_line))
    def runner():
        proc = subprocess.Popen(cmd_line, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        while True:
            line = proc.stdout.readline()
            if line:
                yield line

            if not line and proc.poll() is not None:
                break

    for line in runner():
        sys.stdout.write(line.decode())


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timestamp():
    """ex) Outputs 202104220830"""
    return datetime.datetime.now().strftime('%y%m%d%H%M')


def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file(), f'{path_to_config} not found, cwd={Path(".").resolve()}'
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg


def hash_text(text, L=128):
    hashed = hashlib.shake_128(text.encode()).hexdigest(L//2 + 1)
    return hashed[:L]


def setup_logger(name='', filename=None, level=logging.INFO):
    # Thanks to https://stackoverflow.com/a/53553516/6528729
    from imp import reload
    reload(logging)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=level, filename=filename)
    logger = logging.getLogger(name)
    console = logging.StreamHandler()
    console.setLevel(level)
    logger.addHandler(console)


def flatten_list(lists):
    return list(chain.from_iterable(lists))


def append_to_csv(csv_filename, data):
    filename = Path(csv_filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(filename) if filename.exists() else pd.DataFrame()
    df = pd.concat([df, data], ignore_index=True).to_csv(filename, index=False)
