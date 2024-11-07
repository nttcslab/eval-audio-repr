"""Zero-shot evaluation runner.

* Download the AudioSet class label definition if you evaluate models on it. *
    wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

NAME
    zeroshot.py

SYNOPSIS
    zeroshot.py CONFIG_FILE TASK <flags>

POSITIONAL ARGUMENTS
    CONFIG_FILE
    TASK

FLAGS
    -o, --options=OPTIONS
        Default: ''
    -u, --unit_sec=UNIT_SEC
        Type: Optional[]
        Default: None
"""

from evar.common import (np, pd, Path, kwarg_cfg, hash_text,
                         torch, logging, append_to_csv,
                         app_setup_logger, RESULT_DIR)
from evar.data import create_dataloader
import fire
from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from functools import partial

import evar.ar_m2d
from lineareval import make_cfg, short_model_desc


torch.backends.cudnn.benchmark = True
# Workaround for "RuntimeError: Too many open files. Communication with the workers is no longer possible."
torch.multiprocessing.set_sharing_strategy('file_system')


def class_to_caption(task, classes):
    if task == 'cremad':
        classes = [{'ANG': 'angry person talking',
            'DIS': 'someone talking in disgust',
            'FEA': 'someone talking with a sense of fear',
            'HAP': 'someone talking happily and joyfully',
            'NEU': 'someone talking calmly',
            'SAD': 'someone talking sadly',
        }[c] for c in classes]
    elif task == 'gtzan':
        classes = [c + ' music' for c in classes]  # 0.7482758620689656
    elif task == 'nsynth':
        classes = ['the musical instrument sound of ' + c for c in classes]  # 0.319
    elif task == 'as':
        df = pd.read_csv('class_labels_indices.csv')
        labelmap = {k:v for k, v in df[['mid', 'display_name']].values}
        classes = [labelmap[c] for c in classes]

    captions = [x.replace('_', ' ') + " can be heard" for x in classes]
    return captions


def to_embeddings(ar, data_loader, device, _id=None, fold=1):
    if len(data_loader) == 0:
        return None, None

    logging.info(f'Getting {_id} {data_loader.dataset.split} embeddings...')

    ar.eval()
    embs, gts = [], []
    for X, y in tqdm(data_loader, mininterval=5.0):
        with torch.no_grad():
            X = X if ar.cfg.return_filename else X.to(device)
            cur_emb = ar.encode_audio(X)
            embs.append(cur_emb.detach().cpu())
        gts.append(y)
    embs = torch.vstack(embs).to(torch.float)
    if len(gts[0].shape) > 1:
        gts = torch.vstack(gts)
    else:
        gts = torch.hstack(gts)

    return embs, gts


def is_zeroshot_ready(cfg):
    return True


def eval_map(y_score, y_true):
    average_precision = metrics.average_precision_score(
        y_true, y_score)
    return average_precision


def eval_acc(y_score, y_true):
    preds = np.argmax(y_score, axis=-1)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=preds)
    return accuracy


def zeroshot_downstream(config_file, task, options='', unit_sec=None):
    cfg, n_folds, _ = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec, original_data=True)
    seed = 42
    cfg.runtime_cfg = kwarg_cfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logpath = app_setup_logger(cfg, level=logging.INFO) # Add this when debugging deeper: level=logging.DEBUG
    if not is_zeroshot_ready(cfg):
        logging.info(f'ZS not supported with {config_file}')
        return

    ar, caption_embeddings = None, None
    y_preds = []
    y_labels = []
    for fold in range(1, n_folds + 1):
        train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg, fold=fold, seed=seed, balanced_random=False, pin_memory=False, always_wav=True)
        logging.info(f'Train:{len(train_loader.dataset)}, valid:{len(valid_loader.dataset)}, test:{len(test_loader.dataset)}, multi label:{multi_label}')
        activation_fn = torch.nn.functional.sigmoid if multi_label else partial(torch.nn.functional.softmax, dim=1)

        if ar is None:
            ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
            ar.precompute(device, train_loader)
            ar.eval()
            logging.info(short_model_desc(ar))

        # caption embeddings
        if caption_embeddings is None:
            classes = train_loader.dataset.classes
            captions = class_to_caption(task, classes)
            print('Captions:', captions[:3], '...')
            # convert one by one to save memory for a large text encoder. -- caption_embeddings = ar.encode_text(captions).detach().cpu()
            embs = [ar.encode_text([c]).detach().cpu() for c in captions]
            caption_embeddings = torch.vstack(embs)

        # audio embeddings
        audio_embeddings, gts = to_embeddings(ar, test_loader, device, _id=cfg.id, fold=fold)

        # zero-shot inference
        similarity = ar.compute_similarity(audio_embeddings, caption_embeddings)
        y_pred = activation_fn(similarity.detach().cpu()).numpy()
        y_preds.append(y_pred)
        y_labels.append(gts.detach().cpu().numpy())

    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    metric_fn = eval_map if multi_label else eval_acc
    result = metric_fn(y_preds, y_labels)
    print(f'{task} result: {result}')

    re_hashed = hash_text(str(cfg), L=8)

    task_name = 'zs_'+task if task == cfg.task_name else cfg.task_name

    report = f'Zero-shot evaluation: {cfg.id[:-8]+re_hashed} {task_name} -> {result:.5f}\n{cfg}'
    result_df = pd.DataFrame({
        'representation': [cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]], # AR name
        'task': [task_name],
        'score': [result],
        'run_id': [re_hashed],
        'report': [report],
    })
    append_to_csv(f'{RESULT_DIR}/scores.csv', result_df)
    logging.info(report)
    logging.info(f' -> {RESULT_DIR}/scores.csv')


if __name__ == '__main__':
    fire.Fire(zeroshot_downstream)
