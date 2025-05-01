"""Linear evaluation runner.

NAME
    lineareval.py

SYNOPSIS
    lineareval.py CONFIG_FILE TASK <flags>

POSITIONAL ARGUMENTS
    CONFIG_FILE
    TASK

FLAGS
    --options=OPTIONS
        Default: Nothing to change.
    --seed=SEED
        Random seed used to train and test a linear (or MLP) model.
        Default: 42
    --lr=LR
        Default: None
    --hidden=HIDDEN
        Defines a small model to solve the task.
        `()` means linear evaluation, and `(512,)` means one hidden layer with 512 units for example.
        Default: (), i.e., linear evaluation
    --standard_scaler=STANDARD_SCALER
        Default: True
    --mixup=MIXUP
        Default: False
    --epochs=EPOCHS
        Default: None
    --early_stop_epochs=EARLY_STOP_EPOCHS
        Default: None
    --step=STEP
        Default: '1pass'
"""

from evar.common import (np, pd, Path, load_yaml_config, complete_cfg, kwarg_cfg, hash_text,
                         torch, logging, seed_everything, append_to_csv,
                         app_setup_logger, setup_dir, RESULT_DIR, LOG_DIR)
from evar.data import create_dataloader
from evar.ds_tasks import get_defs
import fire
from tqdm import tqdm
from pathlib import Path
from evar.utils.torch_mlp_clf2 import TorchMLPClassifier2
# Representations
import evar.ar_spec
import evar.ar_cnn14
import evar.ar_esresnext_fbsp
import evar.ar_openl3
import evar.ar_vggish
import evar.ar_trill
import evar.ar_coala
import evar.ar_wav2vec2
import evar.ar_data2vec
import evar.ar_hubert
import evar.ar_wavlm
import evar.ar_ast
import evar.ar_m2d
import evar.ar_byola
import evar.ar_byola2
import evar.ar_data2vec
import evar.ar_atst
import evar.ar_atst_frame
import evar.ar_beats
import evar.ar_ced
import evar.ar_htsat
import evar.ar_laionclap
import evar.ar_msclap
import evar.ar_wavcaps
import evar.ar_opera
import evar.ar_dasheng


torch.backends.cudnn.benchmark = True
# Workaround for "RuntimeError: Too many open files. Communication with the workers is no longer possible."
torch.multiprocessing.set_sharing_strategy('file_system')


def get_cache_info(data_loader, _id, fold):
    cache_file = Path(f'work/cache/embs-{_id}-{data_loader.dataset.split}-{fold}.npy')
    cache_gt_file = Path(f'work/cache/embs-{_id}-{data_loader.dataset.split}-{fold}-gt.npy')
    return cache_file.exists(), cache_file, cache_gt_file


def to_embeddings(emb_ar, data_loader, device, _id=None, fold=1, cache=False):
    if len(data_loader) == 0:
        return None, None

    if cache: # Load cache
        cache_exists, cache_file, cache_gt_file = get_cache_info(data_loader, _id, fold)
        if cache_exists:
            logging.info(f' using cached embeddings: {cache_file.stem}')
            return np.load(cache_file), np.load(cache_gt_file)

    logging.info(f'Getting {_id} {data_loader.dataset.split} embeddings...')

    emb_ar.eval()
    embs, gts = [], []
    for X, y in tqdm(data_loader, mininterval=5.0):
        with torch.no_grad():
            X = X if emb_ar.module.cfg.return_filename else X.to(device)
            embs.append(emb_ar(X).detach().cpu())
        gts.append(y)
    embs = torch.vstack(embs).numpy()
    if len(gts[0].shape) > 1:
        gts = torch.vstack(gts).numpy()
    else:
        gts = torch.hstack(gts).numpy()

    if cache: # Save cache
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(cache_file, embs)
        np.save(cache_gt_file, gts)

    return embs, gts


def _one_linear_eval(X, y, X_val, y_val, X_test, y_test, hidden_sizes, epochs, early_stop_epochs, lr, classes, standard_scaler, mixup, debug): 
    logging.info(f'🚀 Started {"Linear" if hidden_sizes == () else f"MLP with {hidden_sizes}"} evaluation:')
    clf = TorchMLPClassifier2(hidden_layer_sizes=hidden_sizes, max_iter=epochs, learning_rate_init=lr,
                              early_stopping=early_stop_epochs > 0, n_iter_no_change=early_stop_epochs,
                              standard_scaler=standard_scaler, mixup=mixup, debug=debug)
    clf.fit(X, y, X_val=X_val, y_val=y_val)

    score, df = clf.score(X_test, y_test, classes)
    return score, df


def make_cfg(config_file, task, options, extras={}, cancel_aug=False, abs_unit_sec=None, original_data=False):
    cfg = load_yaml_config(config_file)
    cfg = complete_cfg(cfg, options, no_id=True)
    task_metadata, task_data, n_folds, unit_sec, balanced = get_defs(cfg, task, original_data=original_data)
    # cancel augmentation if required
    if cancel_aug:
        cfg.freq_mask = None
        cfg.time_mask = None
        cfg.mixup = 0.0
        cfg.rotate_wav = False
    # unit_sec can be configured at runtime
    if abs_unit_sec is not None:
        unit_sec = abs_unit_sec
    # update some parameters.
    update_options = f'+task_metadata={task_metadata},+task_data={task_data}'
    update_options += f',+unit_samples={int(cfg.sample_rate * unit_sec)}'
    cfg = complete_cfg(cfg, update_options, no_id=True)
    # overwrite by extra command line
    options = []
    for k, v in extras.items():
        if v is not None:
            options.append(f'{k}={v}')
    options = ','.join(options)
    cfg = complete_cfg(cfg, options)
    # Set task name
    if 'task_name' not in cfg:
        cfg['task_name'] = task
    # Return file_name instead of waveform when loading an audio
    if 'return_filename' not in cfg:
        cfg['return_filename'] = False
    # Statistics for normalization
    if 'mean' not in cfg:
        cfg['mean'] = cfg['std'] = None

    return cfg, n_folds, balanced


def short_model_desc(model, head_len=5, tail_len=1):
    text = repr(model).split('\n')
    text = text[:head_len] + ['  :'] + (text[-tail_len:] if tail_len > 0 else [''])
    return '\n'.join(text)


def lineareval_downstream(config_file, task, options='', seed=42, lr=None, hidden=(), standard_scaler=True, mixup=False,
                          epochs=None, early_stop_epochs=None, unit_sec=None, step='1pass'):
    cfg, n_folds, _ = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
    lr = lr or cfg.lr_lineareval
    epochs = epochs or 200
    early_stop_epochs = early_stop_epochs or cfg.early_stop_epochs
    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, standard_scaler=standard_scaler, mixup=mixup,
                                epochs=epochs, early_stop_epochs=early_stop_epochs)
    two_pass = (step != '1pass')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    logpath = app_setup_logger(cfg, level=logging.INFO) # Add this when debugging deeper: level=logging.DEBUG

    scores, ar, emb_ar = [], None, None
    for fold in range(1, n_folds + 1):
        # Dataloaders.
        train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg, fold=fold, seed=seed, balanced_random=False, pin_memory=False)
        logging.info(f'Train:{len(train_loader.dataset)}, valid:{len(valid_loader.dataset)}, test:{len(test_loader.dataset)}, multi label:{multi_label}')
        classes = train_loader.dataset.classes

        # Make audio representation model.
        if ar is None and step != '2pass_2_train_test':
            cache_exists, _,_ = get_cache_info(train_loader, cfg.id, fold)
            if (not two_pass) or (not cache_exists):
                ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
                ar.precompute(device, train_loader)
                emb_ar = torch.nn.DataParallel(ar).to(device)
                logging.info(short_model_desc(ar))

        # Convert to embeddings.
        X, y = to_embeddings(emb_ar, train_loader, device, cfg.id, fold=fold, cache=two_pass)
        X_val, y_val = to_embeddings(emb_ar, valid_loader, device, cfg.id, fold=fold, cache=two_pass)
        X_test, y_test = to_embeddings(emb_ar, test_loader, device, cfg.id, fold=fold, cache=two_pass)

        if step == '2pass_1_precompute_only': continue

        score, df = _one_linear_eval(X, y, X_val, y_val, X_test, y_test, hidden_sizes=hidden, epochs=epochs,
                                     early_stop_epochs=early_stop_epochs, lr=lr, classes=classes,
                                     standard_scaler=standard_scaler, mixup=mixup, debug=(fold == 1))
        scores.append(score)
        if n_folds > 1:
            print(f' fold={fold}: {score:.5f}')

    if step == '2pass_1_precompute_only': return

    mean_score = np.mean(scores)
    re_hashed = hash_text(str(cfg), L=8)
    score_file = logpath/f'{cfg.id[:-9].replace("AR_", "").replace("_", "-")}-LE_{re_hashed}_{mean_score:.5f}.csv'
    df.to_csv(score_file, index=False)

    report = f'Linear evaluation: {cfg.id[:-8]+re_hashed} {cfg.task_name} -> {mean_score:.5f}\n{cfg}\n{score_file}'
    result_df = pd.DataFrame({
        'representation': [cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]], # AR name
        'task': [cfg.task_name],
        'score': [mean_score],
        'run_id': [re_hashed],
        'report': [report],
    })
    append_to_csv(f'{RESULT_DIR}/scores.csv', result_df)
    logging.info(report)
    logging.info(f' -> {RESULT_DIR}/scores.csv')


if __name__ == '__main__':
    setup_dir([RESULT_DIR, LOG_DIR])
    fire.Fire(lineareval_downstream)
