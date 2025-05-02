"""Main solver program for BMD-HS evaluation.
"""

from evar.common import (sys, np, pd, kwarg_cfg, Path,
    torch, logging, append_to_csv, RESULT_DIR)
import torchaudio
import fire

from evar.data import create_dataloader
import evar
from lineareval import make_cfg
from finetune import TaskNetwork, finetune_main

from tqdm import tqdm
import sklearn


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_gts= [], []
    for batch in loader:
        with torch.no_grad():
            X, y_gt = batch
            all_probs.append(model(X.to(device)).detach().cpu().numpy())
        all_gts.append(y_gt.numpy())
    y_score = np.vstack(all_probs)
    y_true = np.vstack(all_gts)
    return y_true, y_score


def eval_main(config_file, task, checkpoint, options='', seed=42, lr=None, hidden=(), epochs=None, early_stop_epochs=None, warmup_epochs=None,
              mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
              optim='sgd', unit_sec=None, verbose=False, data_path='work', eval_mode=None):
    
    cfg, n_folds, balanced = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
    lr = lr or cfg.ft_lr
    cfg.mixup = mixup if mixup is not None else cfg.mixup
    cfg.ft_early_stop_epochs = early_stop_epochs if early_stop_epochs is not None else cfg.ft_early_stop_epochs
    cfg.warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg.warmup_epochs
    cfg.ft_epochs = epochs or cfg.ft_epochs
    cfg.ft_freq_mask = freq_mask if freq_mask is not None else cfg.ft_freq_mask
    cfg.ft_time_mask = time_mask if time_mask is not None else cfg.ft_time_mask
    cfg.ft_rrc = rrc if rrc is not None else (cfg.ft_rrc if 'ft_rrc' in cfg else False)
    cfg.training_mask = training_mask if training_mask is not None else (cfg.training_mask if 'training_mask' in cfg else 0.0)
    cfg.ft_bs = batch_size or cfg.ft_bs
    cfg.optim = optim
    cfg.unit_sec = unit_sec
    cfg.data_path = data_path

    train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg, fold=n_folds-1, seed=seed, batch_size=cfg.ft_bs,
        always_one_hot=True, balanced_random=balanced)
    print('Classes:', train_loader.dataset.classes)
    cfg.eval_checkpoint = checkpoint

    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, mixup=cfg.mixup, bs=cfg.ft_bs,
                                freq_mask=cfg.ft_freq_mask, time_mask=cfg.ft_time_mask, rrc=cfg.ft_rrc, epochs=cfg.ft_epochs,
                                early_stop_epochs=cfg.ft_early_stop_epochs, n_class=len(train_loader.dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make a fresh model
    ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
    if hasattr(train_loader, 'lms_mode') and train_loader.lms_mode:
        ar.precompute_lms(device, train_loader)
    else:
        ar.precompute(device, train_loader)
    task_model = TaskNetwork(cfg, ar).to(device)
    task_model_dp = torch.nn.DataParallel(task_model).to(device)
    # Load checkpoint
    print('Using checkpoint', checkpoint)
    print(task_model_dp.load_state_dict(torch.load(checkpoint, map_location=device)))
    task_model_dp.eval()

    y_true, y_pred = evaluate(task_model_dp, test_loader, device)

    df = test_loader.dataset.df
    df['y_true'] = [y for y in y_true]
    df['y_pred'] = [y for y in y_pred]
    df['id'] = df.file_name.apply(lambda x: str(x).split('/')[1].split('_')[1])
    # classes = test_loader.dataset.classes
    df = df.groupby('id')[['y_true', 'y_pred']].mean()
    patient_y_true = df.y_true.apply(lambda x: np.argmax(x))
    patient_y_pred = df.y_pred.apply(lambda x: np.argmax(x))
    sample_y_true = y_true.argmax(1)
    sample_y_pred = y_pred.argmax(1)

    def calc_metrics(y_true, y_pred):
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        recall = sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        icbhi = (sensitivity + specificity) / 2.0

        report =  f'Accuracy: {accuracy:.5f}, Sensitivity: {sensitivity:.5f}, Specificity: {specificity:.5f}'
        report += f', F1 Score: {f1:.5f}, ICBHI Score: {icbhi:.5f}'
        return accuracy, sensitivity, specificity, f1, icbhi, report

    p_acc, p_sens, p_spec, p_f1, p_icbhi, p_report = calc_metrics(patient_y_true, patient_y_pred)
    s_acc, s_sens, s_spec, s_f1, s_icbhi, s_report = calc_metrics(sample_y_true, sample_y_pred)

    name  = f'{cfg.id}{"" if cfg.weight_file != "" else "/rnd"}-'
    report =  f'BMD-HS solution {name} on {task} -> \n Patient level ' + p_report + '\n Sample level ' + s_report
    report += f'\n best weight: {checkpoint}, config: {cfg}'
    logging.info(report)

    extra = f'-lr{lr}-h{hidden}-e{cfg.ft_epochs}'
    result_df = pd.DataFrame({
        'representation': [(cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]) + extra],
        'task': [task],
        'p_acc': [p_acc],
        'p_sens': [p_sens],
        'p_spec': [p_spec],
        'p_f1': [p_f1],
        'p_icbhi': [p_icbhi],
        's_acc': [s_acc],
        's_sens': [s_sens],
        's_spec': [s_spec],
        's_f1': [s_f1],
        's_icbhi': [s_icbhi],
        'weight_file': [cfg.weight_file],
        'best_checkpoint': [checkpoint],
        'run_id': [cfg.id],
        'report': [report],
    })
    append_to_csv(f'{RESULT_DIR}/bmdhs-scores.csv', result_df)


class WeightedCE:
    def __init__(self, labels) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        weights = weights / weights.sum()
        self.celoss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
        self.__name__ = f'CrossEntropyLoss(weight={weights})'
    def __call__(self, preds, gts):
        loss = self.celoss(preds, gts)
        return loss


def solve_bmdhs(config_file, task, options='', seed=42, lr=None, hidden=(128,), epochs=None, early_stop_epochs=None, warmup_epochs=None,
                  mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
                  optim='sgd', unit_sec=None, verbose=False, data_path='work', eval_only=None, eval_mode=None):

    assert task in [f'bmdhs{n}' for n in range(1, 3+1)]

    # We train a model using the original fine-tuner from the EVAR (finetune_main), and the best_path holds the path of the best weight.
    # This part is the same training process as what we have been doing in BYOL-A and M2D.
    if eval_only is None:
        cfg, n_folds, balanced = make_cfg(config_file, task, options, extras={}, abs_unit_sec=unit_sec)
        train_loader, _, _, _ = create_dataloader(cfg, fold=0, seed=0, batch_size=1, balanced_random=balanced, pin_memory=False)
        labels = train_loader.dataset.labels.numpy()
        weighted_ce = WeightedCE(labels) 
        report, scores, best_path, name, cfg, logpath = finetune_main(config_file, task, options=options, seed=seed, lr=lr, hidden=hidden, epochs=epochs,
            early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
            mixup=mixup, freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, batch_size=batch_size,
            optim=optim, unit_sec=unit_sec, freeze_ar=True, loss_fn=weighted_ce, verbose=verbose, data_path=data_path)
        del report, scores, name, cfg, logpath
    else:
        best_path = eval_only

    # Then, we evaluate the trained model specifically for the CirCor problem setting.
    return eval_main(config_file, task, best_path, options=options, seed=seed, lr=lr, hidden=hidden, epochs=epochs,
        early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
        mixup=mixup, freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, batch_size=batch_size,
        optim=optim, unit_sec=unit_sec, verbose=verbose, data_path=data_path, eval_mode=eval_mode)


if __name__ == '__main__':
    fire.Fire(solve_bmdhs)
