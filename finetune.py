"""Fine-tuning evaluator.
"""

from evar.common import (sys, np, pd, EasyDict, kwarg_cfg,
    torch, F, logging, append_to_csv, app_setup_logger, seed_everything, RESULT_DIR)
import fire
import time
from sklearn import metrics, utils

import torchaudio
import timm.scheduler
import timm.optim

from evar.data import create_dataloader
from evar.model_utils import show_layers_trainable, MLP
from lineareval import *


torch.backends.cudnn.benchmark = True


# copied and modified from https://github.com/nttcslab/byol-a
import random
class RandomResizeCrop(torch.nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward_one(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = torch.nn.functional.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True).squeeze(0)
        return lms.to(torch.float)

    def forward(self, lms):
        if len(lms.shape) == 3:
            return self.forward_one(lms)
        for i in range(len(lms)):
            lms[i] = self.forward_one(lms[i])
        return lms

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string


class SpecAugment:
    @staticmethod
    def is_required(freqm, timem):
        if freqm > 0:
            return True
        if timem > 0:
            return True
        return False

    def __init__(self, freqm, timem):
        self.freqmask = torchaudio.transforms.FrequencyMasking(freqm) if freqm > 0 else None
        self.timemask = torchaudio.transforms.TimeMasking(timem) if timem > 0 else None

    def __call__(self, lms):
        if self.freqmask is not None:
            lms = self.freqmask(lms)
        if self.timemask is not None:
            lms = self.timemask(lms)
        return lms


class AudioFineuneAug:
    def __init__(self, freqm, timem, rrc=False):
        self.spec_aug = SpecAugment(freqm, timem) if SpecAugment.is_required(freqm, timem) else None
        self.rrc = RandomResizeCrop() if rrc else None
        if self.spec_aug is not None:
            logging.info(f' using SpecAugmentation with {freqm}, {timem}.')
        if self.rrc is not None:
            logging.info(f' using {self.rrc}')

    def __call__(self, lms):
        lms = lms if self.spec_aug is None else self.spec_aug(lms)
        lms = lms if self.rrc is None else self.rrc(lms)
        return lms


def loss_nll(logits, gts):
    # Args: logits: (N, C), gts: (N, C) [0,1] hot soft label after mixup is applied.
    preds = F.log_softmax(logits, dim=-1)
    loss = -torch.mean(gts * preds)
    return loss


def loss_bce(logits, gts):
    return F.binary_cross_entropy_with_logits(logits, gts) # no need to apply F.sigmoid(logits)


def eval_map(y_score, y_true, classes):
    average_precision = metrics.average_precision_score(
        y_true, y_score, average=None)
    auc = metrics.roc_auc_score(y_true, y_score, average=None)
    return average_precision.mean(), pd.DataFrame({'ap': average_precision, 'auc': auc, 'class': classes})


def eval_acc(y_score, y_true, classes):
    preds = np.argmax(y_score, axis=-1)
    labels = np.argmax(y_true, axis=-1)
    accuracy = labels == preds
    def class_name(indexed): return [classes[l] for l in indexed]
    return accuracy.mean(), pd.DataFrame({'GT': class_name(labels), 'prediction': class_name(preds)})


class Mixup(object):
    def __init__(self, mixup_alpha=0.1):
        self.mixup_alpha = mixup_alpha
        logging.info(f' using mixup with alpha={mixup_alpha}')

    def get_lambda(self, batch_size, device):
        lambdas = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
        self.lambdas = torch.tensor(lambdas).to(torch.float).to(device)
        self.counter_indexes = np.random.permutation(batch_size)

    def __call__(self, x_and_y):
        if self.mixup_alpha == 0.0:
            return x_and_y
        def do_mixup(x, mixup_lambda):
            x = x.transpose(0, -1)
            out = x * mixup_lambda + x[..., self.counter_indexes] * (1.0 - mixup_lambda)
            return out.transpose(0, -1)
        self.get_lambda(len(x_and_y[0]), x_and_y[0].device)
        x_and_y = [do_mixup(z, self.lambdas) for z in x_and_y]
        return x_and_y


def evaluate(model, loader, device, eval_fn, classes):
    model.eval()
    all_probs, all_gts= [], []
    for batch in loader:
        with torch.no_grad():
            X, y_gt = batch
            all_probs.append(model(X.to(device)).detach().cpu().numpy())
        all_gts.append(y_gt.numpy())
    y_score = np.vstack(all_probs)
    y_true = np.vstack(all_gts)
    return eval_fn(y_score, y_true, classes)


def arg_conf_str(args, defaults={
    'lr': (0.0, 'lr', 'z'),
    'mixup': (0.0, 'mu', 'z'),
    'freq_mask': (0, 'fm', 'asis'),
    'time_mask': (0, 'tm', 'asis'),
    'balanced': (False, 'bal', 'b'),
    'warmup_epochs': (5, 'wu', 'asis'),
    'seed': (42, 's', 'asis'),
    'training_mask': (0.0, 'tx', 'z'),
    'rrc': (False, 'R', 'b'),
    'optim': ('sgd', 'O', 'asis'),
    'unit_sec': (None, 's', 'asis'),
}):
    confstr = ''
    for k in defaults:
        try:
            arg_value = eval('args.' + k)
        except:
            continue # no parameter k for the run.
        if arg_value == defaults[k][0]:
            continue
        arg_key, value_format = defaults[k][1:]
        value = str(arg_value)
        if value_format == 'z':
            value = value.replace('0.', '')
        elif value_format == 'b':
            value = '' # nothing to add
        elif value_format == 'head':
            value = value[:1]
        confstr += arg_key + value
    return confstr


def _train(cfg, ar_model, device, logpath, train_loader, valid_loader, test_loader, multi_label, seed, lr, balanced, verbose):
    classes = train_loader.dataset.classes

    loss_fn = loss_bce if multi_label else loss_nll
    eval_fn = eval_map if multi_label else eval_acc
    crit_str = 'mAP' if eval_fn == eval_map else 'acc'
    optimizer = {
        'adamw': torch.optim.AdamW(ar_model.parameters(), lr=lr, weight_decay=0.0001, betas=(0.9, 0.95), eps=1e-08, amsgrad=True),
        'sgd': torch.optim.SGD(ar_model.parameters(), lr, momentum=0.9, weight_decay=0),
        'lars': timm.optim.Lars(ar_model.parameters(), lr, momentum=0.9, weight_decay=0),
        'lamb': timm.optim.Lamb(ar_model.parameters(), lr),
    }[cfg.optim]
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=cfg.ft_epochs, lr_min=1e-7, warmup_t=cfg.warmup_epochs, warmup_lr_init=0)
    logging.info(f'Using {loss_fn.__name__}, {eval_fn.__name__}, and {optimizer}')

    # Training begins here.
    time1 = time.time()
    best_result, best_path, best_epoch = 0.0, None, 0
    epoch_iters = len(train_loader)
    console_iters = max(10, epoch_iters // 10)

    # Set test set as validation set if not available; i.e., val result = test result in this case.
    if len(valid_loader.dataset) == 0:
        print(' ** Fine-tuning using Evaluation set result as test result **')
        valid_loader = test_loader

    # Augmentations for fine tuning
    mixup = Mixup(mixup_alpha=cfg.mixup)
    aug_fn = AudioFineuneAug(cfg.ft_freq_mask, cfg.ft_time_mask, rrc=cfg.ft_rrc)
    ar_model.module.ar.set_augment_tf_feature_fn(aug_fn)

    # Name this session
    name  = f'{cfg.id}{"" if cfg.weight_file != "" else "/rnd"}-'
    name += arg_conf_str(EasyDict({'mixup': cfg.mixup, 'freq_mask': cfg.ft_freq_mask, 'time_mask': cfg.ft_time_mask,
        'rrc': cfg.ft_rrc, 'lr': lr, 'warmup_epochs': cfg.warmup_epochs, 'balanced': balanced, 'seed': seed, 'training_mask': cfg.training_mask,
        'optim': cfg.optim, 'unit_sec': cfg.unit_sec}))

    for epoch in range(cfg.ft_epochs):
        for iter, batch in enumerate(train_loader):
            # Train
            ar_model.train()
            X_aug, y_aug = mixup(batch)
            X_aug, y_aug = X_aug.to(device), y_aug.to(device)

            probs = ar_model(X_aug)
            loss = loss_fn(probs, y_aug)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            micro_epoch = epoch + iter/epoch_iters
            scheduler.step(micro_epoch)

            if iter % console_iters == 0:
                logging.info(f'Epoch [{epoch}] iter: {iter}/{epoch_iters}, elapsed: {time.time() - time1:.3f}s,'
                           + f' lr: {optimizer.param_groups[0]["lr"]:.8f} loss: {float(loss):.8f}')
                time1 = time.time()

            # balanced training = infinity training iterations -> manually break
            if balanced and iter + 1 >= epoch_iters:
                break

        # Epoch done -> Evaluate
        print('validating')
        val_result, df = evaluate(ar_model, valid_loader, device, eval_fn, classes)
        report = f'{name} | epoch/iter {epoch}/{iter}: '
        report += f'val {crit_str}: {val_result:.5f}, loss: {float(loss):.5f}'

        # Save the best model
        new_best_record = best_result < val_result
        if new_best_record: # following PANNs implementation, measuring potential performance.
            best_result = val_result
            best_epoch = epoch
            if best_path is not None:
                best_path.unlink()
            best_path = logpath/f'weights_ep{epoch}it{iter}-{val_result:.5f}_loss{loss:.4f}.pth'
            torch.save(ar_model.state_dict(), best_path)
            logging.info(f'Saved weight as {best_path}')
            df.to_csv(logpath/f'ep{epoch}it{iter}-{val_result:.5f}.csv')
        report += f', best: {best_result:.5f}@{best_epoch}'

        # Report to log and dashboard
        logging.info(report)

        # Stop condition
        if cfg.ft_early_stop_epochs > 0 and epoch > best_epoch + cfg.ft_early_stop_epochs:
            logging.info(f'Early stopping now, the best epoch was {best_epoch}.')
            break

    # Test result
    if valid_loader != test_loader:
        logging.info(f'Load best weight from {best_path}')
        ar_model.load_state_dict(torch.load(best_path))
        print('testing')
        best_result, df = evaluate(ar_model, test_loader, device, eval_fn, classes)
        logging.info(f'Final test {crit_str}: {best_result:.5f}')
    else:
        logging.info(f'Best {crit_str}: {best_result:.5f}')

    return best_result, best_path, name


class TaskHead(torch.nn.Module):
    def __init__(self, dim, n_class=1000, hidden=()):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(dim, affine=False)
        self.mlp = MLP(input_size=dim, hidden_sizes=hidden, output_size=n_class, mean=0.0, std=0.01, bias=0.)

    def forward(self, x):
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        return self.mlp(x)


class TaskNetwork(torch.nn.Module):
    def __init__(self, cfg, ar):
        super().__init__()
        self.cfg = EasyDict(cfg.copy())
        self.ar = ar
        print(cfg.feature_d, cfg.runtime_cfg.hidden, cfg.runtime_cfg.n_class)
        self.head = TaskHead(dim=cfg.feature_d, n_class=cfg.runtime_cfg.n_class, hidden=cfg.runtime_cfg.hidden)

        print('Backbone encoder:')
        show_layers_trainable(self.ar, show_all_trainable=False)
        print('Head:')
        show_layers_trainable(self.head)

    def forward(self, batch_audio):
        x = self.ar(batch_audio)
        x = self.head(x)
        return x # returning logits, not probs


def run_eval(config_file, task, options='', seed=42, lr=None, hidden=(), mixup=None, batch_size=None,
                          epochs=None, early_stop_epochs=None, warmup_epochs=None,
                          freq_mask=None, time_mask=None, rrc=None, training_mask=None,
                          optim='sgd', unit_sec=None, verbose=True, eval_only=False, data_path='work'):
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

    # Make audio representation model and downstream task model.
    train_loader, _, _, _ = create_dataloader(cfg, fold=0, seed=seed, batch_size=cfg.ft_bs, balanced_random=balanced, pin_memory=False)

    cfg.runtime_cfg = kwarg_cfg(lr=lr, seed=seed, hidden=hidden, mixup=cfg.mixup, bs=cfg.ft_bs,
                                freq_mask=cfg.ft_freq_mask, time_mask=cfg.ft_time_mask, rrc=cfg.ft_rrc, epochs=cfg.ft_epochs,
                                early_stop_epochs=cfg.ft_early_stop_epochs, n_class=len(train_loader.dataset.classes))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)
    logpath = app_setup_logger(cfg, level=logging.INFO) # Add this when debugging deeper: level=logging.DEBUG

    scores = []
    for fold in range(1, n_folds + 1):
        logging.info(f'\n🚀 Start fine-tuning {f"fold#{fold}/{n_folds}" if n_folds > 1 else ""} with logging in {logpath}')

        # Dataloaders for current fold.
        train_loader, valid_loader, test_loader, multi_label = create_dataloader(cfg, fold=fold, seed=seed, batch_size=cfg.ft_bs,
            always_one_hot=True, balanced_random=balanced)
        logging.info(f'Train:{len(train_loader.dataset)}, valid:{len(valid_loader.dataset)}, test:{len(test_loader.dataset)}, multi label:{multi_label}, balanced:{balanced}')

        # Make a fresh model
        ar = eval('evar.'+cfg.audio_repr)(cfg).to(device)
        if hasattr(train_loader, 'lms_mode') and train_loader.lms_mode:
            ar.precompute_lms(device, train_loader)
        else:
            ar.precompute(device, train_loader)
        task_model = TaskNetwork(cfg, ar).to(device)
        task_model_dp = torch.nn.DataParallel(task_model).to(device)
        logging.info(f'Head = {task_model.head}')

        if eval_only:
            task_model_dp.load_state_dict(torch.load(cfg.weight_file, map_location=device))
            eval_fn, crit_str = (eval_map, 'mAP') if multi_label else (eval_acc, 'acc')
            best_result, df = evaluate(task_model_dp, test_loader, device, eval_fn, train_loader.dataset.classes)
            logging.info(f'Evaluation result of {crit_str}: {best_result:.5f}')
            return [best_result], cfg.weight_file, 'eval only', cfg, logpath

        best_result, best_path, name = _train(cfg, task_model_dp, device, logpath, train_loader, valid_loader, test_loader,
            multi_label, seed, lr, balanced, verbose)

        scores.append(best_result)
        if n_folds > 1:
            print(f' fold={fold}: {best_result:.5f}')

    return scores, best_path, name, cfg, logpath


def finetune_main(config_file, task, options='', seed=42, lr=None, hidden=(), epochs=None, early_stop_epochs=None, warmup_epochs=None,
                  mixup=None, freq_mask=None, time_mask=None, rrc=None, training_mask=None, batch_size=None,
                  optim='sgd', unit_sec=None, verbose=False, eval_only=False, data_path='work'):
    scores, best_path, name, cfg, logpath = run_eval(config_file, task, options=options, seed=seed, lr=lr, hidden=hidden, mixup=mixup,
        batch_size=batch_size, epochs=epochs, early_stop_epochs=early_stop_epochs, warmup_epochs=warmup_epochs,
        freq_mask=freq_mask, time_mask=time_mask, rrc=rrc, training_mask=training_mask, optim=optim,
        unit_sec=unit_sec, verbose=verbose, eval_only=eval_only, data_path=data_path)
    mean_score = np.mean(scores)
    report = f'Finetuning {name} on {cfg.task_name} -> mean score: {mean_score:.5f}'
    if eval_only:
        print(report)
        return '', scores, best_path, name, cfg, logpath

    score_file = logpath/f'{cfg.task_name}_{cfg.audio_repr.replace("AR_", "").replace("_", "-")}-FT_{cfg.id[-8:]}_{mean_score:.5f}.csv'
    best_report = logpath/(best_path.stem.split('_')[1] + '.csv')
    best_report.rename(score_file)

    if len(scores) > 1:
        report += ', scores: [' + ', '.join([f'{score:.5f}' for score in scores]) + ']'
    report += f', best weight: {best_path}, score file: {score_file}, config: {cfg}'
    logging.info(report)

    result_df = pd.DataFrame({
        'representation': [cfg.id.split('.')[-1][3:-9] if '.AR_' in cfg.id else cfg.id[:-9]], # AR name
        'task': [cfg.task_name],
        'score': [mean_score],
        'run_id': [cfg.id],
        'report': [report],
    })
    append_to_csv(f'{RESULT_DIR}/ft-scores.csv', result_df)
    return report, scores, best_path, name, cfg, logpath


if __name__ == '__main__':
    fire.Fire(finetune_main)
