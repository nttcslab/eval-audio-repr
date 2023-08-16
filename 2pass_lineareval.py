"""2-pass linear evaluation runner.

This program is a wrapper for the lineareval.py enables:
- Multiple runs of the lineareval.py with a cache of embeddings.
- Evaluating Tensorflow models.

## Evaluation flow

This will run lineareval.py twice or more so that we can decouple inference and linear evaluation phase,
making it possible to use the TF model in the inference phase.

1. Run lineareval.py with `--step=2pass_1_precompute_only`.
   Conduct inference by any model (whichever TF or torch) to convert raw audio into embeddings, and store embedding in a cache.
2. Run lineareval.py with `--step=2pass_2_train_test`. Conduct linear evaluation by using embeddings from the cache using torch.
3. (if repeat > 1) Repeat the step 2 with incremented random seed.
"""

from evar.utils import run_command
import fire


def lineareval_two_pass(config_file, task, options='', lr=None, hidden=(), standard_scaler=True, mixup=False,
    early_stop_epochs=None, step=None, repeat=3, seed=None):

    seed = seed or 42
    command_line = [
        'python',
        'lineareval.py',
        config_file,
        task, 
        f'--options={options}',
        f'--lr={lr}', 
        f'--hidden={hidden}',
        f'--standard_scaler={standard_scaler}',
        f'--mixup={mixup}',
        f'--early_stop_epochs={early_stop_epochs}'
    ]

    run_command(command_line + [f'--seed={seed}', '--step=2pass_1_precompute_only'])
    for i in range(repeat):
        run_command(command_line + [f'--seed={seed + i}', '--step=2pass_2_train_test'])


if __name__ == '__main__':
    fire.Fire(lineareval_two_pass)
