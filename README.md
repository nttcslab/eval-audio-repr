# EVAR ~ Evaluation package for Audio Representations

This repository offers a comprehensive evaluation package for audio representations (ARs) as employed in our papers. Its key features include:

- Reproducible evaluation across a variety of audio downstream tasks, with prefixed train/valid/test set splits provided.
- A unified AR interface for ease of use.
- Support linear evaluation, zero-shot evaluation, and fine-tuning.
- Support for 12+ tasks and 10+ models.

In early 2021, we lacked a cohesive codebase for evaluating models across various tasks under consistent test settings, which prompted the creation of this repository.
By the end of 2021, other similar options, such as ([SERAB](https://github.com/Neclow/serab/), [SUPERB](https://superbbenchmark.org/), [HEAR 2021 NeurIPS Challenge](https://neuralaudio.ai/hear2021-datasets.html), and [HARES](https://arxiv.org/abs/2111.12124)), had emerged. However, this repository was developed independently for our specific study.

This evaluation package is intended for researchers who wish to compare ARs under the same test setup as employed in our study.
The papers used EVAR are:

- M2D (TASLP 2024): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Masked Modeling Duo: Towards a Universal Audio Pre-Training Framework,” IEEE/ACM Trans. Audio, Speech, Language Process., vol. vol. 32, pp. 2391-2406, 2024](http://dx.doi.org/10.1109/TASLP.2024.3389636).* 👉 [GitHub](https://github.com/nttcslab/m2d)
- M2D-CLAP (Interspeech 2024): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, M. Yasuda, S. Tsubaki, and K. Imoto, "M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representation," to appear at Interspeech 2024](https://arxiv.org/abs/2406.02032).* 👉 [GitHub](https://github.com/nttcslab/m2d/tree/master/clap)
- Niizumi et al. (IEEE EMBC 2024): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection," to appear at IEEE EMBC, 2024](https://arxiv.org/abs/2404.17107).* 👉  [GitHub](https://github.com/nttcslab/m2d/tree/master/app/circor)
- BYOL-A (TASLP 2023): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “BYOL for Audio: Exploring pre-trained general-purpose audio representations,” IEEE/ACM Trans. Audio, Speech, Language Process., vol. 31, pp. 137–151, 2023](http://dx.doi.org/10.1109/TASLP.2022.3221007).* 👉 [GitHub](https://github.com/nttcslab/byol-a/tree/master/v2)
- SELFIE (ICASSP 2023) *[B. Nguyen, S. Uhlich and F. Cardinaux, "Improving Self-Supervised Learning for Audio Representations by Feature Diversity and Decorrelation," in ICASSP, 2023](https://ieeexplore.ieee.org/document/10097100).*
- M2D (ICASSP 2023): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input," in ICASSP, 2023](https://ieeexplore.ieee.org/document/10097236).* 👉 [GitHub](https://github.com/nttcslab/m2d)
- M2D for Speech (Interspeech 2023): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation," in Interspeech, 2023](https://www.isca-speech.org/archive/interspeech_2023/niizumi23_interspeech.html).* 👉 Corrected [arXiv](https://arxiv.org/abs/2305.14079) version 👉 [GitHub](https://github.com/nttcslab/m2d/tree/master/speech)
- MSM-MAE (HEAR 2021, PMLR): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation," in HEAR: Holistic Evaluation of Audio Representations (NeurIPS 2021 Competition), vol. 166, 2022, pp. 1–24](https://arxiv.org/abs/2204.12260).* 👉  [GitHub](https://github.com/nttcslab/msm-mae)
- Niizumi et al. (EUSIPCO 2022): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "Composing General Audio Representation by Fusing Multilayer Features of a Pre-trained Model," in EUSIPCO, 2022](https://arxiv.org/abs/2205.08138).* 👉  [GitHub](https://github.com/nttcslab/composing-general-audio-repr)
- BYOL-A (IJCNN 2021): *[D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, "BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation," in IJCNN, 2021](https://arxiv.org/abs/2103.06695).* 👉 [GitHub](https://github.com/nttcslab/byol-a)

## Update History

### Jun 20, 2024 -- Supported autio to text or text to audio retrieval (ATR).
- ATR evaluator: `retr_a2t_t2a.py`

### Mar 25, 2024 -- Fixed minor issues.
- New feature: Fine-tuning program supports a new option "eval-only".
- Refactoring: Logging folder name so that it uses the weight path name.
- Fix: Small issues with CLAP model wrappers.

### Jan 25, 2024 -- Supported zero-shot evaluation for CLAP models.
- Zero-shot evaluator: `zeroshot.py`
- New model: Supported (for linear and zero-shot evaluation) WavCaps, LAION CLAP, and MS CLAP.
- Fix: An issue related to resampling quality by migrating from torchaudio to soundfile+librosa.
- Fix: M2D to use fixed normalization statistic values.

<details><summary>Older history</summary>

### Jan 12, 2024 -- Supported weighted CE loss with fine-tuning and added more models.
- Loss function: Supported (for fine-tuning) weighted cross entropy loss.
- New model: Supported (for linear evaluation) ATST/ATST-Frame, BEATs, CED, HTSAT.

### Jan 17, 2023 -- Supported evaluating multilayer features by stacking layer-wise features.
- Added the `output_layers` option in the wav2vec2/data2vec/hubert/wavlm config files.

### Jan 12, 2023 -- Supported Fine-tuning on AudioSet20K and additional models.
- Added the **fine-tuning** script for the evaluations of M2D.
- New task: Supported AudioSet20K.
- New model: Supported (for linear evaluation) data2vec, HuBERT, and WavLM.
- New model: Supported (for linear evaluation and fine-tuning) BYOL-A (v2, TASLP 2023).

</details>

## 1. Quick start (Linear evaluation)

The following show how to prepare CREMA-D dataset and evaluate OpenL3 (music) features on CREMA-D.

0. Follow the steps in "2-1. Step 1: Install modeules, and download depending source code", in short:

    ```sh
    git clone https://github.com/nttcslab/eval-audio-repr.git evar
    cd evar
    curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py
    curl https://raw.githubusercontent.com/XinhaoMei/WavCaps/master/retrieval/tools/utils.py -o evar/utils/wavcaps_utils.py
    pip install -r requirements.txt
    ```

1. Download CREMA-D dataset. This downloads all the .wav files under a folder `downloads/cremad`.

    ```
    $ python evar/utils/download_cremad.py downloads/cremad
    ```

2. Preprocess (resample) data samples. This will make copies of all the .wav files under `downloads/cremad` to `work/48k/cremad` with a sampling rate of 48,000 Hz.

    $ python prepare_wav.py downloads/cremad work/48k/cremad 48000

3. Prepare OpenL3 code and weight. Our implementation ([evar/ar_openl3.py](evar/ar_openl3.py)) uses [torchopenl3](https://github.com/torchopenl3/torchopenl3).

    $ pip install torchopenl3

4. Evaluate. The 48,000 Hz .wav files from `work/48k/cremad` are encoded to mbedding vectors by the OpenL3, then linear evaluation program taks the embeddings as input. The result will be appended to a file `results/scores.csv`.

    $ python lineareval.py config/openl3mus.yaml cremad

## 2. Setup

**Warning**: Setup takes long, especially downloading datasets.

You will:

1. Install modeules, and download external source code.
2. Download datasets and create metadata files.
3. Download model implementation and weights.

### 2-0. Step 0: Clone as `evar`.
To make it easy, we clone as `evar`.

```sh
git clone https://github.com/nttcslab/eval-audio-repr.git evar
```

### 2-1. Step 1: Install modeules, and download depending source code
Run following once to download your copy of the external source code.

```sh
curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py
```

Install modules listed on [requirements.txt](requirements.txt).
If you use Anaconda, you might create an environment as the following example:

```sh
conda create -n evar python=3.8
conda activate evar
pip install -r requirements.txt
```

### 2-2. Step 2: Setup datasets

See 👉 [Preparing-datasets.md](Preparing-datasets.md).

### 2-3. Step 3: Setup models

See 👉 [Preparing-models.md](Preparing-models.md).

## 3. Linear evaluation

The following describes the evaluation steps with an exemplar command line:

    $ python lineareval.py config/openl3mus.yaml cremad

- The main program (`lineareval.py`) converts CREMA-D audio samples into embedding vectors by calling the OpenL3 model.
  - OpenL3 requires 48 kHz audio, thus samples located in the `work/48k` folder are used.
  - The model (OpenL3 in this example) is instantiated according to the config file (config/openl3.yaml). The config file defines the detail of the model instance, such as the pre-trained weight file to load.
- The main program trains a linear model utilizing `TorchMLPClassifier2`, an MLPClassifier implementation near compatible with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
- The main program evaluates the trained linear model with test samples. It reports `accuracy` for multi-class single label classification tasks or `mAP` for multi-class multi-label tasks.

The followings show the structure of the folders:

```
evar/
  evar           Evaluation codes.
  evar/utils     Helper utilitiy codes.
  evar/metadata  <SOME CSVs TO BE CREATED IN SETUP STEPS> Metadata (file name/split/label) CSV files.
  external       Folder to clone/store external resources such as codes and weights.
  logs           <CREATED RUNTIME> Folder to store logs.
  results        <CREATED RUNTIME> `scores.csv` will accumulate resulting scores.
  work           <TO BE CREATED IN SETUP> Folder to serve .wav samples.
  work/16k         for 16,000 Hz samples.
  work/22k         for 22,000 Hz samples -- not 22,050 Hz, For COALA.
  work/32k         for 32,000 Hz samples.
  work/44k         for 44,100 Hz samples.
  work/48k         for 48,000 Hz samples.
```

### 3-1. Example

The followings is a example of evaluating [BYOL-A](https://github.com/nttcslab/byol-a) with [GTZAN](https://ieeexplore.ieee.org/document/1021072).
(See [Evaluation-examples.md](Evaluation-examples.md) for example command lines.)

```
$ python 2pass_lineareval.py config/byola.yaml gtzan batch_size=64
>>> python lineareval.py config/byola.yaml gtzan --options=batch_size=64 --lr=None --hidden=() --standard_scaler=True --mixup=False --early_stop_epochs=None --seed=42 --step=2pass_1_precompute_only
   :

Train:443, valid:197, test:290, multi label:False
 using network pretrained weight: AudioNTT2020-BYOLA-64x96d2048.pth
<All keys matched successfully>
Logging to logs/gtzan_ar_byola.AR_BYOLA_6bd7e19e/log.txt
['features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked', 'features.4.weight', 'features.4.bias', 'features.5.weight', 'features
.5.bias', 'features.5.running_mean', 'features.5.running_var', 'features.5.num_batches_tracked', 'features.8.weight', 'features.8.bias', 'features.9.weight', 'features.9.bias', 'features.9.running_mean', 'features.9.running_var', 'features.9.num_batches_tracked', 'fc.0.weight', 'fc.0.bias', 'fc.3.weight', 'fc.3.bias']                                                                                                                                                              
using spectrogram norimalization stats: [-3.7112076  3.5103734]
  (module): AR_BYOLA(
    (to_feature): ToLogMelSpec(
      (to_spec): MelSpectrogram(
        Mel filter banks size = (64, 513), trainable_mel=False
        (stft): STFT(n_fft=1024, Fourier Kernel size=(513, 1, 1024), iSTFT=False, trainable=False)
  :

Getting gtzan_ar_byola.AR_BYOLA_6bd7e19e train embeddings...
100%|██████████| 7/7 [00:03<00:00,  2.28it/s]
Getting gtzan_ar_byola.AR_BYOLA_6bd7e19e valid embeddings...
100%|██████████| 4/4 [00:01<00:00,  2.30it/s]
Getting gtzan_ar_byola.AR_BYOLA_6bd7e19e test embeddings... 
100%|██████████| 5/5 [00:02<00:00,  2.23it/s]
>>> python lineareval.py config/byola.yaml gtzan --options=batch_size=64 --lr=None --hidden=() --standard_scaler=True --mixup=False --early_stop_epochs=None --seed=42 --step=2pass_2_train_test
  :

Train:443, valid:197, test:290, multi label:False
 using cached embeddings: embs-gtzan_ar_byola.AR_BYOLA_6bd7e19e-train-1
 using cached embeddings: embs-gtzan_ar_byola.AR_BYOLA_6bd7e19e-valid-1
 using cached embeddings: embs-gtzan_ar_byola.AR_BYOLA_6bd7e19e-test-1
🚀 Started Linear evaluation:
 stats|train: mean=-0.0000, std=0.9079
 stats|valid: mean=-0.0333, std=1.0472
Training model: MLP(
  (mlp): Sequential(
    (0): Linear(in_features=2048, out_features=10, bias=True)
  )
)
Details - metric: acc, loss: <function loss_nll_with_logits at 0x7f7a1a2a0160>, optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0003
    weight_decay: 1e-08
), n_class: 10
epoch 0001/200: lr: 0.0003000: loss=33.254899 val_acc=0.2436548 val_loss=40.7875748
epoch 0002/200: lr: 0.0003000: loss=25.966087 val_acc=0.3959391 val_loss=35.5625954
epoch 0003/200: lr: 0.0003000: loss=21.259017 val_acc=0.4517766 val_loss=32.1851768
  :
epoch 0103/200: lr: 0.0003000: loss=0.646740 val_acc=0.6751269 val_loss=21.1744614
epoch 0104/200: lr: 0.0003000: loss=0.635991 val_acc=0.6751269 val_loss=21.1834354
Training complete in 0m 1s
Best val_acc@84 = 0.6852791878172588
Best val_loss@84 = 20.660442352294922
 stats|test: mean=-0.0388, std=0.9933
Linear evaluation: gtzan_ar_byola.AR_BYOLA_39f1b473 gtzan -> 0.75862
```

`results/scores.csv` example:

```
BYOLA,gtzan,0.7586206896551724,39f1b473,"Linear evaluation: gtzan_ar_byola.AR_BYOLA_39f1b473 gtzan -> 0.75862
{'audio_repr': 'ar_byola.AR_BYOLA', 'weight_file': 'external/byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', 'feature_d': 2048, 'sample_rate': 16000, 'n_fft': 1024, 'window_size': 1024, 'hop_size': 160, 'n_mels': 64, 'f_min': 60, 'f_max': 7800, 'temporal_pooling_type': 'mean_max', 'batch_size': 64, 'lr_lineareval': 0.0003, 'lr_finetune_frozen': 0.001, 'lr_finetune_finetune': 0.001, 'report_per_epochs': 20, 'early_stop_epochs': 20, 'task_metadata': 'evar/metadata/gtzan.csv', 'task_data': 'work/16k/gtzan', 'unit_samples': 480000, 'id': 'gtzan_ar_byola.AR_BYOLA_6bd7e19e', 'runtime_cfg': {'lr': 0.0003, 'seed': 44, 'hidden': [], 'standard_scaler': True, 'mixup': False, 'epochs': 200, 'early_stop_epochs': 20, 'id': 'fd0d06e8'}}
logs/gtzan_ar_byola.AR_BYOLA_6bd7e19e/gtzan-ar-byola.BYOLA-LE_39f1b473_0.75862.csv"
```

## 4. Fine-tuning

The fine-tuning command line is analogous to that of the linear evaluation; we utilize the script `finetune.py` as demonstrated in the following example:

    $ python finetune.py config/byola.yaml as20k --lr=1.0 --freq_mask 30 --time_mask 100 --mixup 0.3 --rrc True

- This example employs the BYOL-A model and fine-tunes it on the AudioSet20K with specified augmentation settings. We typically calibrate these settings to a specific task; overriding the settings in the command line streamlines the workflow.
  - The fundamental settings are based on the file config/byola.yaml, which can be overridden by the command line parameters.
  - The `--freq_mask 30 --time_mask 100` parameters set the SpecAugment settings.
  - The `--mixup 0.3` parameter specifies the mixing ratio for the mixup.
  - The `--rrc True` parameter enables the random resize crop (RRC) augmentation.
- The script reports the `accuracy` for multi-class single-label classification tasks or `mAP` for multi-class multi-label tasks, consistent with the linear evaluation.
- The learning rate scheduling employs the cosine annealing with a warm-up phase.

The following parameters are configurable within the .yaml file:
  - `warmup_epochs`: The number of epochs allocated for warm-up (e.g., 5).
  - `mixup`: The alpha value for mixup (e.g., 0.5)
  - `ft_bs`: The batch size employed during fine-tuning (e.g., 256).
  - `ft_lr`: The learning rate (scheduled via cosine annealing) for fine-tuning (e.g., 0.001)
  - `ft_early_stop_epochs`: The number of early stopping epochs, set to -1 to disable early stopping
  - `ft_epochs`: The number of epochs allocated for fine-tuning (e.g., 200)
  - `ft_freq_mask`: The setting for SpecAugment frequency mask (e.g., 30)
  - `ft_time_mask`: The setting for SpecAugment time mask (e.g., 100)
  - `ft_rrc`: Set to True to enable RRC.

### 4-1. Fine-tuning example

The followings is a example of evaluating [BYOL-A](https://github.com/nttcslab/byol-a) on AudioSet20K.

```sh
/lab/eval$ python finetune.py config/byola.yaml as20k --lr=1.0 --freq_mask 30 --time_mask 100 --mixup 0.3 --rrc True
+task_metadata=evar/metadata/as20k.csv,+task_data=work/16k/as,+unit_samples=160000
Logging to logs/as20k_ar_byola.AR_BYOLA_bd42a61e/log.txt
  :
🚀 Start fine-tuning  with logging in logs/as20k_ar_byola.AR_BYOLA_bd42a61e
  :
 ** Fine-tuning using Evaluation set result as test result **
 using mixup with alpha=0.3
 using SpecAugmentation with 30, 100.
 using RandomResizeCrop(virtual_crop_size=(1.0, 1.5), time_scale=(0.6, 1.5), freq_scale=(0.6, 1.5))
Epoch [0] iter: 0/86, elapsed: 4.085s, lr: 0.00000000 loss: 0.71351832
Epoch [0] iter: 10/86, elapsed: 4.724s, lr: 0.02325581 loss: 0.71286535
Epoch [0] iter: 20/86, elapsed: 4.377s, lr: 0.04651163 loss: 0.70928347
Epoch [0] iter: 30/86, elapsed: 4.481s, lr: 0.06976744 loss: 0.70343441
Epoch [0] iter: 40/86, elapsed: 4.372s, lr: 0.09302326 loss: 0.70040292
Epoch [0] iter: 50/86, elapsed: 4.412s, lr: 0.11627907 loss: 0.69242024
Epoch [0] iter: 60/86, elapsed: 4.175s, lr: 0.13953488 loss: 0.68464863
Epoch [0] iter: 70/86, elapsed: 4.103s, lr: 0.16279070 loss: 0.67849201
Epoch [0] iter: 80/86, elapsed: 3.967s, lr: 0.18604651 loss: 0.66996628
validating
Saved weight as logs/as20k_ar_byola.AR_BYOLA_bd42a61e/weights_ep0it85-0.00786_loss0.6650.pth
as20k_ar_byola.AR_BYOLA_bd42a61e-lr1.0mu3fm30tm100tx5R | epoch/iter 0/85: val mAP: 0.00786, loss: 0.66500, best: 0.00786@0
Epoch [1] iter: 0/86, elapsed: 37.298s, lr: 0.20000000 loss: 0.66475827
Epoch [1] iter: 10/86, elapsed: 5.657s, lr: 0.22325581 loss: 0.65429634
  :
Epoch [199] iter: 70/86, elapsed: 4.784s, lr: 0.00000224 loss: 0.02135683
Epoch [199] iter: 80/86, elapsed: 4.399s, lr: 0.00000040 loss: 0.02403579
validating
as20k_ar_byola.AR_BYOLA_bd42a61e-lr1.0mu3fm30tm100tx5R | epoch/iter 199/85: val mAP: 0.22109, loss: 0.02174, best: 0.22579@159
Best mAP: 0.22579
Finetuning as20k_ar_byola.AR_BYOLA_bd42a61e-lr1.0mu3fm30tm100tx5R on as20k -> mean score: 0.22579, best weight: logs/as20k_ar_byola.AR_BYOLA_bd42a61e/weights_ep159it85-0.22579_loss0.0214.pth, score file: logs/as20k_ar_byola.AR_BYOLA_bd42a61e/as20k_ar-byola.BYOLA-FT_bd42a61e_0.22579.csv, config: {'audio_repr': 'ar_byola.AR_BYOLA', 'weight_file': 'external/byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', 'feature_d': 2048, 'sample_rate': 16000, 'n_fft': 1024, 'window_size': 1024, 'hop_size': 160, 'n_mels': 64, 'f_min': 60, 'f_max': 7800, 'temporal_pooling_type': 'mean_max', 'batch_size': 256, 'lr_lineareval': 0.0003, 'report_per_epochs': 20, 'early_stop_epochs': 20, 'warmup_epochs': 5, 'mixup': 0.3, 'ft_bs': 256, 'ft_lr': 0.001, 'ft_early_stop_epochs': -1, 'ft_epochs': 200, 'ft_freq_mask': 30, 'ft_time_mask': 100, 'ft_rrc': True, 'task_metadata': 'evar/metadata/as20k.csv', 'task_data': 'work/16k/as', 'unit_samples': 160000, 'id': 'as20k_ar_byola.AR_BYOLA_bd42a61e', 'training_mask': 0.5, 'optim': 'sgd', 'unit_sec': None, 'runtime_cfg': {'lr': 1.0, 'seed': 42, 'hidden': [], 'mixup': 0.3, 'bs': 256, 'freq_mask': 30, 'time_mask': 100, 'rrc': True, 'epochs': 200, 'early_stop_epochs': -1, 'n_class': 527, 'id': '1f5f3070'}}
```

The fine-tuning results will be stored in `results/ft-scores.csv`.

## 5. Zero-shot

You can evaluate a zero-shot (ZS) classification using an evaluator script, `zeroshot.py`.

**Prepare data for ZS**

ZS uses the original, intact task data to ensure the best performance. You need to prepare data specifically for ZS. Please see the [Zero-shot evaluation data](Preparing-datasets.md#zero-shot-evaluation-data).

Be sure to download the AudioSet class label definition if you evaluate models on it.

    wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

**NOTE about Captions:**

While ZS requires converting a label into caption text, we implemented it in the `class_to_caption` function in the `zeroshot.py`.
You can edit the conversion rule in the function for your purposes.


### 5-1. ZS example

The ESC-50 example follows:

```sh
$ python zeroshot.py config/wavcaps.yaml esc50

+task_metadata=evar/metadata/esc50.csv,+task_data=work/original/ESC-50-master,+unit_samples=160000
Logging to logs/esc50_ar_wavcaps.AR_WavCaps_be6742a7/log.txt
{'audio_repr': 'ar_wavcaps.AR_WavCaps', 'weight_file': 'external/WavCaps/HTSAT-BERT-PT.pt', 'feature_d': 768, 'sample_rate': 32000, 'n_fft': 1024, 'window_size': 1024, 'hop_size': 320, 'n_mels': 64, 'f_min': 50, 'f_max': 14000, 'window': 'hanning', 'training_mask': 0.0, 'flat_f
eatures': False, 'batch_size': 128, 'lr_lineareval': 0.0003, 'report_per_epochs': 50, 'early_stop_epochs': 20, 'warmup_epochs': 5, 'mixup': 0.5, 'ft_bs': 128, 'ft_lr': 2.0, 'ft_early_stop_epochs': -1, 'ft_epochs': 200, 'ft_freq_mask': 8, 'ft_time_mask': 64, 'ft_noise': 0.0, 'ft
_rrc': True, 'name': '', 'task_metadata': 'evar/metadata/esc50.csv', 'task_data': 'work/32k/esc50', 'unit_samples': 160000, 'id': 'esc50_ar_wavcaps.AR_WavCaps_d7371b11', 'task_name': 'esc50', 'return_filename': False, 'runtime_cfg': {'id': '468067f3'}}
Train:1600, valid:0, test:400, multi label:False  

Captions: ['airplane can be heard', 'breathing can be heard', 'brushing teeth can be heard'] ...
Getting esc50_ar_wavcaps.AR_WavCaps_d7371b11 test embeddings...      
100%|████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.07s/it]
Train:1600, valid:0, test:400, multi label:False   
Getting esc50_ar_wavcaps.AR_WavCaps_d7371b11 test embeddings...
  :
100%|████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.51it/s]
esc50 result: 0.9485
Zero-shot evaluation: esc50_ar_wavcaps.AR_WavCaps_221affa2 zs_esc50 -> 0.94850
{'audio_repr': 'ar_wavcaps.AR_WavCaps', 'weight_file': 'external/WavCaps/HTSAT-BERT-PT.pt', 'feature_d': 768, 'sample_rate': 32000, 'n_fft': 1024, 'window_size': 1024, 'hop_size': 320, 'n_mels': 64, 'f_min': 50, 'f_max': 14000, 'window': 'hanning', 'batch_size': 128, 'lr_lineareval': 0.0003, 'report_per_epochs': 50, 'early_stop_epochs': 20, 'warmup_epochs': 5, 'mixup': 0.5, 'ft_bs': 128, 'ft_lr': 2.0, 'ft_early_stop_epochs': -1, 'ft_epochs': 200, 'ft_freq_mask': 8, 'ft_time_mask': 64, 'ft_noise': 0.0, 'ft_rrc': True, 'name': '', 'task_metadata': 'evar/metadata/esc50.csv', 'task_data': 'work/original/ESC-50-master', 'unit_samples': 160000, 'id': 'esc50_ar_wavcaps.AR_WavCaps_be6742a7', 'task_name': 'esc50', 'return_filename': False, 'mean': None, 'std': None, 'runtime_cfg': {'id': '468067f3'}}
 -> results/scores.csv
```

## 6. Autio to text or text to audio retrieval (ATR)

ATR is available with the `retr_a2t_t2a.py`.

Refer to the [data setup instructions](Preparing-datasets.md#audio-captioning-audio-to-text-retrieval-datasets).

**NOTE: Our current implementation is evaluation only (for testing the CLAP models).**

Tasks are:

- `clotho`: ATR using the [Clotho](https://github.com/audio-captioning/clotho-dataset) dataset
- `audiocaps`: ATR using the [AudioCaps](https://audiocaps.github.io/) dataset
- `ja_audiocaps`: ATR using the Japanese captions of the [ML-AudioCaps](https://github.com/sarulab-speech/ml-audiocaps) dataset

### 6-1. ATR example

WavCaps examples.

    python retr_a2t_t2a.py config/wavcaps.yaml clotho
    python retr_a2t_t2a.py config/wavcaps.yaml audiocaps
    python retr_a2t_t2a.py config/wavcaps.yaml ja_audiocaps

The following is the WavCaps example evaluated on AudioCaps. We confirm the results close to the paper.

```sh
$ python retr_a2t_t2a.py config/wavcaps.yaml audiocaps
+task_metadata=evar/metadata/audiocaps.csv,+task_data=work/original/audiocaps,+unit_samples=320000

Logging to logs/WavCaps-HTSAT-BERT-PT_audiocaps_0afe26da/log.txt
{'audio_repr': 'ar_wavcaps.AR_WavCaps', 'weight_file': 'external/WavCaps/HTSAT-BERT-PT.pt', 'feature_d': 768, 'sample_rate': 32000, 'n_fft': 1024, 'window_size': 1024, 'hop_size': 320, 'n_mels': 64, 'f_min': 50, 'f_max': 14000, 'window': 'hanning', 'batch_size': 128, 'lr_lineareval': 0.0003, 'report_per_epochs': 50, 'early_stop_epochs': 20, 'warmup_epochs': 5, 'mixup': 0.5, 'ft_bs': 128, 'ft_lr': 2.0, 'ft_early_stop_epochs': -1, 'ft_epochs': 200, 'ft_freq_mask': 8, 'ft_time_mask': 64, 'ft_noise': 0.0, 'ft_rrc': True, 'name': '', 'task_metadata': 'evar/metadata/audiocaps.csv', 'task_data': 'work/original/audiocaps', 'unit_samples': 320000, 'id': 'WavCaps-HTSAT-BERT-PT_audiocaps_0afe26da', 'task_name': 'audiocaps', 'return_filename': False, 'mean': None, 'std': None, 'runtime_cfg': {'id': '468067f3'}}
AR_WavCaps(
  (backbone): ASE(
    (audio_encoder): AudioEncoder(
      (audio_enc): HTSAT_Swin_Transformer(
        (audio_feats_extractor): AudioFeature(
  :
)
Getting WavCaps-HTSAT-BERT-PT_audiocaps_0afe26da embeddings for 957 samples from test split ...
100%|█████████████████████████████| 957/957 [00:16<00:00, 58.52it/s]
Embedding dimensions = audio:torch.Size([4785, 1024]), caption:torch.Size([4785, 1024])
test: Caption to audio: r1: 50.99, r5: 82.24, r10: 88.82, r50: 98.75, medr: 1.00, meanr: 5.48, mAP10: 36.491
test: Audio to caption: r1: 37.43, r5: 72.12, r10: 84.74, r50: 97.66, medr: 2.00, meanr: 7.32, mAP10: 52.044
```

## 7. Other information

### 7-1. Supported datasets

The followings are supported datasets with a short name and subdomain:

1. AudioSet20K (as20k, SER)
2. AudioSet (as, SER) * experimental
3. ESC-50 (esc50, SER)
4. US8K (us8k, SER)
5. FSD50K (fsd50k, SER)
6. SPCV1/V2 (spcv1 or spcv2, NOSS)
7. VoxForge (voxforge, NOSS)
8. VoxCeleb1 (vc1, NOSS)
9. CREMA-D (cremad, NOSS)
10. GTZAN (gtzan, Music)
11. NSynth instrument family (nsynth, Music)
12. Pitch Audio Dataset (Surge synthesizer) (surge, Music)
13. (ML-)AudioCaps (ATR)
14. Clotho (ATR)

### 7-2. Supported pre-trained models

The followings are supported:

- WavCaps
- LAION CLAP
- MS CLAP  (caution: very slow to load audio files.)
- ATST(-Clip), ATST-Frame
- BEATs
- CED (using a pre-trained weight on the Huggingface)
- HTS-AT
- VGGish
- PANNs' CNN14
- ESResNe(X)t-fbsp
- OpenL3
- AST
- Wav2Vec2 (using a pre-trained weight on the Huggingface)
- Data2vec (using a pre-trained weight on the Huggingface)
- HuBERT (using a pre-trained weight on the Huggingface)
- WavLM (using a pre-trained weight on the Huggingface)
- TRILL
- COALA
- BYOL-A

## License

See [LICENSE](LICENSE) for the detail.

## Acknowledgements / References

- VGGish: [S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal, D. Platt, R. A. Saurous, B. Seybold, M. Slaney, R. Weiss, and K. Wilson, “CNN architectures for largescale audio classification,” in ICASSP, 2017, pp. 131–135](https://arxiv.org/abs/1609.09430)
  - https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch
- PANNs: [Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “PANNs: Large-scale pretrained audio neural networks for audiopattern recognition,” TASLP, vol. 28, pp. 2880–2894, 2020.](https://arxiv.org/abs/1912.10211)
  - https://github.com/qiuqiangkong/audioset_tagging_cnn
  - https://github.com/qiuqiangkong/panns_transfer_to_gtzan
- ESResNe(X)t-fbsp: [A. Guzhov, F. Raue, J. Hees, and A. Dengel, “ESResNe(X)t-fbsp: Learning robust time-frequency transformation of audio,” in IJCNN, Jul 2021.](https://arxiv.org/abs/2104.11587)
  - https://github.com/AndreyGuzhov/ESResNeXt-fbsp
- OpenL3: [J. Cramer, H.-H. Wu, J. Salamon, and J. P. Bello, “Look, listen and learn more: Design choices for deep audio embeddings,” in ICASSP, Brighton, UK, May 2019, pp. 3852––3 856.](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf)
  - https://github.com/torchopenl3/torchopenl3
- AST: [Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio Spectrogram Transformer,” Interspeech 2021, Aug 2021.](https://arxiv.org/abs/2104.01778)
  - https://github.com/YuanGongND/ast
- Wav2Vec2: [A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in NeurIPS, 2020.](https://arxiv.org/abs/2006.11477)
  - https://github.com/huggingface/transformers
  - https://huggingface.co/docs/transformers/model_doc/wav2vec2
- Data2vec: [A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, “data2vec: A general framework for self-supervised learning in speech, vision and language,” in ICML, 2022, pp. 1298–1312.](https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/)
  - https://huggingface.co/docs/transformers/model_doc/data2vec
- HuBERT: [Hsu, Wei-Ning, et al. “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, 2021, pp. 3451–60.](https://ai.facebook.com/blog/hubert-self-supervised-representation-learning-for-speech-recognition-generation-and-compression/)
  - https://huggingface.co/docs/transformers/model_doc/hubert
- WavLM: [Chen, Sanyuan, et al. “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, Oct. 2022, pp. 1505–18.](https://arxiv.org/abs/2110.13900)
  - https://huggingface.co/docs/transformers/model_doc/wavlm
- TRILL: [J. Shor, A. Jansen, R. Maor, O. Lang, O. Tuval, F. d. C. Quitry, M. Tagliasacchi, I. Shavitt, D. Emanuel, and Y. Haviv, “Towards learning a universal non-semantic representation of speech,” in Interspeech, Oct 2020.](https://arxiv.org/abs/2002.12764)
  - https://aihub.cloud.google.com/u/0/p/products%2F41239b97-c960-479a-be50-ae7a23ae1561
- COALA: [X. Favory, K. Drossos, T. Virtanen, and X. Serra, “Coala: Co-aligned autoencoders for learning semantically enriched audio representations,” in ICML, Jul 2020.](https://arxiv.org/abs/2006.08386)
  - https://github.com/xavierfav/coala
- BYOL-A (IJCNN2021): [Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation," IJCNN 2021](https://arxiv.org/abs/2103.06695)
  - https://github.com/nttcslab/byol-a
- BYOL-A (TASLP 2023): [D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “BYOL for Audio: Exploring pre-trained general-purpose audio representations,” IEEE/ACM Trans. Audio, Speech, Language Process., vol. 31, pp. 137–151, 2023](http://dx.doi.org/10.1109/TASLP.2022.3221007)
  - https://github.com/nttcslab/byol-a/tree/master/v2

