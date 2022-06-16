# EVAR ~ Evaluation package for Audio Representations

This repository provides an evaluation package for audio representations (ARs) used in our papers. Features:

- Reproducible evaluation on various audio downstream tasks.
  - Train/valid/test set splits provided.
- Unified AR interface.
- Linear evaluation, run quick.
- Initially EVAR supports 11 tasks and nine models.

In early 2021, we had no codebase for evaluating models on various tasks under unified test settings, which motivated us to create this repository.

By the end of 2021, we had a couple of similar options ([SERAB](https://github.com/Neclow/serab/), [SUPERB](https://superbbenchmark.org/), [HEAR 2021 NeurIPS Challenge](https://neuralaudio.ai/hear2021-datasets.html), and [HARES](https://arxiv.org/abs/2111.12124)). Compared to them now, this has been created independently for our study.

This is for your research if you want to compare ARs under the same test set up with us, including:

- BYOL-A (IJCNN 2021): *[Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation," IJCNN 2021](https://arxiv.org/abs/2103.06695).* 👉 [GitHub](https://github.com/nttcslab/byol-a)
- MSM-MAE (T.B.D.): *[Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation," 2022](https://arxiv.org/abs/2204.12260).* 👉  [GitHub](https://github.com/nttcslab/msm-mae)
- Niizumi et al. (EUSIPCO 2022): *[Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "Composing General Audio Representation by Fusing Multilayer Features of a Pre-trained Model," EUSIPCO 2022](https://arxiv.org/abs/2205.08138).* 👉  [GitHub](https://github.com/nttcslab/composing-general-audio-repr)

## 1. Quick start

The following show how to prepare CREMA-D dataset and evaluate OpenL3 (music) features on CREMA-D.

0. Follow the steps in "2-1. Step 1: Install modeules, and download depending source code", in short:

    ```sh
    curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py
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

## 2. Overview

The following shows the evaluation flow with an example command line:

    $ python lineareval.py config/openl3mus.yaml cremad

- The main program (`lineareval.py`) converts CREMA-D audio samples to embedding vectors by calling the OpenL3 model.
  - OpenL3 requires 48 kHz audio, then samples in the `work/48k` folder are used.
  - The model (OpenL3 in this example) is instantiated according to the config file (config/openl3.yaml). The config file describes the detail of the model instance, such as the pre-trained weight file to load.
- The main program trains a linear model using `TorchMLPClassifier2`, an MLPClassifier implementation near compatible with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
- The main program tests the trained linear model with test samples. It reports `accuracy` for multi-class single label classification tasks or `mAP` for multi-class multi-label tasks.

The following shows the folder structure:

```
/evar           Evaluation codes.
/evar/utils     Helper utilitiy codes.
/evar/metadata  <SOME CSVs TO BE CREATED IN SETUP STEPS> Metadata (file name/split/label) CSV files.
/external       Folder to clone/store external resources such as codes and weights.
/logs           <CREATED RUNTIME> Folder to store logs.
/results        <CREATED RUNTIME> `scores.csv` will accumulate resulting scores.
/work           <TO BE CREATED IN SETUP> Folder to serve .wav samples.
/work/16k         for 16,000 Hz samples.
/work/22k         for 22,000 Hz samples -- not 22,050 Hz, For COALA.
/work/32k         for 32,000 Hz samples.
/work/44k         for 44,100 Hz samples.
/work/48k         for 48,000 Hz samples.
```

### 2-1. Example

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

## 3. Other information

### 3-1. Supported datasets

Initially supported datasets are listed below with a short name and subdomain:

- ESC-50 (esc50, SER)
- US8K (us8k, SER)
- FSD50K (fsd50k, SER)
- SPCV1/V2 (spcv1 or spcv2, NOSS)
- VoxForge (voxforge, NOSS)
- VoxCeleb1 (vc1, NOSS)
- CREMA-D (cremad, NOSS)
- GTZAN (gtzan, Music)
- NSynth instrument family (nsynth, Music)
- Pitch Audio Dataset (Surge synthesizer) (surge, Music)

### 3-2. Supported pre-trained models

Initially supported models are:

- VGGish
- PANNs' CNN14
- ESResNe(X)t-fbsp
- OpenL3
- AST
- Wav2Vec2 (with a pre-trained weight on the Huggingface)
- TRILL
- COALA
- BYOL-A

## 4. License

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
- TRILL: [J. Shor, A. Jansen, R. Maor, O. Lang, O. Tuval, F. d. C. Quitry, M. Tagliasacchi, I. Shavitt, D. Emanuel, and Y. Haviv, “Towards learning a universal non-semantic representation of speech,” in Interspeech, Oct 2020.](https://arxiv.org/abs/2002.12764)
  - https://aihub.cloud.google.com/u/0/p/products%2F41239b97-c960-479a-be50-ae7a23ae1561
- COALA: [X. Favory, K. Drossos, T. Virtanen, and X. Serra, “Coala: Co-aligned autoencoders for learning semantically enriched audio representations,” in ICML, Jul 2020.](https://arxiv.org/abs/2006.08386)
  - https://github.com/xavierfav/coala
- BYOL-A: [Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation," 2021](https://arxiv.org/abs/2103.06695)
  - https://github.com/nttcslab/byol-a


