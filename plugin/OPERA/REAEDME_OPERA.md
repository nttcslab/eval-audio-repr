# Towards Pre-training an Effective Respiratory Audio Foundation Model

This repository provides:
- Code to integrate EVAR as a plugin model into the [OPERA](https://github.com/evelyn0414/OPERA) benchmark for respiratory sounds.
  This extension enables OPERA to evaluate 20+ additional audio models supported by EVAR.
- Scripts and instructions to reproduce the results of our Interspeech 2025 paper.
- Pre-trained M2D+Resp weights that achieved the best results in our Interspeech 2025 paper. 👉 [release](https://github.com/nttcslab/eval-audio-repr/releases/tag/0.1.0)
- Example notebook for using M2D+Resp. 👉 [Example-M2D+Resp.ipynb](Example-M2D+Resp.ipynb)

## How to Integrate with the OPERA Benchmark

Follow the steps below to integrate EVAR into your OPERA directory cloned from GitHub.

*NOTE*: In addition to the integration steps, set the environment variable `EVAR` to point to the local EVAR folder so that OPERA can reference it.

```sh
export EVAR=/lab/eval-audio-repr

git clone https://github.com/evelyn0414/OPERA.git
cd OPERA
git checkout 3622310e667afb8aa40169050b4dd45de75946a2

cp $EVAR/plugin/OPERA/evar_*.sh .
patch -p1 < $EVAR/plugin/OPERA/evar_opera_diff.patch
```

The setup will enable the extraction of features using EVAR models from within OPERA and their evaluation on OPERA tasks. Scripts for each model (e.g., `evar_ast.sh`) are provided to evaluate models and can be used to reproduce the results in the paper.

For the task datasets for OPERA, follow the instructions provided by the OPERA.


## Reproducing our Interspeech 2025 Paper

The provided `evar_*.sh` are the scripts used to evaluate each model in the paper. These scripts run OPERA’s `processing` to extract features from task-specific audio samples using the target models, followed by `linear_eval` to evaluate task performance. For each task, the average and standard deviation of five evaluation runs are recorded in a CSV file, `opera-scores.csv`.

For some models (HuBERT, wav2vec 2.0, WavLM, M2D), the script performs evaluations across all layers, and the layer-wise results are recorded in `opera-scores.csv`.

For HuBERT, wav2vec 2.0, and WavLM, we used the best-performing layers HuBERT_7, wav2vec2_7, and WavLM_6, respectively, in the reported results.
The M2D script requires two arguments: the first is the path to the weight file, and the second is the feature vector dimension of the model (3840 for standard M2D).

```sh
bash ./evar_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 3840
```

### Pre-training for "Ablations using M2D"

Note: **The M2D+Resp weight is available. 👉 [release](https://github.com/nttcslab/eval-audio-repr/releases/tag/0.1.0)**

The ablation study involves pre-training with M2D.

1. Set up M2D and prepare for pre-training on AudioSet. --> Follow the instructions: [M2D:Pre-training From Scratch](https://github.com/nttcslab/m2d?tab=readme-ov-file#3-pre-training-from-scratch) and [M2D:Pre-training data](https://github.com/nttcslab/m2d/blob/master/data/README.md)
2. Prepare the respiratory dataset. --> Follow the instructions: [Collecting_Respiratory_Data.ipynb](Collecting_Respiratory_Data.ipynb)
3. Create pre-training data lists combining the respiratory data and AudioSet. --> Follow the instructions: [Preparing_Pretraining_Data.ipynb](Preparing_Pretraining_Data.ipynb)

Once the above preparations are complete, you can pre-train using the command line below. After the pre-training, use the resulting model with the OPERA evaluation script as the following example:

```sh
bash ./evar_m2d.sh m2d_vit_base-80x608p16x16p16k-250826-AS+Resp400K/checkpoint-50.pth 3840
```

#### Command lines for M2D pre-training

```sh
# Fur: Resp only -- m2d_x_vit_base-80x608p16x16p16k-241221-MdfRFM1Ddffsd50ks3blr0001bs128lo1nr.3-e600
python train_audio.py --epochs 600 --save_freq 100 --eval_after 600 --model m2d_x_vit_base --resume m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --input_size 80x608 --patch_size 16x16 --batch_size 128 --accum_iter 1 --csv_main data/files_R_F_M_1.csv --csv_bg_noise data/files_f_s_d_5_0_k.csv --noise_ratio 0.3 --seed 3 --teacher m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --blr 0.0001 --loss_off 1.0 --loss_m2d 1.0
# Fur: AS+Resp 100K -- m2d_vit_base-80x608p16x16p16k-241220-MdfASRFM1s3-e50
torchrun --nproc_per_node=4 train_audio.py --epochs 50 --save_freq 50 --eval_after 50 --model m2d_vit_base --resume m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --input_size 80x608 --patch_size 16x16 --batch_size 512 --accum_iter 1 --csv_main data/files_A_S_R_F_M_1.csv --noise_ratio 0.0 --seed 3 --blr 3e-4
# Fur: AS+Resp 200K -- m2d_vit_base-80x608p16x16p16k-241223-MdfASRFM2s3-e50
torchrun --nproc_per_node=4 train_audio.py --epochs 50 --save_freq 50 --eval_after 50 --model m2d_vit_base --resume m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --input_size 80x608 --patch_size 16x16 --batch_size 512 --accum_iter 1 --csv_main data/files_A_S_R_F_M_2.csv --noise_ratio 0.0 --seed 3 --blr 3e-4
# Fur: AS+Resp 300K -- m2d_vit_base-80x608p16x16p16k-241225-MdfASRFM3s3-e50
torchrun --nproc_per_node=4 train_audio.py --epochs 50 --save_freq 50 --eval_after 50 --model m2d_vit_base --resume m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --input_size 80x608 --patch_size 16x16 --batch_size 512 --accum_iter 1 --csv_main data/files_A_S_R_F_M_3.csv --noise_ratio 0.0 --seed 3 --blr 3e-4
# Fur: AS+Resp 400K (M2D+Resp) -- m2d_vit_base-80x608p16x16p16k-250106-MdfASRFM4s3-e50
torchrun --nproc_per_node=4 train_audio.py --epochs 50 --save_freq 50 --eval_after 50 --model m2d_vit_base --resume m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --input_size 80x608 --patch_size 16x16 --batch_size 512 --accum_iter 1 --csv_main data/files_A_S_R_F_M_4.csv --noise_ratio 0.0 --seed 3 --blr 3e-4
# Fur: AS+Resp 500K -- m2d_vit_base-80x608p16x16p16k-250104-MdfASRFM5s3-e50
torchrun --nproc_per_node=4 train_audio.py --epochs 50 --save_freq 50 --eval_after 50 --model m2d_vit_base --resume m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth --input_size 80x608 --patch_size 16x16 --batch_size 512 --accum_iter 1 --csv_main data/files_A_S_R_F_M_5.csv --noise_ratio 0.0 --seed 3 --blr 3e-4
# Scratch: AS+Resp 400K -- m2d_vit_base-80x608p16x16p16k-250103-MdfASRFM3s3-e300
torchrun --nproc_per_node=4 -m train_audio --input_size 80x608 --patch_size 16x16 --epochs 300 --batch_size 512 --accum_iter 1 --save_freq 50 --seed 3 --model m2d_vit_base --csv_main data/files_A_S_R_F_M_3.csv --noise_ratio 0. --loss_off 0.
# Scratch: LibriSpeech m2d_vit_base-80x608p80x4-230227-nr.2
```


## Referecnces

- OPERA: *[Y. Zhang, T. Xia, J. Han, Y. Wu, G. Rizos, Y. Liu, M. Mosuily, J. Chauhan, and C. Mascolo, “Towards open respiratory acoustic foundation models: Pretraining and benchmarking,” in NeurIPS, 2024.](https://neurips.cc/virtual/2024/poster/97457).* 👉  [GitHub](https://github.com/evelyn0414/OPERA).

- Niizumi et al. (Interspeech 2025): *[D. Niizumi, D. Takeuchi, M. Yasuda, B. T. Nguyen, Y. Ohishi, and N. Harada, "Towards Pre-training an Effective Respiratory Audio Foundation Model," at Interspeech, 2025](https://www.isca-archive.org/interspeech_2025/niizumi25_interspeech.html).* 👉  [GitHub](https://github.com/nttcslab/eval-audio-repr/tree/main/plugin/OPERA).

