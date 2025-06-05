## MARBLE Benchmark Integration

This repository provides the code to integrate EVAR as a plugin model into the MARBLE benchmark, enabling MARBLE to evaluate pre-trained audio representation models supported by EVAR. Based on this MARBLE extension, it also provides scripts and instructions to reproduce the results from the M2D2 paper.

## How to Integrate with the MARBLE Benchmark

Follow the steps below to integrate EVAR into your MARBLE directory cloned from GitHub.

*NOTE*: In addition to the integration steps, set the environment variable `EVAR` to point to the local EVAR folder so that MARBLE can reference it.

```sh
export EVAR=/lab/eval-audio-repr

git clone https://github.com/a43992899/MARBLE-Benchmark.git
cd MARBLE-Benchmark
git checkout d9300e335eefdad8d6b825418e8c44b22d0919c7

patch -p1 < $EVAR/plugin/MARBLE/evar_marble_diff.patch
cp -r $EVAR/plugin/MARBLE/benchmark/models/evar benchmark/models
cp -r $EVAR/plugin/MARBLE/configs/evar configs
cp $EVAR/plugin/MARBLE/evar_marble.sh .
```

For the task datasets for MARBLE, follow the instructions provided by the MARBLE.

## Evaluating models on MARBLE

Once you prepare EVAR on MARBLE, you can use the script `evar_marble.sh` to evaluate models. The following is an example of M2D.

```sh
bash evar_marble.sh m2d /your/m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 7 5 feat-m2d-mr7 3840
```

The results will be stored in CSV files, such as `score_EMO.csv`.

### More example command lines

```sh
EVAR=/your/evar bash evar_marble.sh beats_plus /your/BEATs_iter3_plus_AS2M.pt 7 5
EVAR=/your/evar bash evar_marble.sh atst_frame /your/atstframe_base.ckpt 7 5
EVAR=/your/evar bash evar_marble.sh msclap 2023 7 5
EVAR=/your/evar bash evar_marble.sh m2d /your/m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 7 5 feat-m2d-mr7
EVAR=/your/evar bash evar_marble.sh m2d /your/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep67it3124-0.47941.pth 7 5 feat-m2d-mr7-as
EVAR=/your/evar bash evar_marble.sh m2d /your/clap/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth 7 5 feat-m2d-clap
EVAR=/your/evar bash evar_marble.sh m2d /your/msm_mae_vit_base-80x608p16x16-220924-mr75/checkpoint-300.pth 7 5 feat-msm-mae
```
