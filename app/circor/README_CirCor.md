# CirCor evaluation

We provide code to evaluate CirCor with various models.
In addition, the exact stratified data splits used in the paper are provided for reproducibility.

**NOTE: The code freezes the audio representation model weights.**

Prepare code and download datasets before your evaluation.

## Prepare codebase

In this folder `app/circor`, run the following:

```sh
git clone https://github.com/Benjamin-Walker/heart-murmur-detection.git
(cd heart-murmur-detection && git checkout 60f5420918b151e06932f70a52649d9562f0be2d)
patch -p1 < patch-heart-murmur-detection.diff

wget https://raw.githubusercontent.com/nttcslab/m2d/refs/heads/master/app/circor/datalist_stratified_data1.csv
wget https://raw.githubusercontent.com/nttcslab/m2d/refs/heads/master/app/circor/datalist_stratified_data2.csv
wget https://raw.githubusercontent.com/nttcslab/m2d/refs/heads/master/app/circor/datalist_stratified_data3.csv
```

## Download and rearrange dataset

In this folder `app/circor`, download the dataset:

```sh
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```

Then, do the following to rearrange data files into stratified splits and copy them under `heart-murmur-detection/data` and `../../work/16k/circor`.

```sh
python rearrange_data.py
```

It also creates metadata files as `../../evar/metadata/circor[1-3].csv`.

## Run evaluations

In the **root folder of EVAR**, run the scripts `ev_*.sh`. The following is the complete set of command lines for the paper.

The results will be recorded in `results/circor-scores.csv`.

```sh
bash app/circor/ev_ast.sh 1 5 7 0.03
bash app/circor/ev_ast.sh 2 5 7 0.03
bash app/circor/ev_ast.sh 3 5 7 0.03

bash app/circor/ev_beats.sh 1 5 7 0.03
bash app/circor/ev_beats.sh 2 5 7 0.03
bash app/circor/ev_beats.sh 3 5 7 0.03

bash app/circor/ev_byola.sh 1 5 7 0.1
bash app/circor/ev_byola.sh 2 5 7 0.1
bash app/circor/ev_byola.sh 3 5 7 0.1

bash app/circor/ev_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 1 5 7 0.1
bash app/circor/ev_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 2 5 7 0.1
bash app/circor/ev_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 3 5 7 0.1
```
