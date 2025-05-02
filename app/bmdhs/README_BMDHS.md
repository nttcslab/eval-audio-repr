# CirCor evaluation

We provide code to evaluate BMD-HS with various models.
In addition, the exact stratified data splits used in the paper are provided for reproducibility.

**NOTE: The code freezes the audio representation model weights.**

Prepare data and metadata files before your evaluation.

In this folder `app/bmdhs`, download the dataset and fix one file name:

```sh
git clone https://github.com/mHealthBuet/BMD-HS-Dataset
mv BMD-HS-Dataset/train/MD_085_sit_Tri6_06.wav BMD-HS-Dataset/train/MD_085_sit_Tri.wav
```

Then, the following will resample/copy data files from `BMD-HS-Dataset` to `../../work/16k/bmdhs`.

```sh
python ../../prepare_wav.py BMD-HS-Dataset/ ../../work/16k/bmdhs 16000
```

In addition, the following will create metadata files as `../../evar/metadata/bmdhs[1-3].csv`.

```sh
python make_metadata.py
```

## Run evaluations

In the **root folder of EVAR**, run the scripts `ev_*.sh`. The following is the complete set of command lines for the paper.

The results will be recorded in `results/bmdhs-scores.csv`.

```sh
bash app/bmdhs/ev_ast.sh 1 5 42 0.1
bash app/bmdhs/ev_ast.sh 2 5 42 0.1
bash app/bmdhs/ev_ast.sh 3 5 42 0.1

bash app/bmdhs/ev_beats.sh 1 5 42 0.1
bash app/bmdhs/ev_beats.sh 2 5 42 0.1
bash app/bmdhs/ev_beats.sh 3 5 42 0.1

bash app/bmdhs/ev_byola.sh 1 5 42 0.1
bash app/bmdhs/ev_byola.sh 2 5 42 0.1
bash app/bmdhs/ev_byola.sh 3 5 42 0.1

bash app/bmdhs/ev_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 1 5 42 0.1
bash app/bmdhs/ev_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 2 5 42 0.1
bash app/bmdhs/ev_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 3 5 42 0.1
```



