# ICBHI 2017 and SPRSound evaluation

We provide code to evaluate ICBHI 2017 and SPRSound with various models.

**NOTE: The code freezes the audio representation model weights.**

Prepare code and download datasets before running the evaluation.

## Prepare code

```sh
pip install torchinfo
git clone https://github.com/ilyassmoummad/scl_icbhi2017.git
cd scl_icbhi2017
git reset --hard 915c1120719a9357d662c5fe484bce7fbe845139
mv dataset.py augmentations.py utils.py losses.py args.py ..
mv data ..
mv ce.py ..
cd ..
patch -p2 < patch_scl_icbhi2017_evar.diff
rm -fr scl_icbhi2017
```

## Download ICBHI 2017

```sh
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip --no-check-certificate

unzip ICBHI_final_database.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'
mv ICBHI_final_database/* data/ICBHI
rmdir ICBHI_final_database
```

## Download SPRS

```sh
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
(cd SPRSound && git reset --hard 45b0d5d435ff320c46585762fa1090afd0ebb318)
cp -r SPRSound/train_wav SPRSound/test_wav data/SPRS/
```

## Run evaluations

The following examples run evaluations on ICBHI 2017 for the models.

```sh
bash ev_icbhi_beats.sh
bash ev_icbhi_m2d.sh ../../m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth
```

Find the shell scripts for more evaluations.

**NOTE: All the evaluations employ a transformer head except for ev_icbhi_mlp_m2d.sh, which uses MLP instead.** 

The following is the list of command lines for reproduction.

```sh
bash ev_icbhi_ast.sh 5
bash ev_icbhi_beats.sh 5
bash ev_icbhi_byola.sh 5
bash ev_icbhi_opera.sh 5
bash ev_icbhi_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 5
bash ev_icbhi_m2d.sh m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth 5

bash ev_sprs_ast.sh 5
bash ev_sprs_beats.sh 5
bash ev_sprs_byola.sh 5
bash ev_sprs_opera.sh 5
bash ev_sprs_m2d.sh m2d_vit_base-80x608p16x16-221006-mr7/checkpoint-300.pth 5
bash ev_sprs_m2d.sh m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth 5

# Ablations: M2D (16×4, MLP)
bash ev_icbhi_mlp_m2d.sh m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth 5
bash ev_sprs_mlp_m2d.sh m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth 5
```
