#!/bin/bash

weight=$1
split=$2
n_iter=$3
seed=$4
lr_prm=$5
bs=128
epochs=100
gpu=0

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
  seed=$((seed + 1))
  cmdline="CUDA_VISIBLE_DEVICES=$gpu python -m app.bmdhs.solve_bmdhs config/m2d.yaml bmdhs$split weight_file=$weight,encoder_only=True --lr=$lr_prm --freq_mask 0 --time_mask 0 --training_mask 0.0 --mixup 0.0 --rrc False --epochs $epochs --warmup_epochs 0 --seed $seed --batch_size $bs"
  echo $cmdline
  eval $cmdline
done
