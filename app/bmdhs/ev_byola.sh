#!/bin/bash

split=$1
n_iter=$2
seed=$3
lr_prm=$4
bs=128
epochs=30
gpu=0

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
  seed=$((seed + 1))
  cmdline="CUDA_VISIBLE_DEVICES=$gpu python -m app.bmdhs.solve_bmdhs config/byola.yaml bmdhs$split --lr=$lr_prm --freq_mask 0 --time_mask 0 --training_mask 0.0 --mixup 0.0 --rrc False --epochs $epochs --warmup_epochs 0 --seed $seed --batch_size $bs"
  echo $cmdline
  eval $cmdline
done
