#!/bin/bash

weight=$1
split=$2
n_iter=$3
seed=$4
lr_prm=$5
bs=1024
gpu=0
hidden='\(128,\)'
reweight=True

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
  seed=$((seed + 1))
  cmdline="CUDA_VISIBLE_DEVICES=$gpu python -m app.circor.solve_circor config/m2d.yaml circor$split weight_file=$weight,encoder_only=True --lr=$lr_prm --freq_mask 0 --time_mask 0 --training_mask 0.0 --mixup 0.0 --rrc False --epochs 50 --warmup_epochs 0 --seed $seed --batch_size $bs --hidden $hidden --reweight $reweight"
  echo $cmdline
  eval $cmdline
done
