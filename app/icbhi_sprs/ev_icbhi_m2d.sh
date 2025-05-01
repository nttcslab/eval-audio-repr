#!/bin/bash

if [ $# -lt 2 ]; then
  n_iter=3
else
  n_iter=$2
fi

if [ $# -lt 3 ]; then
  lr_prm=0.00003
else
  lr_prm=$3
fi
bs=256
spl=1
head=tfm
extra=--freeze_body
# --freeze_embed

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
    cmdline="CUDA_VISIBLE_DEVICES=0 python solve.py ../../config/m2d.yaml --epochs 150  --bs $bs --lr $lr_prm --weightspath $1 --head $head $extra --split_iter $spl"
    echo $cmdline
    eval $cmdline
done
