#!/bin/bash

if [ $# -lt 1 ]; then
  n_iter=3
else
  n_iter=$1
fi

if [ $# -lt 2 ]; then
  lr_prm=0.003
else
  lr_prm=$2
fi
bs=256
spl=1
head=tfm
extra=--freeze_body

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
    cmdline="CUDA_VISIBLE_DEVICES=0 python solve.py ../../config/opera.yaml --epochs 150 --weightspath ../../external/OPERA/encoder-operaCT.ckpt --bs $bs --lr $lr_prm --head $head $extra --split_iter $spl"
    echo $cmdline
    eval $cmdline
done
