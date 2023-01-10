#!/bin/bash

# ResNet-50, training time of GPU K80, M40, P100, V100
model="ResNet50"
compute_time=( 1.64 0.71 0.45 0.23 0.13 )
mkdir $model
for t in ${compute_time[*]}
do
    vmtouch /129.59.234.237/
    python3 main.py -b 64 --sim-compute-time $t --epochs 1 --mini-batches 50
    mv exp.npy $model/$t.npy
done

# AlexNet, tranining time of GPU K80, M40, P100, V100
model="AlexNet"
mkdir $model
compute_time=( 0.31 0.13 0.077 0.046 0.035 )
for t in ${compute_time[*]}
do
    vmtouch /129.59.234.237/
    python3 main.py -b 128 --sim-compute-time $t --epochs 1 --mini-batches 50
    mv exp.npy $model/$t.npy
done