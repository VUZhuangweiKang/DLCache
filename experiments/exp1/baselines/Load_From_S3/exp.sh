#!/bin/bash

compute_time=( 0.01 0.1 0.5 1.0 2.0 3.0 )
batch_size=( 256 512 1024 2048 4096 )
num_workers=( 0 1 2 4 6 8 )
exp_id=0
for t in ${compute_time[*]}
do
    for bs in ${batch_size[*]}
    do
        for w in ${num_workers[*]}
        do
            python3 main.py -b $bs -j $w --sim-compute-time $t --epochs 1
            mv exp.npy data/exp$exp_id.npy
            ((exp_id=exp_id+1))
        done
    done
done