#!/bin/bash

compute_time=( 0.01 0.1 0.5 1.0 2.0 3.0 )
batch_size=( 256 512 1024 2048 4096 )
num_workers=( 0 1 2 4 6 8 )
for t in ${compute_time[*]}
do
    for bs in ${batch_size[*]}
    do
        for w in ${num_workers[*]}
        do
            FILE=data/exp_${t}_${bs}_${w}.npy
            if [ ! -f "$FILE" ]; then
                python3 main.py -b $bs -j $w --sim-compute-time $t --epochs 1
                sleep 3
                mv exp.npy data/exp_${t}_${bs}_${w}.npy  
            fi
        done
    done
done