#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.

repeat=3
# compute_time=( 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 )
compute_time=( 0.4 0.8 1.2 1.6 2.0 2.4 )
batch_size=( 256 512 1024 2048 )
node="172.31.90.206"

for((i=2;i<$repeat;i++))
do
    for t in ${compute_time[*]}
    do
        for b in ${batch_size[*]}
        do
            data_dir=data/run$i/$t/$b
            mkdir -p $data_dir
            vmtouch -e /$node/
            python3 main.py -b $b --sim-compute-time $t --epochs 1 --mini-batches 100
            mv *.npy $data_dir/
            mv /share/train_cache_usage.npy $data_dir/
        done
    done
done
