#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.

i=0
w=4
compute_time=( 0.5 1.0 1.5 2.0 2.5 3.0 )
batch_size=( 256 512 1024 2048 )

for t in ${compute_time[*]}
do
    for b in ${batch_size[*]}
    do
        data_dir=data/run$i/$t/$b
        mkdir -p $data_dir
        vmtouch -e /nfs_storage/
        python3 main.py -b $b --sim-compute-time $t --epochs 1 --mini-batches 100 -j $w
        mv /tmp/*.npy $data_dir/
    done
done
