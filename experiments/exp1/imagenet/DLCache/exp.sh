#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.

i=2
w=4
compute_time=( 0.4 0.8 1.2 1.6 2.0 2.4 )
batch_size=( 512 1024 2048 )
node="172.31.92.64"
autoscale_workers=true

for t in ${compute_time[*]}
do
    for b in ${batch_size[*]}
    do
        data_dir=data/run$i/$t/$b
        mkdir -p $data_dir
        vmtouch -e /$node/
        python3 main.py -b $b --sim-compute-time $t --epochs 1 --mini-batches 100 -j $w --autoscale_workers $autoscale_workers
        mv /tmp/*.npy $data_dir/
        mv /share/train_cache_usage.npy $data_dir/
    done
done
