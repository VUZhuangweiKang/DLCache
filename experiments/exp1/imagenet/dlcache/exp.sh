#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.
# Therefore, we set the compute time as 0.4s - 4.0s with batch size 512.
compute_time=( 0.4 1.2 2.0 2.8 3.6 4.0 )

for t in ${compute_time[*]}
do
    mkdir -p data/$t
    kubectl exec ilsvrc --container client -- vmtouch -e /129.59.234.237/
    sudo ssh -i ~/.ssh/id_rsa ubuntu@129.59.234.237 "vmtouch -e /nfs_storage"
    kubectl exec ilsvrc -- python3 main.py -b 512 --sim-compute-time $t --epochs 1 --mini-batches 50
    kubectl cp ilsvrc:/app/exp.npy data/$t/load_time.npy --container job
    for((i=0;i<4;i++));
    do
        kubectl cp ilsvrc:/app/cache_hits_$i.npy data/$t/cache_hits_$i.npy --container job
    done
    kubectl cp ilsvrc:/share/train_cache_usage.npy data/$t/train_cache_usage.npy --container job
done