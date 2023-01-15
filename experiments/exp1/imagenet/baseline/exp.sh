#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.
# Therefore, we set the compute time as 0.4s - 4.0s with batch size 512.
compute_time=( 0.4 1.2 2.0 2.8 3.6 4.0 )
pod="ilsvrc-baseline27"
node="129.59.234.237"

for((i=1;i<6;i++))
do
    for t in ${compute_time[*]}
    do
        mkdir -p data/run$i/$t
        kubectl exec $pod -- vmtouch -e /nfs_storage/
        sudo ssh -i ~/.ssh/id_rsa ubuntu@$node vmtouch -e /nfs_storage/
        kubectl exec $pod -- python3 main.py -b 512 --sim-compute-time $t --epochs 1 --mini-batches 50
        kubectl cp $pod:/app/exp.npy data/run$i/$t/load_time.npy
    done
done
