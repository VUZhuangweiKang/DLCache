#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.
# Therefore, we set the compute time as 0.4s - 4.0s with batch size 512.
compute_time=( 0.4 1.2 2.0 2.8 3.6 4.0 )
node="129.59.234.237"
pod="ilsvrc27"

for((i=4;i<6;i++))
do
    for t in ${compute_time[*]}
    do
        mkdir -p data/run$i/$t
        kubectl exec $pod --container client -- vmtouch -e /$node/
        sudo ssh -i ~/.ssh/id_rsa ubuntu@$node vmtouch -e /nfs_storage
        kubectl exec $pod -- python3 main.py -b 512 --sim-compute-time $t --epochs 1 --mini-batches 50
        kubectl cp $pod:/app/exp.npy data/run$i/$t/load_time.npy --container job
        for((k=0;k<4;k++))
        do
            kubectl cp $pod:/app/cache_hits_$k.npy data/run$i/$t/cache_hits_$k.npy --container job
        done
        kubectl cp $pod:/share/train_cache_usage.npy data/run$i/$t/train_cache_usage.npy --container job
        kubectl exec $pod pkill python3
    done
done
