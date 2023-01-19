#!/bin/sh

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.

repeat=1
compute_time=( 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 )
batch_size=( 256 512 1024 2048 )
pod="ilsvrc"
node="172.31.92.64"

for((i=0;i<$repeat;i++))
do
    for t in ${compute_time[*]}
    do
        for b in ${batch_size[*]}
        do
            data_dir=data/run$i/$t/$b
            mkdir -p $data_dir
            kubectl exec $pod pkill python3 --container client
            kubectl exec $pod pkill python3 --container job
            sleep 2
            kubectl exec $pod --container client -- vmtouch -e /$node/
            kubectl exec $pod --container client -- python3 client.py
            sleep 5
            kubectl exec $pod -- python3 main.py -b $b --sim-compute-time $t --epochs 1 --mini-batches 100
            kubectl cp $pod:/app/load_time.npy $data_dir/load_time.npy --container job
            for((k=0;k<4;k++))
            do
                kubectl cp $pod:/app/cache_hits_$k.npy $data_dir/cache_hits_$k.npy --container job
            done
            kubectl cp $pod:/app/train_cache_usage.npy $data_dir/train_cache_usage.npy --container client
            sleep 3
        done
    done
done
