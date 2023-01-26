#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.

i=0
w=4
compute_time=( 0.4 0.8 1.2 1.6 2.0 2.4 )
batch_size=( 32 64 128 )

test=1
total_test=$((${#compute_time[@]} * ${#batch_size[@]}))

for t in ${compute_time[*]}
do
    for b in ${batch_size[*]}
    do
        data_dir=data/run$i/$t/$b
        mkdir -p $data_dir
        vmtouch -e /nfs_storage/
        echo "Exp[$test/$total_test]: worker=$w, batch_size=$b, compute_time=$t"
        echo "$w,$b,1,100,1,$t" > /app/dlcache_exp.txt
        python train.py +configs=librispeech
        mv *.npy $data_dir/
        ((test=test+1))
    done
done

