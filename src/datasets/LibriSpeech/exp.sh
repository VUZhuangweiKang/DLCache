#!/bin/bash

# According to the benchmark results in https://lambdalabs.com/gpu-benchmarks,
# the training time of ResNet-50 range from 135 to 1197 images/s.

i=0
w=4
test=1
compute_time=( 1.0 1.4 1.8 2.2 2.6)
batch_size=( 8, 16, 32 )
node="172.31.92.64"

total_test=$((${#compute_time[@]} * ${#batch_size[@]}))
for t in ${compute_time[*]}
do
    for b in ${batch_size[*]}
    do
        data_dir=data/run$i/$t/$b
        mkdir -p $data_dir
        vmtouch -e /$node/
        echo "Exp[$test/$total_test]: worker=$w, batch_size=$b, compute_time=$t"
        echo "$w,$b,1,100,10,$t" > /app/dlcache_exp.txt
        python train.py +configs=librispeech
        mv *.npy $data_dir/
        mv /share/train_cache_usage.npy $data_dir/
        python3 report.py $i $t $b
        ((test=test+1))
    done
done
