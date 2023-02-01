#!/bin/bash

# According to the benchmark results in http://www.cs.toronto.edu/ecosystem/papers/Echo-Arxiv.pdf,
# the throughput of DeepSpeech2 range from 4-12samples/s for batch size from 8, 16, 24, 32

i=0
w=8
test=1
compute_time=( 0.4 0.8 1.2 1.6 2.0 2.4 )
batch_size=( 128 256 512 )
total_test=$((${#compute_time[@]} * ${#batch_size[@]}))

for t in ${compute_time[*]}
do
    for b in ${batch_size[*]}
    do
        data_dir=data/run$i/$t/$b
        mkdir -p $data_dir
        vmtouch -e /nfs_storage/
        echo "Exp[$test/$total_test]: worker=$w, batch_size=$b, compute_time=$t"
        echo "$w,$b,1,100,10,$t" > /app/dlcache_exp.txt
        python train.py +configs=librispeech
        mv /tmp/*.npy $data_dir/
        python3 report.py $i $t $b
        ((test=test+1))
    done
done

