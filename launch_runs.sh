#!/bin/bash

learning_rates=(3e-4 6e-4 1e-3)
dataset="cifar10"
gpu_offset=0

for i in {0..3}; do
    #export CUDA_VISIBLE_DEVICES=$((i + gpu_offset))
    echo "Using GPU $((i + gpu_offset))"
    echo "Learning rate: ${learning_rates[$i]}"
    screen -dmS "tr$((i + gpu_offset))" bash -c "source load.sh; python train_gpt.py --lr ${learning_rates[$i]} --dataset ${dataset}; read"
    sleep 20
done

# you can bring these down with
# screen -ls | grep -E "tr[0-3]" | cut -d. -f1 | xargs -I {} screen -X -S {} quit
