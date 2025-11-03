#!/bin/bash

for policy in "mincost" "minlatency"
do
    for seed in {0..9}
    do
        echo "Running Seed: ${seed}"
        exp_name="mmqa-final-${policy}-k6-j4-budget150-seed${seed}"
        python mmqa-demo.py --progress --policy $policy --k 6 --j 4 --sample-budget 150 --seed $seed --exp-name $exp_name --gpt4-mini-only
    done
done
