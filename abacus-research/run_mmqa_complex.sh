#!/bin/bash

# for seed in {0..9}
# Lotus error'ed on seed 5 and 7, so we limit to these seeds only for a consistent comparison
for seed in 0 1 2 3 4 6 8 9
do
    policy="maxquality"
    exp_name="mmqa-complex-${policy}-k6-j4-budget350-seed${seed}"
    FILE="mmqa-complex-data/${exp_name}-stats.json"
    if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
    else
        echo "Running Seed: ${seed} -- ${policy}"
        python mmqa-complex-demo.py --progress --k 6 --j 4 --sample-budget 350 --seed $seed --exp-name $exp_name --gpt4-mini-only
    fi
done
