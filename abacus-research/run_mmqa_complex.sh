#!/bin/bash

# for seed in {0..9}
# for seed in 0 1 2 3 4 6 8 9
for seed in 3 4 6 8 9
do
    echo "Running Seed: ${seed}"
    exp_name="mmqa-complex-final-mab-k6-j4-budget250-seed${seed}"
    python mmqa-complex-demo.py --progress --k 6 --j 4 --sample-budget 250 --seed $seed --exp-name $exp_name --gpt4-mini-only
done
