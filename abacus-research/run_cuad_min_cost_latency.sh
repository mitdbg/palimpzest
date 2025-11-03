#!/bin/bash

for policy in "mincost" "minlatency"
do
  for seed in {0..9}
  do
    echo "Running Seed: ${seed}"
    exp_name="cuad-final-${policy}-k6-j4-budget50-seed${seed}"
    python cuad-demo.py --policy $policy --k 6 --j 4 --sample-budget 50 --seed $seed --exp-name $exp_name --gpt4-mini-only
  done
done
