#!/bin/bash


for policy in "mincost" "minlatency"
do
  for seed in {0..9}
  do
    # set variables
    budget=150
    k=6
    j=4

    echo "Running Seed: ${seed}"
    exp_name="biodex-final-${policy}-k6-j4-budget150-seed${seed}"
    python biodex-demo.py --progress --policy $policy --val-examples 20 --k 6 --j 4 --sample-budget 150 --seed $seed --exp-name $exp_name --gpt4-mini-only

  done
done
