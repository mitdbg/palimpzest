#!/bin/bash

for seed in {0..9}
do
  policy="maxquality"
  echo "Running Seed: ${seed} -- policy: ${policy}"
  exp_name="cuad-${policy}-k6-j4-budget50-seed${seed}"
  python cuad-demo.py --k 6 --j 4 --sample-budget 50 --seed $seed --exp-name $exp_name --gpt4-mini-only
done
