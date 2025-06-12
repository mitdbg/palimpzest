#!/bin/bash

for seed in {0..9}
do
  echo "Running Seed: ${seed}"
  exp_name="cuad-final-mab-k6-j4-budget50-seed${seed}-sequential-parallel"
  python demos/cuad-demo.py --k 6 --j 4 --sample-budget 50 --seed $seed --exp-name $exp_name --execution-strategy sequential-parallel
done
