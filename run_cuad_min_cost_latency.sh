#!/bin/bash

for seed in {0..9}
do
  # Min. Cost At Fixed Quality
  exp_name="cuad-k6-j4-budget50-seed${seed}-mincostatfixedquality"
  FILE="opt-profiling-data/${exp_name}-metrics.json"
  if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
  else
      echo "Running Seed: ${seed} -- MinCostAtFixedQuality"
      python demos/cuad-demo.py --policy mincostatfixedquality --quality 0.475 --k 6 --j 4 --sample-budget 50 --seed $seed --exp-name $exp_name --execution-strategy sequential-parallel
  fi

  # Min. Latency At Fixed Quality
  exp_name="cuad-k6-j4-budget50-seed${seed}-minlatencyatfixedquality"
  FILE="opt-profiling-data/${exp_name}-metrics.json"
  if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
  else
      echo "Running Seed: ${seed} -- MinLatencyAtFixedQuality"
      python demos/cuad-demo.py --policy minlatencyatfixedquality --quality 0.475 --k 6 --j 4 --sample-budget 50 --seed $seed --exp-name $exp_name --execution-strategy sequential-parallel
  fi
done
