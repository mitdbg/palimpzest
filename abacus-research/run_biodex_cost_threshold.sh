#!/bin/bash


for cost in 1.0 2.0 4.0 8.0 999.99
do
  for seed in {0..9}
  do
    # set variables
    budget=450
    k=48
    j=3

    # no priors
    exp_name="biodex-pareto-cost${cost}-budget${budget}-k${k}-j${j}-seed${seed}"
    FILE="max-quality-at-cost-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- cost: ${cost} -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
      python biodex-max-quality-at-cost.py --progress --k $k --j $j --sample-budget $budget --cost $cost --seed $seed --exp-name $exp_name
    fi

    # sample priors
    exp_name="biodex-pareto-cost${cost}-with-priors-budget${budget}-k${k}-j${j}-seed${seed}"
    FILE="max-quality-at-cost-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- cost: ${cost} -- SAMPLE PRIORS -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
      python biodex-max-quality-at-cost.py --progress --priors-file biodex-priors.json --k $k --j $j --sample-budget $budget --cost $cost --seed $seed --exp-name $exp_name
    fi
  done
done
