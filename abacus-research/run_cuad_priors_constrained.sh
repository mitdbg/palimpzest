#!/bin/bash

for sample_budget in 5 10 20 50
do
  for seed in {0..9}
  do
    k=0
    j=0
    if [[ $sample_budget -eq 5 ]]; then
      k=2
      j=3
    elif [[ $sample_budget -eq 10 ]]; then
      k=3
      j=2
    elif [[ $sample_budget -eq 20 ]]; then
      k=3
      j=3
    elif [[ $sample_budget -eq 50 ]]; then
      k=6
      j=4
    fi

    # run without priors
    exp_name="cuad-no-priors-constrained-k${k}-j${j}-budget${sample_budget}-seed${seed}"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- priors: NO PRIORS -- k: ${k} -- j: ${j} -- budget: ${sample_budget}"
      python cuad-demo.py --constrained --k $k --j $j --sample-budget $sample_budget --seed $seed --exp-name $exp_name
    fi

    # run with sample based priors
    exp_name="cuad-with-priors-constrained-k${k}-j${j}-budget${sample_budget}-seed${seed}"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- priors: WITH PRIORS -- k: ${k} -- j: ${j} -- budget: ${sample_budget}"
      python cuad-demo.py --constrained --priors-file cuad-priors.json --k $k --j $j --sample-budget $sample_budget --seed $seed --exp-name $exp_name
    fi

    # run with cheap priors 
    exp_name="cuad-cheap-priors-constrained-k${k}-j${j}-budget${sample_budget}-seed${seed}"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- priors: CHEAP PRIORS -- k: ${k} -- j: ${j} -- budget: ${sample_budget}"
      python cuad-demo.py --constrained --priors-file cheap-priors.json --k $k --j $j --sample-budget $sample_budget --seed $seed --exp-name $exp_name
    fi
  done
done
