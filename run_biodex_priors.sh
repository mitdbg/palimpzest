#!/bin/bash

for sample_budget in 10 20 50 100 # 5
do
  for seed in {0..9}
  do
    k=0
    j=0
    if [[ $sample_budget -eq 5 ]]; then
      k=2
      j=1
    elif [[ $sample_budget -eq 10 ]]; then
      k=2
      j=2
    elif [[ $sample_budget -eq 20 ]]; then
      k=2
      j=2
    elif [[ $sample_budget -eq 50 ]]; then
      k=3
      j=3
    elif [[ $sample_budget -eq 100 ]]; then
      k=4
      j=4
    elif [[ $sample_budget -eq 150 ]]; then
      k=6
      j=4
    fi

    # run without priors
    exp_name="biodex-no-priors-k${k}-j${j}-budget${sample_budget}-seed${seed}"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- priors: NO PRIORS -- k: ${k} -- j: ${j} -- budget: ${sample_budget}"
      python demos/biodex-demo.py --progress --k $k --j $j --sample-budget $sample_budget --seed $seed --exp-name $exp_name
    fi

    # run with sample-based priors
    exp_name="biodex-with-priors-k${k}-j${j}-budget${sample_budget}-seed${seed}"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- priors: WITH PRIORS -- k: ${k} -- j: ${j} -- budget: ${sample_budget}"
      python demos/biodex-demo.py --progress --priors-file biodex-priors.json --k $k --j $j --sample-budget $sample_budget --seed $seed --exp-name $exp_name
    fi

    # run with cheap priors 
    exp_name="biodex-cheap-priors-k${k}-j${j}-budget${sample_budget}-seed${seed}"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
      echo "Skipping because $FILE exists."
    else
      echo "Running Seed: ${seed} -- priors: CHEAP PRIORS -- k: ${k} -- j: ${j} -- budget: ${sample_budget}"
      python demos/biodex-demo.py --progress --priors-file cheap-priors.json --k $k --j $j --sample-budget $sample_budget --seed $seed --exp-name $exp_name
    fi
  done
done
