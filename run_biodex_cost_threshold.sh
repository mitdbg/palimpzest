#!/bin/bash


for cost in 1.0 2.0 4.0 8.0 999.99 # 8.0 4.0 2.0 1.0
do
  for seed in {0..9}
  do
    for budget in 450 # 150 300 450
    do
      k=0
      j=0
      if [[ $budget -eq 150 ]]; then
        k=6
        j=4
      elif [[ $budget -eq 300 ]]; then
        k=24
        j=5
      elif [[ $budget -eq 450 ]]; then
        k=48
        j=3
      elif [[ $budget -eq 1000 ]]; then
        k=96
        j=5
      fi
      # no priors
      exp_name="biodex-pareto-cost${cost}-budget${budget}-k${k}-j${j}-seed${seed}"
      FILE="max-quality-at-cost-data/${exp_name}-metrics.json"
      if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
      else
        echo "Running Seed: ${seed} -- cost: ${cost} -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
        python demos/biodex-max-quality-at-cost.py --progress --k $k --j $j --sample-budget $budget --cost $cost --seed $seed --exp-name $exp_name
      fi

      # sample priors
      exp_name="biodex-pareto-cost${cost}-with-priors-budget${budget}-k${k}-j${j}-seed${seed}"
      FILE="max-quality-at-cost-data/${exp_name}-metrics.json"
      if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
      else
        echo "Running Seed: ${seed} -- cost: ${cost} -- SAMPLE PRIORS -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
        python demos/biodex-max-quality-at-cost.py --progress --priors-file biodex-priors.json --k $k --j $j --sample-budget $budget --cost $cost --seed $seed --exp-name $exp_name
      fi

      # # naive priors
      # exp_name="biodex-${strategy}-cost${cost}-cheap-priors-budget${budget}-k${k}-j${j}-seed${seed}"
      # FILE="max-quality-at-cost-data/${exp_name}-metrics.json"
      # if [ -f $FILE ]; then
      #   echo "Skipping because $FILE exists."
      # else
      #   echo "Running Seed: ${seed} -- cost: ${cost} -- CHEAP PRIORS -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
      #   python demos/biodex-max-quality-at-cost.py --progress --priors-file cheap-priors.json --k $k --j $j --sample-budget $budget --optimizer-strategy $strategy --cost $cost --seed $seed --exp-name $exp_name
      # fi
    done
  done
done
