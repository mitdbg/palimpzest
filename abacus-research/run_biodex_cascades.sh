#!/bin/bash


for seed in {0..9}
do
  for budget in 150 300 450
  do
    for strategy in "greedy" "pareto"
    do
      cost=0.5
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
        j=6
      fi
      # no priors
      exp_name="biodex-${strategy}-cost${cost}-budget${budget}-k${k}-j${j}-seed${seed}"
      FILE="pareto-cascades-data/${exp_name}-metrics.json"
      if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
      else
        echo "Running Seed: ${seed} -- cost: ${cost} -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
        python biodex-pareto-cascades.py --progress --k $k --j $j --sample-budget $budget --optimizer-strategy $strategy --cost $cost --seed $seed --exp-name $exp_name
      fi

      # sample priors
      exp_name="biodex-${strategy}-cost${cost}-with-priors-budget${budget}-k${k}-j${j}-seed${seed}"
      FILE="pareto-cascades-data/${exp_name}-metrics.json"
      if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
      else
        echo "Running Seed: ${seed} -- cost: ${cost} -- SAMPLE PRIORS -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
        python biodex-pareto-cascades.py --progress --priors-file biodex-priors-cascades.json --k $k --j $j --sample-budget $budget --optimizer-strategy $strategy --cost $cost --seed $seed --exp-name $exp_name
      fi

      # naive priors
      exp_name="biodex-${strategy}-cost${cost}-cheap-priors-budget${budget}-k${k}-j${j}-seed${seed}"
      FILE="pareto-cascades-data/${exp_name}-metrics.json"
      if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
      else
        echo "Running Seed: ${seed} -- cost: ${cost} -- CHEAP PRIORS -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
        python biodex-pareto-cascades.py --progress --priors-file cheap-priors-cascades.json --k $k --j $j --sample-budget $budget --optimizer-strategy $strategy --cost $cost --seed $seed --exp-name $exp_name
      fi
    done
  done
done
