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

# for metric in "cost" "latency"
# do
#   for seed in {0..9}
#   do
#     # set variables
#     budget=150
#     k=6
#     j=4

    # # no priors
    # exp_name="biodex-pareto-min-${metric}-budget${budget}-k${k}-j${j}-seed${seed}"
    # FILE="min-${metric}-at-quality-data/${exp_name}-metrics.json"
    # if [ -f $FILE ]; then
    #   echo "Skipping because $FILE exists."
    # else
    #   echo "Running Seed: ${seed} -- metric: ${metric} -- budget: ${budget} -- k: ${k} -- j: ${j} -- strategy: ${strategy}"
    #   python biodex-min-at-fixed-quality.py --progress --k $k --j $j --sample-budget $budget --metric $metric --seed $seed --exp-name $exp_name
    # fi

#   done
# done
