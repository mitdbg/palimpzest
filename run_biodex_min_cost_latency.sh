#!/bin/bash

for seed in {0..9}
do
    # Min. Cost At Fixed Quality
    exp_name="biodex-k6-j4-budget150-seed${seed}-mincostatfixedquality"
    FILE="opt-profiling-data/${exp_name}-metrics.json"
    if [ -f $FILE ]; then
        echo "Skipping because $FILE exists."
    else
        echo "Running Seed: ${seed} -- MinCostAtFixedQuality"
        python demos/biodex-demo.py --progress --policy mincostatfixedquality --quality 0.216 --val-examples 20 --k 6 --j 4 --sample-budget 150 --execution-strategy sequential-parallel --seed $seed --exp-name $exp_name
    fi

    # # Min. Latency At Fixed Quality
    # exp_name="biodex-k6-j4-budget150-seed${seed}-minlatencyatfixedquality"
    # FILE="opt-profiling-data/${exp_name}-metrics.json"
    # if [ -f $FILE ]; then
    #     echo "Skipping because $FILE exists."
    # else
    #     echo "Running Seed: ${seed} -- MinLatencyAtFixedQuality"
    #     python demos/biodex-demo.py --progress --policy minlatencyatfixedquality --quality 0.216 --val-examples 20 --k 6 --j 4 --sample-budget 150 --execution-strategy sequential-parallel --seed $seed --exp-name $exp_name
    # fi
done
