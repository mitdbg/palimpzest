#!/bin/bash

for seed in {0..9}
do
    for priors in none naive sample
    do
        for sentinel in mab random
        do
            for opt in pareto greedy
            do
                k=6
                j=4
                if [[ $sentinel -eq "random" ]]; then
                    j=8
                fi
                priors_file="none"
                if [[ $priors -eq "naive" ]]; then
                    priors_file="cheap-priors.json"
                elif [[ $priors -eq "sample" ]]; then
                    priors_file="biodex-priors.json"
                fi

                exp_name="ablation-${priors}-${sentinel}-${opt}-seed${seed}"
                FILE="ablation-data/${exp_name}-metrics.json"
                if [ -f $FILE ]; then
                    echo "Skipping because $FILE exists."
                else
                    echo "Running Seed: ${seed} -- priors: ${priors} (${priors_file}) -- sentinel: ${sentinel} -- k: ${k} -- j: ${j} -- opt: ${opt}"
                    python biodex-ablation.py --priors-file $priors_file --k $k --j $j --sample-budget 150 --optimizer-strategy $opt --sentinel-execution-strategy $sentinel --seed $seed --exp-name $exp_name
                fi
            done
        done
    done
done
