#!/bin/bash

### REAL ESTATE
# for model in "gpt-4o" "gpt-4o-mini" "llama" "mixtral"
# do
#     python demos/optimizerDemo.py --verbose --workload real-estate --datasetid real-estate-eval-100 --engine nosentinel --executor sequential --policy maxquality --model $model
# done

# for rank in 2 4 6 8 10
# do
#     for num_samples in 5 10 15 20 25
#     do
#         if (( $num_samples < $rank + 1 )); then
#             continue
#         fi
#         echo "RANK: ${rank} -- NUM_SAMPLES: ${num_samples}"
#         sleep 5.0
#         python demos/optimizerDemo.py --verbose \
#             --workload real-estate \
#             --datasetid real-estate-eval-100 \
#             --engine sentinel \
#             --executor sequential \
#             --policy maxquality \
#             --num-samples $num_samples \
#             --rank $rank
#     done
# done

# compute fully materialized sample matrices
export LOG_MATRICES="TRUE"
for num_samples in 100
do
    rank=$((num_samples - 1))
    echo "RANK: ${rank} -- NUM_SAMPLES: ${num_samples}"
    python demos/matrixCompletion.py --verbose \
                --workload real-estate \
                --datasetid real-estate-eval-100 \
                --num-samples $num_samples \
                --rank $rank
done
export LOG_MATRICES="FALSE"


### BIODEX
# for model in "gpt-4o" "gpt-4o-mini" "llama" "mixtral"
# do
#     python demos/optimizerDemo.py --verbose --workload biodex --datasetid biodex --engine nosentinel --executor sequential --policy maxquality --model $model
# done

# # compute fully materialized sample matrices
# export LOG_MATRICES="TRUE"
# for num_samples in 25 # 5 10 15 20 25
# do
#     rank=$((num_samples - 1))
#     echo "RANK: ${rank} -- NUM_SAMPLES: ${num_samples}"
#     python demos/matrixCompletion.py --verbose \
#                 --workload biodex \
#                 --datasetid biodex \
#                 --engine sentinel \
#                 --executor parallel \
#                 --policy maxquality \
#                 --num-samples $num_samples \
#                 --rank $rank
# done
# export LOG_MATRICES="FALSE"
