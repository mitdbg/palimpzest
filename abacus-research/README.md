## Chroma Embeddings and MMQA files
You can download the chroma embeddings we computed for MMQA and BioDEX by executing the following:
```sh
$ ./download_embeddings_and_mmqa.sh
```
This folder also contains questions for the different splits of MMQA -- of which we only use `MMQA_dev.jsonl` for scoring PZ's output. If you need the full MMQA dataset for any reason (e.g. to visualize at which images are being retrieved by a pipeline), you can find it here: https://github.com/allenai/multimodalqa/tree/master.

## Table 2
The following scripts create the data for Abacus in Table 2 in our Abacus paper.
- `run_biodex.sh`
- `run_cuad.sh`
- `run_mmqa_complex.sh`

## Table 3
The following scripts create the data for Abacus in Table 3 in our Abacus paper.
- `run_biodex_min_cost_latency.sh`
- `run_cuad_min_cost_latency.sh`
- `run_mmqa_complex_min_cost_latency.sh`

## Figure 6
The following scripts create the data for Figure 6 in our Abacus paper.
- `run_biodex_priors.sh`
- `run_biodex_priors_constrained.sh`
- `run_cuad_priors.sh`
- `run_cuad_priors_constrained.sh`

## Figure 7
The `run_biodex_cost_threshold.sh` and `run_cuad_cost_threshold.sh` scripts create the data for Figure 6 in our Abacus paper.

## Figure 8
The `run_ablation_study.sh` script creates the data for Figure 8 in our Abacus paper.