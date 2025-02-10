# PZ SWE-BENCH

This directory is dedicated to developing and evaluating PZ programs for the SWE-Bench benchmark. The below steps detail on how to generate the swe-bench dataset documents and run the current PZ swe-bench scripts to generate patches.

## Generating SWE-Bench Documents

1. Run `python create_dataset.py --num_instances <num_instances>` where num_instances is the number of code issues you would like to generate patches for. There should now be a new directory in the `testdata` directory named `swe-bench-oracle-lite` containing the code issues
2. Register the dataset in PZ by executing `pz reg --path testdata/swe-bench-oracle-lite --name swe-bench-oracle-lite`

## Generating Patches

1. cd into the `swe-bench` directory and run `python swe-bench.py`. Once complete, the patches will be output into a `output.json` file.
