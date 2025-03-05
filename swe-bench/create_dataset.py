from datasets import load_dataset
import json
import os 
import argparse

parser = argparse.ArgumentParser(description="Process a subset of SWE-bench_oracle dataset.")
parser.add_argument("--num_instances", type=int, required=True, help="Number of instances to process.")
args = parser.parse_args()
NUM_INSTANCES = args.num_instances

# Load the SWE-bench_oracle dataset
dataset = load_dataset("princeton-nlp/SWE-bench_oracle")

# Access the training set
test_data = dataset['test']

output_dir = "../testdata/swe-bench-oracle"
os.makedirs(output_dir, exist_ok=True)

for idx, row in enumerate(test_data):
  if idx > NUM_INSTANCES: 
    break 

  instance_id = row["instance_id"]
  file_path = os.path.join(output_dir, f"{instance_id}.txt")

  with open(file_path, 'w') as f: 
    json.dump(row, f, indent=4)