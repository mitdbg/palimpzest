from datasets import load_dataset
import json
import os 

NUM_INSTANCES = 20

# Load the SWE-bench_oracle dataset
dataset = load_dataset("princeton-nlp/SWE-bench_oracle")

# Access the training set, for example
test_data = dataset['test']

output_dir = "../testdata/swe-bench-oracle-lite"
os.makedirs(output_dir, exist_ok=True)

for idx, row in enumerate(test_data):
  # Just do one for testing
  if idx > NUM_INSTANCES: 
    break 

  instance_id = row["instance_id"]
  file_path = os.path.join(output_dir, f"{instance_id}.txt")

  with open(file_path, 'w') as f: 
    json.dump(row, f, indent=4)