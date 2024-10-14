import json
import os
import random
import shutil

def generate_fever(data, num_samples, out_folder):    
    for i in range(num_samples):
        out_file = f"{out_folder}/{data[i]["id"]}"
        content = data[i]["claim"]
        # print(content, out_file)
        
        with open(out_file, "w") as f:
            f.write(content)

if __name__ == "__main__":
    in_file = "testdata/paper_test.jsonl"
    
    with open(in_file, "r") as f:
        data = [json.loads(line) for line in f]
        
    random.shuffle(data)
    
    # for num_samples in [10]:
    for num_samples in [10, 100, 1000]:
        out_folder = f"testdata/fever-{num_samples}"
        
        try:
            os.mkdir(out_folder)
        except FileExistsError:
            shutil.rmtree(out_folder)
            os.mkdir(out_folder)

        generate_fever(data, num_samples, out_folder)
        pass    