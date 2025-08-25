#!/usr/bin/env python
"""
Script to download CUAD dataset and set up local data directory.
This replaces the need for HuggingFace datasets library.
"""

import os
import urllib.request
import zipfile


def setup_cuad_data():
    # Create cuad-data directory
    data_dir = "cuad-data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Download CUAD data zip file
    data_url = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
    zip_path = os.path.join(data_dir, "data.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading CUAD data from {data_url}...")
        urllib.request.urlretrieve(data_url, zip_path)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"Data already downloaded at {zip_path}")
    
    # Extract the zip file
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"Extracted data to {data_dir}")
    
    # Download the dataset loading script (for reference, not actually used)
    script_url = "https://huggingface.co/datasets/theatticusproject/cuad-qa/resolve/main/cuad-qa.py"
    script_path = os.path.join(data_dir, "cuad-qa.py")
    
    if not os.path.exists(script_path):
        print(f"Downloading CUAD dataset script from {script_url}...")
        urllib.request.urlretrieve(script_url, script_path)
        print(f"Downloaded to {script_path}")
    else:
        print(f"Script already exists at {script_path}")
    
    # List extracted files
    print("\nExtracted files:")
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            file_path = os.path.join(data_dir, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  - {file} ({size:.2f} MB)")
    
    print("\nSetup complete! CUAD data is ready in the 'cuad-data' directory.")
    print("\nTo use this data in your scripts, update the data loading to:")
    print("  - train data: cuad-data/train_separate_questions.json")
    print("  - test data: cuad-data/test.json")
    
    return data_dir

if __name__ == "__main__":
    setup_cuad_data()