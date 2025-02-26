import json
import os
import tarfile
import urllib.request

import requests

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pneuma import Pneuma

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_src")


def post_process_dataset(dataset_path: str):
    for table in os.listdir(dataset_path):
        os.rename(
            f"{dataset_path}/{table}", f"{dataset_path}/{table.split('_SEP_')[1]}"
        )
    print("Dataset processed")


def extract_tar(tar_name: str, extract_path="."):
    try:
        with tarfile.open(tar_name, "r") as tar:
            tar.extractall(path=extract_path)
            print(f"Extracted all files to '{extract_path}'")
        os.remove(tar_name)
        print(f"Removed the tar file: '{tar_name}'")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_chembl():
    if "pneuma_chembl_10K" in os.listdir(DATA_DIR):
        print("Dataset already downloaded")
        return

    tar_name = "pneuma_chembl_10K.tar"
    urllib.request.urlretrieve(
        f"https://storage.googleapis.com/pneuma_open/{tar_name}",
        filename=os.path.join(DATA_DIR, tar_name),
    )
    extract_tar(os.path.join(DATA_DIR, tar_name), DATA_DIR)
    post_process_dataset(os.path.join(DATA_DIR, tar_name[:-4]))


def download_questions():
    if "questions.jsonl" in os.listdir(DATA_DIR):
        print("Questions already downloaded")
        return

    URL = "https://docs.google.com/uc?export=download"
    FILE_ID = "1vdddvHeHdNgAquceBEO4dK-LMyLUtPv1"

    session = requests.Session()
    response = session.get(URL, params={"id": FILE_ID}, stream=True)

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(
                URL, params={"id": FILE_ID, "confirm": value}, stream=True
            )
            break

    with open(os.path.join(DATA_DIR, "questions.jsonl"), "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"Questions downloaded")


def main():
    # Step 1: Download dataset & questions
    if "data_src" not in os.listdir(SCRIPT_DIR):
        os.mkdir(DATA_DIR)
    download_chembl()
    download_questions()

    # Step 2: Initialize Pneuma
    out_path = "pneuma-demo"
    pneuma = Pneuma(
        out_path=out_path,
        llm_path="Qwen/Qwen2.5-7B-Instruct",  # Change to a smaller LLM if necessary
        embed_path="BAAI/bge-base-en-v1.5",
    )

    # Alternative: Use OpenAI models (gpt-4o-mini & text-embedding-3-small); may take longer
    # pneuma = Pneuma(
    #     out_path=out_path,
    #     openai_api_key=os.environ['OPENAI_API_KEY'],
    #     use_local_model=False,
    # )

    pneuma.setup()

    # Step 3: Register dataset
    data_path = "data_src/pneuma_chembl_10K"
    response = pneuma.add_tables(path=data_path, creator="pneuma_pz_demo")
    response = json.loads(response)
    print(response)

    # Step 4: Summarize dataset
    response = pneuma.summarize()
    response = json.loads(response)
    print(response)

    # Step 5: Generate index
    response = pneuma.generate_index(index_name="demo_index")
    response = json.loads(response)
    print(response)


if __name__ == "__main__":
    main()
