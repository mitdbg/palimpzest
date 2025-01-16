import hashlib
import io
import json
import os
import time
from typing import BinaryIO, List
from zipfile import ZipFile

import pandas as pd
import requests
from fastapi import status
from pypdf import PdfReader

from palimpzest.config import Config

COSMOS_ADDRESS = "https://xdd.wisc.edu/cosmos_service"


class PdfParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        with open(pdf_path, "rb") as f:
            self.pdf = f.read()
        self.text = ""
        self.pages = []
        self._parse()

    def _parse(self):
        for page in self.pdf:
            self.text += page.get_text()  # type: ignore
            self.pages.append(page.get_text())  # type: ignore

    def get_text(self) -> str:
        return self.text

    def get_pages(self) -> List[str]:
        return self.pages


def get_md5(file_bytes: bytes) -> str:
    if not isinstance(file_bytes, bytes):
        file_bytes = file_bytes.encode()
    return hashlib.md5(file_bytes).hexdigest()


##
# Function to extract a Cosmos parquet file to Cosmos JSON
##
def cosmos_parquet_to_json(path):
    parquet_df = pd.read_parquet(path)
    parquet_json = parquet_df.to_json()
    parquet_data = json.loads(parquet_json)

    if len(parquet_data) > 0:
        parquet_data_keys = list(parquet_data.keys())
        num_data_rows = max([int(k) for k in parquet_data[parquet_data_keys[0]]])

        row_order_parquet_data = [dict() for i in range(num_data_rows + 1)]
        for field_key, row_data in parquet_data.items():
            for row_idx, datum in row_data.items():
                row_idx_num = int(row_idx)
                row_order_parquet_data[row_idx_num][field_key] = datum

        row_order_parquet_data.sort(
            key=lambda d: (
                d["page_num"],
                d["bounding_box"][0] // 500,
                d["bounding_box"][1],
            )
        )

        edits = list()
        for e1, extraction1 in enumerate(row_order_parquet_data):
            (ext1_x1, ext1_y1, ext1_x2, ext1_y2) = extraction1["bounding_box"]
            if ext1_x1 < 500:
                continue

            ext1_page_num = extraction1["page_num"]
            found_col_break = False
            insertion_index = -1
            t1 = e1
            while t1 > 0:
                extraction2 = row_order_parquet_data[t1 - 1]
                ext2_page_num = extraction2["page_num"]
                if ext1_page_num > ext2_page_num:
                    break

                (ext2_x1, ext2_y1, ext2_x2, ext2_y2) = extraction2["bounding_box"]

                if ext1_y2 <= ext2_y1:
                    ext2_xspan = ext2_x2 - ext2_x1
                    if ext2_xspan >= 800:
                        found_col_break = True
                        insertion_index = t1 - 1
                t1 -= 1
            if found_col_break:
                edits.append(
                    {
                        "del_idx": e1,
                        "ins_idx": insertion_index,
                        "val": extraction1,
                    }
                )
        for edit_dict in edits:
            del row_order_parquet_data[edit_dict["del_idx"]]
            row_order_parquet_data.insert(edit_dict["ins_idx"], edit_dict["val"])
        row_order_parquet_data.sort(key=lambda d: (d["pdf_name"]))

        name2results = dict()
        for row_data in row_order_parquet_data:
            if row_data["pdf_name"] in name2results:
                name2results[row_data["pdf_name"]].append(row_data)
            else:
                name2results[row_data["pdf_name"]] = [row_data]

        return next(iter(name2results.items()))[1]


##
# Function to extract the text 'content' attribute from the Cosmos JSON data
##
def cosmos_json_txt(cosmos_json):
    # Initialize an empty list to store the content texts
    content_texts = []

    # Iterate over each item in the JSON data
    for item in cosmos_json:
        # Extract the 'content' attribute and add it to the list
        content_texts.append(item.get("content", ""))

    return content_texts


def cosmos_client(name: str, data: BinaryIO, output_dir: str, delay=10):
    files = [
        ("pdf", (name, data, "application/pdf")),
    ]
    print(f"Sending {name} to COSMOS")
    response = requests.post(f"{COSMOS_ADDRESS}/process/", files=files)
    print(f"Received response of  {response.json()['status_endpoint']} from COSMOS: {response.status_code}")
    # get md5 of the data
    md5 = get_md5(data)

    if response.status_code == status.HTTP_202_ACCEPTED:
        callback_endpoints = response.json()

        for retry_num in range(400):
            time.sleep(delay)
            poll = requests.get(f"{callback_endpoints['status_endpoint']}")
            print(f"Polling COSMOS on retry num {retry_num + 1}")
            if poll.status_code == status.HTTP_200_OK:
                poll_results = poll.json()
                if poll_results["job_completed"]:
                    cosmos_response = requests.get(f"{callback_endpoints['result_endpoint']}")
                    if cosmos_response.status_code == status.HTTP_200_OK:
                        data = cosmos_response.content
                        with ZipFile(io.BytesIO(data)) as z:
                            output_subdir = os.path.join(
                                output_dir, f"COSMOS_{os.path.splitext(name)[0].replace(' ', '_')}_{md5}"
                            )
                            os.makedirs(output_subdir, exist_ok=True)
                            z.extractall(path=output_subdir)
                            for file in os.listdir(output_subdir):
                                if (
                                    file.endswith(".parquet")
                                    and not file.endswith("_figures.parquet")
                                    and not file.endswith("_pdfs.parquet")
                                    and not file.endswith("_tables.parquet")
                                    and not file.endswith("_sections.parquet")
                                    and not file.endswith("_equations.parquet")
                                ):
                                    print(f"Converting {file} to JSON")
                                    # if error while converting parquet to json, skip this file
                                    try:
                                        json_data = cosmos_parquet_to_json(os.path.join(output_subdir, file))
                                        with open(
                                            os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.json"), "w"
                                        ) as json_file:
                                            json.dump(json_data, json_file)
                                        with open(
                                            os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.txt"), "w"
                                        ) as text_file:
                                            text_file.write("\n".join(cosmos_json_txt(json_data)))
                                        # print(f"{file} : {json_data}")

                                    except Exception as e:
                                        print(f"Error while converting {file} to JSON: {e}")
                                        pass
                        return
                        # raise RuntimeError("COSMOS data doesn't include document file for annotation")

                    else:
                        raise RuntimeError(
                            f"COSMOS Result Error - STATUS CODE: {response.status_code} - {COSMOS_ADDRESS}"
                        )
                # If not, just wait until the next iteration
                else:
                    pass

        # If we reached this point, we time out
        raise TimeoutError(f"Timed out waiting for COSMOS on retry num {retry_num + 1}")

    else:
        raise RuntimeError(f"COSMOS Error - STATUS CODE: {response.status_code} - {COSMOS_ADDRESS}")


##
# Function to extract the text from a PDF file:
# 1. Check if the text file already exists in the cache, if so, read from the cache
# 2. If not, call the cosmos_client function to process the PDF file and cache the text file
##
# NOTE: I don't believe anyone actively depends on this function, but we need to remove the
# dependency on DataDirectory() in order to prevent circular imports. The long-term solution
# is to separate out the pieces of DataDirectory which the DataSources depend on, from the
# pieces which are related to setting / reading external configurations (like "pdfprocessor").
# However, given that I can fix this in two minutes by adding this is a kwarg, I'm going to
# do that for now and revisit the issue if/when this matters.

# TODO(Jun): 1. cosmos returns 202 for me. 2. why only accept "pypdf" and "cosmos" as pdfprocessor?
def get_text_from_pdf(filename, pdf_bytes, pdfprocessor="cosmos", enable_file_cache=True, file_cache_dir="/tmp"):
    pdf_filename = filename
    file_name = os.path.basename(pdf_filename)
    file_name_without_extension = os.path.splitext(file_name)[0]
    text_file_name = f"{file_name_without_extension}.txt"

    if pdfprocessor == "pypdf":
        pdf = PdfReader(io.BytesIO(pdf_bytes))
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
        return all_text
        # return pdf.pages[0].extract_text() # TODO we can only return first page
    else:
        # Get md5 of the pdf_bytes
        md5 = get_md5(pdf_bytes)
        cached_extraction_folder = f"COSMOS_{os.path.splitext(file_name)[0].replace(' ', '_')}_{md5}"
        # Check if pz_file_cache_dir exists in the file system
        pz_file_cache_dir = os.path.join(file_cache_dir, cached_extraction_folder)
        if enable_file_cache and os.path.exists(pz_file_cache_dir):
            print(f"File {text_file_name} already exists in system tmp folder {pz_file_cache_dir}, reading from cache")
            text_file_path = os.path.join(pz_file_cache_dir, text_file_name)
            with open(text_file_path) as file:
                text_content = file.read()
                return text_content

        #
        # CHUNWEI: This code has a bug
        # It checks to see if the text file name is in the registry, but there are two things wrong here.
        # 1) The registry is for 'official' datasets that have been inserted by the user, not cached objects.
        # 2) The filename isn't enough to check for cached results. Maybe the file moved directories, or maybe there are
        # multiple different files with the same name. You need the checksum of the original file to ensure the cached
        # object is valid.
        #
        #    if DataDirectory().exists(text_file_name):
        #        print(f"Text file {text_file_name} already exists, reading from cache")
        #        text_file_path = DataDirectory().get_path(text_file_name)
        #        with open(text_file_path, 'r') as file:
        #            text_content = file.read()
        #            return text_content
        # cosmos_file_dir = file_name_without_extension.replace(" ", "_")
        # get a tmp of the system temp directory

        print(f"Processing {file_name} through COSMOS")
        # Call the cosmos_client function
        cosmos_client(file_name, pdf_bytes, file_cache_dir)
        text_file_path = os.path.join(pz_file_cache_dir, text_file_name)
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Text file {text_file_name} not found in {pz_file_cache_dir}/{text_file_name}")
        # DataDirectory().register_local_file(text_file_path, text_file_name)
        with open(text_file_path) as file:
            text_content = file.read()
            return text_content


if __name__ == "__main__":
    config = Config("default")
    file_path = "../../../testdata/pdfs-tiny/battery.pdf"
    # output_dir = "../../../tests/testFileDirectory/cosmos"
    with open(file_path, "rb") as file:
        text = get_text_from_pdf(file_path, file.read())
        print(text)
        # file_name = os.path.basename(file_path)
        # # Call the cosmos_client function
        # cosmos_client(file_name, file, output_dir)
    # DataDirectory().rm_registered_dataset("sidarthe.annotations.txt")
