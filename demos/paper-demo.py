import argparse
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from palimpzest.constants import Cardinality
from palimpzest.core.data.datasources import UserSource
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, Field, ImageFilepathField, ListField, NumericField, StringField
from palimpzest.core.lib.schemas import Schema, Table, TextFile, XLSFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils.udfs import xls_to_tables

# Addresses far from MIT; we use a simple lookup like this to make the
# experiments re-producible w/out needed a Google API key for geocoding lookups
FAR_AWAY_ADDRS = [
    "Melcher St",
    "Sleeper St",
    "437 D St",
    "Seaport Blvd",
    "50 Liberty Dr",
    "Telegraph St",
    "Columbia Rd",
    "E 6th St",
    "E 7th St",
    "E 5th St",
]


def within_two_miles_of_mit(record: dict):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    try:
        return not any([street.lower() in record["address"].lower() for street in FAR_AWAY_ADDRS])
    except Exception:
        return False


def in_price_range(record: dict):
    try:
        price = record["price"]
        if isinstance(price, str):
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price <= 2e6
    except Exception:
        return False


class Email(TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = StringField(desc="The email address of the sender")
    subject = StringField(desc="The subject of the email")


class CaseData(Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = Field(desc="The ID of the case")
    age_at_diagnosis = Field(desc="The age of the patient at the time of diagnosis")
    race = Field(
        desc="An arbitrary classification of a taxonomic group that is a division of a species.",
    )
    ethnicity = Field(
        desc="Whether an individual describes themselves as Hispanic or Latino or not.",
    )
    gender = Field(desc="Text designations that identify gender.")
    vital_status = Field(desc="The vital status of the patient")
    ajcc_pathologic_t = Field(desc="The AJCC pathologic T")
    ajcc_pathologic_n = Field(desc="The AJCC pathologic N")
    ajcc_pathologic_stage = Field(desc="The AJCC pathologic stage")
    tumor_grade = Field(desc="The tumor grade")
    tumor_focality = Field(desc="The tumor focality")
    tumor_largest_dimension_diameter = Field(desc="The tumor largest dimension diameter")
    primary_diagnosis = Field(desc="The primary diagnosis")
    morphology = Field(desc="The morphology")
    tissue_or_organ_of_origin = Field(desc="The tissue or organ of origin")
    # tumor_code = Field(desc="The tumor code")
    filename = Field(desc="The name of the file the record was extracted from")
    study = Field(
        desc="The last name of the author of the study, from the table name",
    )


class RealEstateListingFiles(Schema):
    """The source text and image data for a real estate listing."""

    listing = StringField(desc="The name of the listing")
    text_content = StringField(desc="The content of the listing's text description")
    image_filepaths = ListField(
        element_type=ImageFilepathField,
        desc="A list of the filepaths for each image of the listing",
    )


class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""

    address = StringField(desc="The address of the property")
    price = NumericField(desc="The listed price of the property")


class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""

    is_modern_and_attractive = BooleanField(
        desc="True if the home interior design is modern and attractive and False otherwise"
    )
    has_natural_sunlight = BooleanField(
        desc="True if the home interior has lots of natural sunlight and False otherwise"
    )


class RealEstateListingSource(UserSource):
    def __init__(self, dataset_id, listings_dir):
        super().__init__(RealEstateListingFiles, dataset_id)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def copy(self):
        return RealEstateListingSource(self.dataset_id, self.listings_dir)

    def __len__(self):
        return len(self.listings)

    def get_size(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob("*"))

    def get_item(self, idx: int):
        # fetch listing
        listing = self.listings[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=listing)
        dr.listing = listing
        dr.image_filepaths = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    dr.text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                dr.image_filepaths.append(os.path.join(listing_dir, file))

        return dr


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--viz", default=False, action="store_true", help="Visualize output in Gradio")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument(
        "--workload", type=str, help="The workload to run. One of enron, real-estate, medical-schema-matching."
    )
    parser.add_argument(
        "--executor",
        type=str,
        help="The plan executor to use. One of sequential, pipelined_single_thread, pipelined_parallel",
        default="sequential",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default="mincost",
    )

    args = parser.parse_args()

    # The user has to indicate the dataset id and the workload
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.workload is None:
        print("Please provide a workload")
        exit(1)

    # create directory for profiling data
    if args.profile:
        os.makedirs("profiling-data", exist_ok=True)

    datasetid = args.datasetid
    workload = args.workload
    visualize = args.viz
    verbose = args.verbose
    profile = args.profile
    policy = MaxQuality()
    if args.policy == "mincost":
        policy = MinCost()
    elif args.policy == "mintime":
        policy = MinTime()
    elif args.policy == "maxquality":
        policy = MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    if workload == "enron":
        # datasetid="enron-eval" for paper evaluation
        plan = Dataset(datasetid, schema=Email)
        plan = plan.filter(
            "The email is not quoting from a news article or an article written by someone outside of Enron"
        )
        plan = plan.filter(
            'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
        )

    elif workload == "real-estate":
        # datasetid="real-estate-eval-100" for paper evaluation
        data_filepath = f"testdata/{datasetid}"
        user_dataset_id = f"{datasetid}-user"
        DataDirectory().register_user_source(
            src=RealEstateListingSource(user_dataset_id, data_filepath),
            dataset_id=user_dataset_id,
        )
        plan = Dataset(user_dataset_id, schema=RealEstateListingFiles)
        plan = plan.convert(TextRealEstateListing, depends_on="text_content")
        plan = plan.convert(ImageRealEstateListing, depends_on="image_filepaths")
        plan = plan.filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    elif workload == "medical-schema-matching":
        # datasetid="biofabric-medium" for paper evaluation
        plan = Dataset(datasetid, schema=XLSFile)
        plan = plan.convert(Table, udf=xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
        plan = plan.filter("The rows of the table contain the patient age")
        plan = plan.convert(CaseData, desc="The patient data in the table", cardinality=Cardinality.ONE_TO_MANY)

    config = QueryProcessorConfig(
        nocache=True,
        verbose=verbose,
        policy=policy,
        execution_strategy=args.executor)

    # Option 1: Use QueryProcessorFactory to create a processor
    # processor = QueryProcessorFactory.create_processor(
    #     datasource=plan,
    #     processing_strategy="no_sentinel",  
    #     execution_strategy="sequential", 
    #     optimizer_strategy="pareto",
    #     config=config
    # )
    # records, execution_stats = processor.execute()

    # Option 2: Use Dataset.run() to run the plan.
    data_record_collection = plan.run(config)
    print(data_record_collection.to_df())
    # save statistics

    if profile:
        stats_path = f"profiling-data/{workload}-profiling.json"
        execution_stats_dict = data_record_collection.execution_stats.to_json()
        with open(stats_path, "w") as f:
            json.dump(execution_stats_dict, f)

    # visualize output in Gradio
    if visualize:
        from palimpzest.utils.demo_helpers import print_table

        plan_str = list(data_record_collection.execution_stats.plan_strs.values())[-1]
        if workload == "enron":
            print_table(data_record_collection.data_records, cols=["sender", "subject"], plan_str=plan_str)

        elif workload == "real-estate":
            fst_imgs, snd_imgs, thrd_imgs, addrs, prices = [], [], [], [], []
            for record in data_record_collection:
                addrs.append(record.address)
                prices.append(record.price)
                for idx, img_name in enumerate(["img1.png", "img2.png", "img3.png"]):
                    path = os.path.join(f"testdata/{datasetid}", record.listing, img_name)
                    img = Image.open(path)
                    img_arr = np.asarray(img)
                    if idx == 0:
                        fst_imgs.append(img_arr)
                    elif idx == 1:
                        snd_imgs.append(img_arr)
                    elif idx == 2:
                        thrd_imgs.append(img_arr)

            with gr.Blocks() as demo:
                fst_img_blocks, snd_img_blocks, thrd_img_blocks, addr_blocks, price_blocks = [], [], [], [], []
                for fst_img, snd_img, thrd_img, addr, price in zip(fst_imgs, snd_imgs, thrd_imgs, addrs, prices):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            fst_img_blocks.append(gr.Image(value=fst_img))
                        with gr.Column():
                            snd_img_blocks.append(gr.Image(value=snd_img))
                        with gr.Column():
                            thrd_img_blocks.append(gr.Image(value=thrd_img))
                    with gr.Row():
                        with gr.Column():
                            addr_blocks.append(gr.Textbox(value=addr, info="Address"))
                        with gr.Column():
                            price_blocks.append(gr.Textbox(value=price, info="Price"))

                plan_str = list(data_record_collection.execution_stats.plan_strs.values())[0]
                gr.Textbox(value=plan_str, info="Query Plan")

            demo.launch()
