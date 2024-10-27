import argparse
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

import palimpzest as pz
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


def within_two_miles_of_mit(record):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    try:
        if any([street.lower() in record.address.lower() for street in FAR_AWAY_ADDRS]):
            return False
        return True
    except Exception:
        return False


def in_price_range(record):
    try:
        price = record.price
        if isinstance(price, str):
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price and price <= 2e6
    except Exception:
        return False


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)


class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(
        desc="An arbitrary classification of a taxonomic group that is a division of a species.",
        required=False,
    )
    ethnicity = pz.Field(
        desc="Whether an individual describes themselves as Hispanic or Latino or not.",
        required=False,
    )
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
    tumor_grade = pz.Field(desc="The tumor grade", required=False)
    tumor_focality = pz.Field(desc="The tumor focality", required=False)
    tumor_largest_dimension_diameter = pz.Field(desc="The tumor largest dimension diameter", required=False)
    primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
    morphology = pz.Field(desc="The morphology", required=False)
    tissue_or_organ_of_origin = pz.Field(desc="The tissue or organ of origin", required=False)
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    filename = pz.Field(desc="The name of the file the record was extracted from", required=False)
    study = pz.Field(
        desc="The last name of the author of the study, from the table name",
        required=False,
    )


class RealEstateListingFiles(pz.Schema):
    """The source text and image data for a real estate listing."""

    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.StringField(desc="The content of the listing's text description", required=True)
    image_contents = pz.ListField(
        element_type=pz.BytesField,
        desc="A list of the contents of each image of the listing",
        required=True,
    )


class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""

    address = pz.StringField(desc="The address of the property")
    price = pz.NumericField(desc="The listed price of the property")


class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""

    is_modern_and_attractive = pz.BooleanField(
        desc="True if the home interior design is modern and attractive and False otherwise"
    )
    has_natural_sunlight = pz.BooleanField(
        desc="True if the home interior has lots of natural sunlight and False otherwise"
    )


class RealEstateListingSource(pz.UserSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)

    def getSize(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob("*"))

    def getItem(self, idx: int):
        # fetch listing
        listing = self.listings[idx]

        # create data record
        dr = pz.DataRecord(self.schema, scan_idx=idx)
        dr.listing = listing
        dr.image_contents = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            bytes_data = None
            with open(os.path.join(listing_dir, file), "rb") as f:
                bytes_data = f.read()
            if file.endswith(".txt"):
                dr.text_content = bytes_data.decode("utf-8")
            elif file.endswith(".png"):
                dr.image_contents.append(bytes_data)

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
        "--engine",
        type=str,
        help="The engine to use. One of sentinel, nosentinel",
        default="sentinel",
    )
    parser.add_argument(
        "--executor",
        type=str,
        help="The plan executor to use. One of sequential, pipelined, parallel",
        default="parallel",
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
    policy = pz.MaxQuality()
    if args.policy == "mincost":
        policy = pz.MinCost()
    elif args.policy == "mintime":
        policy = pz.MinTime()
    elif args.policy == "maxquality":
        policy = pz.MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    execution_engine = None
    engine, executor = args.engine, args.executor
    if engine == "sentinel":
        if executor == "sequential":
            execution_engine = pz.SequentialSingleThreadSentinelExecution
        elif executor == "pipelined":
            execution_engine = pz.PipelinedSingleThreadSentinelExecution
        elif executor == "parallel":
            execution_engine = pz.PipelinedParallelSentinelExecution
        else:
            print("Unknown executor")
            exit(1)
    elif engine == "nosentinel":
        if executor == "sequential":
            execution_engine = pz.SequentialSingleThreadNoSentinelExecution
        elif executor == "pipelined":
            execution_engine = pz.PipelinedSingleThreadNoSentinelExecution
        elif executor == "parallel":
            execution_engine = pz.PipelinedParallelNoSentinelExecution
        else:
            print("Unknown executor")
            exit(1)
    else:
        print("Unknown engine")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    if workload == "enron":
        # datasetid="enron-eval" for paper evaluation
        plan = pz.Dataset(datasetid, schema=Email)
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
        pz.DataDirectory().registerUserSource(
            src=RealEstateListingSource(user_dataset_id, data_filepath),
            dataset_id=user_dataset_id,
        )
        plan = pz.Dataset(user_dataset_id, schema=RealEstateListingFiles)
        plan = plan.convert(TextRealEstateListing, depends_on="text_content")
        plan = plan.convert(ImageRealEstateListing, image_conversion=True, depends_on="image_contents")
        plan = plan.filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    elif workload == "medical-schema-matching":
        # datasetid="biofabric-medium" for paper evaluation
        plan = pz.Dataset(datasetid, schema=pz.XLSFile)
        plan = plan.convert(pz.Table, udf=xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        plan = plan.filter("The rows of the table contain the patient age")
        plan = plan.convert(CaseData, desc="The patient data in the table", cardinality=pz.Cardinality.ONE_TO_MANY)

    # execute pz plan
    records, execution_stats = pz.Execute(
        plan,
        policy,
        nocache=True,
        optimization_strategy=pz.OptimizationStrategy.OPTIMAL,
        execution_engine=execution_engine,
        verbose=verbose,
    )

    # save statistics
    if profile:
        stats_path = f"profiling-data/{workload}-profiling.json"
        execution_stats_dict = execution_stats.to_json()
        with open(stats_path, "w") as f:
            json.dump(execution_stats_dict, f)

    # visualize output in Gradio
    if visualize:
        from palimpzest.utils.demo_helpers import printTable

        plan_str = list(execution_stats.plan_strs.values())[-1]
        if workload == "enron":
            printTable(records, cols=["sender", "subject"], plan_str=plan_str)

        elif workload == "real-estate":
            fst_imgs, snd_imgs, thrd_imgs, addrs, prices = [], [], [], [], []
            for record in records:
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

                plan_str = list(execution_stats.plan_strs.values())[0]
                gr.Textbox(value=plan_str, info="Query Plan")

            demo.launch()
