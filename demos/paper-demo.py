import argparse
import json
import os

import gradio as gr
import numpy as np
from PIL import Image

import palimpzest as pz
from palimpzest.core.lib.fields import ImageFilepathField, ListField
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

email_cols =  [
    {"name": "sender", "type": str, "desc": "The email address of the sender"},
    {"name": "subject", "type": str, "desc": "The subject of the email"},
]

case_data_cols = [
    {"name": "case_submitter_id", "type": str, "desc": "The ID of the case"},
    {"name": "age_at_diagnosis", "type": int | float, "desc": "The age of the patient at the time of diagnosis"},
    {"name": "race", "type": str, "desc": "An arbitrary classification of a taxonomic group that is a division of a species."},
    {"name": "ethnicity", "type": str, "desc": "Whether an individual describes themselves as Hispanic or Latino or not."},
    {"name": "gender", "type": str, "desc": "Text designations that identify gender."},
    {"name": "vital_status", "type": str, "desc": "The vital status of the patient"},
    {"name": "ajcc_pathologic_t", "type": str, "desc": "Code of pathological T (primary tumor) to define the size or contiguous extension of the primary tumor (T), using staging criteria from the American Joint Committee on Cancer (AJCC)."},
    {"name": "ajcc_pathologic_n", "type": str, "desc": "The codes that represent the stage of cancer based on the nodes present (N stage) according to criteria based on multiple editions of the AJCC's Cancer Staging Manual."},
    {"name": "ajcc_pathologic_stage", "type": str, "desc": "The extent of a cancer, especially whether the disease has spread from the original site to other parts of the body based on AJCC staging criteria."},
    {"name": "tumor_grade", "type": int | float, "desc": "Numeric value to express the degree of abnormality of cancer cells, a measure of differentiation and aggressiveness."},
    {"name": "tumor_focality", "type": str, "desc": "The text term used to describe whether the patient's disease originated in a single location or multiple locations."},
    {"name": "tumor_largest_dimension_diameter", "type": int | float, "desc": "The tumor largest dimension diameter."},
    {"name": "primary_diagnosis", "type": str, "desc": "Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "morphology", "type": str, "desc": "The Morphological code of the tumor, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "tissue_or_organ_of_origin", "type": str, "desc": "The text term used to describe the anatomic site of origin, of the patient's malignant disease, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "study", "type": str, "desc": "The last name of the author of the study, from the table name"},
    {"name": "filename", "type": str, "desc": "The name of the file the record was extracted from"}
]

real_estate_listing_cols = [
    {"name": "listing", "type": str, "desc": "The name of the listing"},
    {"name": "text_content", "type": str, "desc": "The content of the listing's text description"},
    {"name": "image_filepaths", "type": ListField(ImageFilepathField), "desc": "A list of the filepaths for each image of the listing"},
]

real_estate_text_cols = [
    {"name": "address", "type": str, "desc": "The address of the property"},
    {"name": "price", "type": int | float, "desc": "The listed price of the property"},
]

real_estate_image_cols = [
    {"name": "is_modern_and_attractive", "type": bool, "desc": "True if the home interior design is modern and attractive and False otherwise"},
    {"name": "has_natural_sunlight", "type": bool, "desc": "True if the home interior has lots of natural sunlight and False otherwise"},
]

table_cols = [
    {"name": "rows", "type": list[str], "desc": "The rows of the table"},
    {"name": "header", "type": list[str], "desc": "The header of the table"},
    {"name": "name", "type": str, "desc": "The name of the table"},
    {"name": "filename", "type": str, "desc": "The name of the file the table was extracted from"}
]


# class RealEstateListingFiles(Schema):
#     """The source text and image data for a real estate listing."""

#     listing = StringField(desc="The name of the listing")
#     text_content = StringField(desc="The content of the listing's text description")
#     image_filepaths = ListField(
#         element_type=ImageFilepathField,
#         desc="A list of the filepaths for each image of the listing",
#     )

class RealEstateListingReader(pz.DataReader):
    def __init__(self, listings_dir):
        super().__init__(real_estate_listing_cols)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)

    def __getitem__(self, idx: int):
        # get listing
        listing = self.listings[idx]

        # get fields
        image_filepaths, text_content = [], None
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                image_filepaths.append(os.path.join(listing_dir, file))

        # construct and return dictionary with fields
        return {"listing": listing, "text_content": text_content, "image_filepaths": image_filepaths}


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--viz", default=False, action="store_true", help="Visualize output in Gradio")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument("--dataset", type=str, help="The path to the dataset")
    parser.add_argument(
        "--workload", type=str, help="The workload to run. One of enron, real-estate, medical-schema-matching."
    )
    parser.add_argument(
        "--executor",
        type=str,
        help="The plan executor to use. One of sequential, pipelined, parallel",
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
    if args.dataset is None:
        print("Please provide a dataset id")
        exit(1)
    if args.workload is None:
        print("Please provide a workload")
        exit(1)

    # create directory for profiling data
    if args.profile:
        os.makedirs("profiling-data", exist_ok=True)

    dataset = args.dataset
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

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    if workload == "enron":
        plan = pz.Dataset(dataset).sem_add_columns(email_cols)
        plan = plan.sem_filter(
            "The email is not quoting from a news article or an article written by someone outside of Enron"
        )
        plan = plan.sem_filter(
            'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
        )

    elif workload == "real-estate":
        plan = pz.Dataset(RealEstateListingReader(dataset))
        plan = plan.sem_add_columns(real_estate_text_cols, depends_on="text_content")
        plan = plan.sem_add_columns(real_estate_image_cols, depends_on="image_filepaths")
        plan = plan.sem_filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    elif workload == "medical-schema-matching":
        plan = pz.Dataset(dataset)
        plan = plan.add_columns(xls_to_tables, cols=table_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        plan = plan.sem_filter("The rows of the table contain the patient age")
        plan = plan.sem_add_columns(case_data_cols, cardinality=pz.Cardinality.ONE_TO_MANY)

    # construct config and run plan
    config = pz.QueryProcessorConfig(
        cache=False,
        verbose=verbose,
        policy=policy,
        execution_strategy=args.executor,
    )
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
                    path = os.path.join(dataset, record.listing, img_name)
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
