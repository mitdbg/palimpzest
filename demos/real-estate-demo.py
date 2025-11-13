import argparse
import os

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

import palimpzest as pz
from palimpzest.core.lib.schemas import ImageFilepath


def print_table(records, cols=None, plan_str=None):
    """Helper function to print execution results using Gradio"""
    if len(records) == 0:
        print("No records met search criteria")
        return

    records = [record.to_dict() for record in records]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols

    with gr.Blocks() as demo:
        gr.Dataframe(records_df[print_cols])

        if plan_str is not None:
            gr.Textbox(value=plan_str, info="Physical Plan")

    demo.launch()


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

real_estate_listing_cols = [
    {"name": "listing", "type": str, "desc": "The name of the listing"},
    {"name": "text_content", "type": str, "desc": "The content of the listing's text description"},
    {"name": "image_filepaths", "type": list[ImageFilepath], "desc": "A list of the filepaths for each image of the listing"},
]

real_estate_text_cols = [
    {"name": "address", "type": str, "desc": "The address of the property"},
    {"name": "price", "type": int | float, "desc": "The listed price of the property"},
]

real_estate_image_cols = [
    {"name": "is_modern_and_attractive", "type": bool, "desc": "True if the home interior design is modern and attractive and False otherwise"},
    {"name": "has_natural_sunlight", "type": bool, "desc": "True if the home interior has lots of natural sunlight and False otherwise"},
]

# class RealEstateValidator(pz.Validator):
#     def __init__(self, labels_file: str):
#         super().__init__()
#         with open(labels_file) as f:
#             self.filename_to_labels = json.load(f)

#     def filter_score_fn(self, filter_str: str, input_record: dict, output: bool) -> float | None:
#         filename = input_record["filename"]
#         labels = self.filename_to_labels[filename]
#         if labels is None:
#             return None

#         if "business transactions" in filter_str:
#             return float(labels["mentions_transaction"] == output)
#         elif "first-hand discussion" in filter_str:
#             return float(labels["firsthand_discussion"] == output)
#         else:
#             return None

#     def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
#         # NOTE: we score the map based on the sender and subject fields only, as summary is too subjective;
#         #       we could also use an LLM judge within this function to score the summary field if desired
#         filename = input_record["filename"]
#         labels = self.filename_to_labels[filename]
#         if labels is None:
#             return None

#         return (float(labels["sender"] == output["sender"]) + float(labels["subject"] == output["subject"])) / 2.0


class RealEstateListingDataset(pz.IterDataset):
    def __init__(self, listings_dir):
        super().__init__(id="real-estate", schema=real_estate_listing_cols)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))
        self.listings = [file for file in self.listings if not file.startswith(".")]

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
    parser.add_argument("--dataset", type=str, help="The path to the dataset")
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default="maxquality",
    )

    args = parser.parse_args()

    # The user has to indicate the dataset id and the workload
    if args.dataset is None:
        print("Please provide a dataset id")
        exit(1)

    dataset = args.dataset
    visualize = args.viz
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

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None and os.getenv("ANTHROPIC_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY, TOGETHER_API_KEY, and ANTHROPIC_API_KEY are unset")

    # create pz plan
    plan = RealEstateListingDataset(dataset)
    plan = plan.sem_map(real_estate_text_cols, depends_on="text_content")
    plan = plan.sem_map(real_estate_image_cols, depends_on="image_filepaths")
    plan = plan.sem_filter(
        "The interior is modern and attractive, and has lots of natural sunlight",
        depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
    )
    plan = plan.filter(within_two_miles_of_mit, depends_on="address")
    plan = plan.filter(in_price_range, depends_on="price")

    # construct config and run plan
    config = pz.QueryProcessorConfig(
        policy=policy,
        available_models=[pz.Model.GPT_5_MINI],
        k=6,
        j=6,
        sample_budget=125,
    )
    data_record_collection = plan.optimize_and_run(config, validator=pz.Validator(model=pz.Model.o4_MINI))
    print(data_record_collection.to_df())

    # preds = data_record_collection.to_df()["listing"].tolist()
    # gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval-100.csv")
    # labels = gt_df.listing.tolist()
    # tp, fp, fn = 0, 0, 0
    # for pred in preds:
    #     if pred in labels:
    #         tp += 1
    #     else:
    #         fp += 1
    # for label in labels:
    #     if label not in preds:
    #         fn += 1
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1: {f1:.4f}")

    # visualize output in Gradio
    if visualize:
        plan_str = list(data_record_collection.execution_stats.plan_strs.values())[-1]
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
