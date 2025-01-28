#!/usr/bin/env python3
"""This scripts is a demo for image processing, it is simply an abridged version of simpleDemo.py"""

import argparse
import os
import time

import gradio as gr
import numpy as np
from PIL import Image

from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import ImageFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class DogImage(ImageFile):
    breed = Field(desc="The breed of the dog")


def build_image_plan(dataset_id):
    images = Dataset(dataset_id, schema=ImageFile)
    filtered_images = images.filter("The image contains one or more dogs")
    dog_images = filtered_images.convert(DogImage, desc="Images of dogs")
    return dog_images


if __name__ == "__main__":
    # parse arguments
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached results", default=True)

    args = parser.parse_args()
    no_cache = args.no_cache
    datasetid = "images-tiny"
    if datasetid not in DataDirectory().list_registered_datasets():
        DataDirectory().register_local_directory(path="testdata/images-tiny", dataset_id="images-tiny")

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    print("Starting image task")
    policy = MaxQuality()
    plan = build_image_plan(datasetid)
    config = QueryProcessorConfig(
        policy=policy,
        nocache=no_cache,
        verbose=True,
        processing_strategy="no_sentinel"
    )
    data_record_collection = plan.run(config)

    print("Obtained records", data_record_collection.data_records)
    imgs, breeds = [], []
    for record in data_record_collection:
        print("Trying to open ", record.filename)
        path = os.path.join("testdata/images-tiny/", record.filename)
        img = Image.open(path).resize((128, 128))
        img_arr = np.asarray(img)
        imgs.append(img_arr)
        breeds.append(record.breed)

    with gr.Blocks() as demo:
        img_blocks, breed_blocks = [], []
        for img, breed in zip(imgs, breeds):
            with gr.Row():
                with gr.Column():
                    img_blocks.append(gr.Image(value=img))
                with gr.Column():
                    breed_blocks.append(gr.Textbox(value=breed))

        plan_str = list(data_record_collection.execution_stats.plan_strs.values())[0]
        gr.Textbox(value=plan_str, info="Query Plan")

    end_time = time.time()
    print("Elapsed time:", end_time - start_time)
    demo.launch()
