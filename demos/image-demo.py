#!/usr/bin/env python3
"""This scripts is a demo for image processing, it is simply an abridged version of simpleDemo.py"""

import argparse
import os
import time

import gradio as gr
import numpy as np
from PIL import Image

import palimpzest as pz

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()

dog_image_cols = [
    {"name": "breed", "type": str, "desc": "The breed of the dog"},
]

def build_image_plan(dataset):
    images = pz.Dataset(dataset)
    filtered_images = images.sem_filter("The image contains one or more dogs")
    dog_images = filtered_images.sem_add_columns(dog_image_cols)
    return dog_images


if __name__ == "__main__":
    # parse arguments
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--cache", action="store_true", help="Use cached results",default=False)

    args = parser.parse_args()
    cache = args.cache
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    print("Starting image task")
    policy = pz.MaxQuality()
    plan = build_image_plan("testdata/images-tiny")
    config = pz.QueryProcessorConfig(
        policy=policy,
        cache=cache,
        verbose=True,
        processing_strategy="no_sentinel",
    )
    data_record_collection = plan.run(config)

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
