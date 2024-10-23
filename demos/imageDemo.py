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
    from palimpzest.utils import load_env

    load_env()


class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required=True)


def buildImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filter("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc="Images of dogs")
    return dogImages


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached results")

    args = parser.parse_args()
    no_cache = args.no_cache
    datasetid = "images-tiny"
    if datasetid not in pz.DataDirectory().listRegisteredDatasets():
        pz.DataDirectory().registerLocalDirectory(path="testdata/images-tiny", dataset_id="images-tiny")

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    print("Starting image task")
    policy = pz.MaxQuality()
    plan = buildImagePlan(datasetid)
    engine = pz.PipelinedParallelNoSentinelExecution
    records, execution_stats = pz.Execute(plan, policy=policy, nocache=no_cache, execution_engine=engine, verbose=True)

    print("Obtained records", records)
    imgs, breeds = [], []
    for record in records:
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

        plan_str = list(execution_stats.plan_strs.values())[0]
        gr.Textbox(value=plan_str, info="Query Plan")

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
    demo.launch()
