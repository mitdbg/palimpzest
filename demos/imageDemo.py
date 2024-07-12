#!/usr/bin/env python3
""" This scripts is a demo for image processing, it is simply an abridged version of simpleDemo.py """
import context
from palimpzest.constants import PZ_DIR
import palimpzest as pz

from tabulate import tabulate
from PIL import Image

import gradio as gr
import numpy as np
import pandas as pd

import argparse
import requests
import json
import time
import os

class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required = True)

def buildImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filter("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc = "Images of dogs")
    return dogImages

if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached results')

    args = parser.parse_args()
    no_cache = args.no_cache
    if no_cache:
        print("WARNING: Removing cache, this will result in API calls")
        cachedir = PZ_DIR + "/data/cache/"
        for f in os.listdir(cachedir):
            if f.endswith(".cached"):
                os.remove(os.path.join(cachedir, f))

    verbose = True
    datasetid = "images-tiny"
    if not datasetid in pz.DataDirectory().listRegisteredDatasets():
        pz.DataDirectory().registerLocalDirectory(
            path="testdata/images-tiny", dataset_id="images-tiny")

    if os.getenv('OPENAI_API_KEY') is None and os.getenv('TOGETHER_API_KEY') is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    print("Starting image task")
    policy = pz.UserChoice()
    rootSet = buildImagePlan(datasetid)
    engine = pz.PipelinedParallelExecution
    records, plan, stats = pz.Execute(rootSet, 
                                    policy = policy,
                                    nocache=True,
                                    execution_engine=engine)

    print("Obtained records", records)
    print(stats)
    imgs, breeds = [], []
    for record in records:
        print("Trying to open ", record.filename)
        path = os.path.join("testdata/images-tiny/", record.filename)
        img = Image.open(path).resize((128,128))
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

        gr.Textbox(value=str(plan), info="Query Plan")

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
    demo.launch()
