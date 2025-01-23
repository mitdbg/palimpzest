import palimpzest as pz

from palimpzest.constants import Cardinality
from palimpzest.elements import DataRecord, GroupBySig

from bs4 import BeautifulSoup
from PIL import Image
from requests_html import HTMLSession # for downloading JavaScript content
from tabulate import tabulate

import gradio as gr
import numpy as np
import pandas as pd

import argparse
import csv
import datetime
import json
import os
import requests
import sys
import time


def printTable(records, cols=None, gradio=False, plan_str=None):
    records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith("_")
        }
        for record in records
    ]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols
    final_df = records_df[print_cols] if not records_df.empty else pd.DataFrame(columns=print_cols)

    if not gradio:
        
        print(tabulate(final_df, headers="keys", tablefmt="psql"))

    else:
        with gr.Blocks() as demo:
            gr.Dataframe(final_df)

            if plan_str is not None:
                gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()
        