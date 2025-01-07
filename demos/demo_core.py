#!/usr/bin/env python3
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from tabulate import tabulate

import palimpzest as pz
from palimpzest.constants import Cardinality
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.elements.records import DataRecord

class ScientificPaper(pz.core.PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""

    title = pz.core.Field(
        desc="The title of the paper. This is a natural language title, not a number or letter.",
        required=True,
    )
    publicationYear = pz.core.Field(desc="The year the paper was published. This is a number.", required=False)
    author = pz.core.Field(desc="The name of the first author of the paper", required=True)
    institution = pz.core.Field(desc="The institution of the first author of the paper", required=True)
    journal = pz.core.Field(desc="The name of the journal the paper was published in", required=True)
    fundingAgency = pz.core.Field(
        desc="The name of the funding agency that supported the research",
        required=False,
    )

class Email(pz.core.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.core.Field(desc="The email address of the sender", required=True)
    subject = pz.core.Field(desc="The subject of the email", required=True)

class DogImage(pz.core.ImageFile):
    breed = pz.core.Field(desc="The breed of the dog", required=True)

def build_sci_paper_plan(dataset_id):
    """A dataset-independent declarative description of authors of good papers"""
    return pz.Dataset(dataset_id, schema=ScientificPaper)

def build_test_pdf_plan(dataset_id):
    """This tests whether we can process a PDF file"""
    return pz.Dataset(dataset_id, schema=pz.PDFFile)

def build_mit_battery_paper_plan(dataset_id):
    """A dataset-independent declarative description of authors of good papers"""
    sci_papers = pz.Dataset(dataset_id, schema=ScientificPaper)
    battery_papers = sci_papers.filter("The paper is about batteries")
    mit_papers = battery_papers.filter("The paper is from MIT")
    return mit_papers

def build_enron_plan(dataset_id):
    """Build a plan for processing Enron email data"""
    from palimpzest.sets import Dataset
    emails = Dataset(dataset_id, schema=Email)
    return emails

def compute_enron_stats(dataset_id):
    """Compute statistics on Enron email data"""
    from palimpzest.sets import Dataset
    emails = Dataset(dataset_id, schema=Email)
    subject_line_lengths = emails.convert(pz.Number, desc="The number of words in the subject field")
    return subject_line_lengths

def enron_gby_plan(dataset_id):
    """Group Enron emails by sender"""
    from palimpzest.sets import Dataset
    emails = Dataset(dataset_id, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = ["sender"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_emails = emails.groupby(gby_desc)
    return grouped_emails

def enron_count_plan(dataset_id):
    """Count total Enron emails"""
    from palimpzest.sets import Dataset
    emails = Dataset(dataset_id, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = []
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    count_emails = emails.groupby(gby_desc)
    return count_emails

def enron_average_count_plan(dataset_id):
    """Calculate average number of emails per sender"""
    from palimpzest.sets import Dataset
    emails = Dataset(dataset_id, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = ["sender"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_emails = emails.groupby(gby_desc)
    ops = ["average"]
    fields = ["count(sender)"]
    groupbyfields = []
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    average_emails_per_sender = grouped_emails.groupby(gby_desc)
    return average_emails_per_sender

def enron_limit_plan(dataset_id, limit=5):
    """Get limited number of Enron emails"""
    from palimpzest.sets import Dataset
    data = Dataset(dataset_id, schema=Email)
    limit_data = data.limit(limit)
    return limit_data

def build_image_plan(dataset_id):
    """Build a plan for processing dog images"""
    from palimpzest.sets import Dataset
    images = Dataset(dataset_id, schema=pz.ImageFile)
    filtered_images = images.filter("The image contains one or more dogs")
    dog_images = filtered_images.convert(DogImage, desc="Images of dogs")
    return dog_images

def build_image_agg_plan(dataset_id):
    """Build a plan for aggregating dog images by breed"""
    from palimpzest.sets import Dataset
    images = Dataset(dataset_id, schema=pz.ImageFile)
    filtered_images = images.filter("The image contains one or more dogs")
    dog_images = filtered_images.convert(DogImage, desc="Images of dogs")
    ops = ["count"]
    fields = ["breed"]
    groupbyfields = ["breed"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_dog_images = dog_images.groupby(gby_desc)
    return grouped_dog_images

def get_task_config(task, datasetid):
    """Get configuration for a specific task"""
    if task == "paper":
        root_set = build_mit_battery_paper_plan(datasetid)
        cols = ["title", "publicationYear", "author", "institution", "journal", "fundingAgency"]
        stat_path = "profiling-data/paper-profiling.json"
    elif task == "enron":
        root_set = build_enron_plan(datasetid)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/enron-profiling.json"
    elif task == "enronGby":
        root_set = enron_gby_plan(datasetid)
        cols = ["sender", "count(sender)"]
        stat_path = "profiling-data/egby-profiling.json"
    elif task in ("enronCount", "count"):
        root_set = enron_count_plan(datasetid)
        cols = ["count(sender)"]
        stat_path = "profiling-data/ecount-profiling.json"
    elif task in ("enronAvgCount", "average"):
        root_set = enron_average_count_plan(datasetid)
        cols = ["average(count(sender))"]
        stat_path = "profiling-data/e-profiling.json"
    elif task == "enronmap":
        root_set = compute_enron_stats(datasetid)
        cols = ["sender", "subject", "value"]
        stat_path = "profiling-data/emap-profiling.json"
    elif task == "pdftest":
        root_set = build_test_pdf_plan(datasetid)
        cols = ["filename"]
        stat_path = "profiling-data/pdftest-profiling.json"
    elif task == "scitest":
        root_set = build_sci_paper_plan(datasetid)
        cols = ["title", "author", "institution", "journal", "fundingAgency"]
        stat_path = "profiling-data/scitest-profiling.json"
    elif task == "image":
        root_set = build_image_plan(datasetid)
        cols = None
        stat_path = "profiling-data/image-profiling.json"
    elif task == "gbyImage":
        root_set = build_image_agg_plan(datasetid)
        cols = ["breed", "count(breed)"]
        stat_path = "profiling-data/gbyImage-profiling.json"
    elif task == "limit":
        root_set = enron_limit_plan(datasetid, 5)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/limit-profiling.json"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return root_set, cols, stat_path

def execute_task(task, datasetid, execution_engine, policy, verbose=False, profile=False):
    """Execute a task and return results"""
    root_set, cols, stat_path = get_task_config(task, datasetid)
    from palimpzest.query import Execute

    records, execution_stats = Execute(
        root_set,
        policy=policy,
        nocache=True,
        allow_token_reduction=False,
        allow_code_synth=False,
        execution_engine=execution_engine,
        verbose=verbose,
    )

    if profile:
        os.makedirs("profiling-data", exist_ok=True)
        with open(stat_path, "w") as f:
            json.dump(execution_stats.to_json(), f)

    return records, execution_stats, cols

def format_results_table(records, cols=None):
    """Format records as a table"""
    records = [{key: record[key] for key in record.get_fields()} for record in records]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols
    final_df = records_df[print_cols] if not records_df.empty else pd.DataFrame(columns=print_cols)
    return tabulate(final_df, headers="keys", tablefmt="psql") 