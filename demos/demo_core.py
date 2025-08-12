#!/usr/bin/env python3
import json
import os

import pandas as pd
from tabulate import tabulate

import palimpzest as pz
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.elements.records import DataRecord

sci_paper_cols = [
    {"name": "title", "type": str, "desc": "The title of the paper. This is a natural language title, not a number or letter."},
    {"name": "publication_year", "type": int, "desc": "The year the paper was published. This is a number."},
    {"name": "author", "type": str, "desc": "The name of the first author of the paper"},
    {"name": "institution", "type": str, "desc": "The institution of the first author of the paper"},
    {"name": "journal", "type": str, "desc": "The name of the journal the paper was published in"},
    {"name": "funding_agency", "type": str, "desc": "The name of the funding agency that supported the research"},
]

email_cols = [
    {"name": "sender", "type": str, "desc": "The email address of the sender"},
    {"name": "subject", "type": str, "desc": "The subject of the email"},
]

dog_image_cols = [
    {"name": "breed", "type": str, "desc": "The breed of the dog"},
]

def build_sci_paper_plan(dataset):
    """A dataset-independent declarative description of authors of good papers"""
    return pz.PDFFileDataset(id="science-papers", path=dataset).sem_add_columns(sci_paper_cols)

def build_test_pdf_plan(dataset):
    """This tests whether we can process a PDF file"""
    return pz.PDFFileDataset(id="pdf-files", path=dataset)

def build_mit_battery_paper_plan(dataset):
    """A dataset-independent declarative description of authors of good papers"""
    sci_papers = pz.PDFFileDataset(id="science-papers", path=dataset).sem_add_columns(sci_paper_cols)
    battery_papers = sci_papers.sem_filter("The paper is about batteries")
    mit_papers = battery_papers.sem_filter("The paper is from MIT")
    return mit_papers

def build_enron_plan(dataset):
    """Build a plan for processing Enron email data"""
    return pz.TextFileDataset(id="enron-emails", path=dataset).sem_add_columns(email_cols)

def compute_enron_stats(dataset):
    """Compute statistics on Enron email data"""
    emails = pz.TextFileDataset(id="enron-emails", path=dataset).sem_add_columns(email_cols)
    subject_line_lengths = emails.sem_add_columns([{"name": "words", "type": int, "desc": "The number of words in the subject field"}])
    return subject_line_lengths

def enron_gby_plan(dataset):
    """Group Enron emails by sender"""
    emails = pz.TextFileDataset(id="enron-emails", path=dataset).sem_add_columns(email_cols)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = ["sender"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_emails = emails.groupby(gby_desc)
    return grouped_emails

def enron_count_plan(dataset):
    """Count total Enron emails"""
    emails = pz.TextFileDataset(id="enron-emails", path=dataset).sem_add_columns(email_cols)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = []
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    count_emails = emails.groupby(gby_desc)
    return count_emails

def enron_average_count_plan(dataset):
    """Calculate average number of emails per sender"""
    emails = pz.TextFileDataset(id="enron-emails", path=dataset).sem_add_columns(email_cols)
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

def enron_limit_plan(dataset, limit=5):
    """Get limited number of Enron emails"""
    emails = pz.TextFileDataset(id="enron-emails", path=dataset).sem_add_columns(email_cols)
    limit_data = emails.limit(limit)
    return limit_data

def build_image_plan(dataset):
    """Build a plan for processing dog images"""
    images = pz.ImageFileDataset(id="dog-images", path=dataset)
    filtered_images = images.sem_filter("The image contains one or more dogs")
    dog_images = filtered_images.sem_add_columns(dog_image_cols)
    return dog_images

def build_image_agg_plan(dataset):
    """Build a plan for aggregating dog images by breed"""
    images = pz.ImageFileDataset(id="dog-images", path=dataset)
    filtered_images = images.sem_filter("The image contains one or more dogs")
    dog_images = filtered_images.sem_add_columns(dog_image_cols)
    ops = ["count"]
    fields = ["breed"]
    groupbyfields = ["breed"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_dog_images = dog_images.groupby(gby_desc)
    return grouped_dog_images

def build_join_plan(dataset1, dataset2):
    """Build a plan that joins two datasets"""
    ds1 = pz.TextFileDataset(id="enron-emails", path=dataset1).sem_add_columns(email_cols)
    ds2 = pz.TextFileDataset(id="other-enron-emails", path=dataset2).sem_add_columns(email_cols)
    joined = ds1.sem_join(ds2, condition="sender")
    return joined

def build_join_image_plan(dataset1, dataset2):
    """Build a plan that joins two datasets with images"""
    ds1 = pz.ImageFileDataset(id="dog-images", path=dataset1).sem_add_columns(dog_image_cols)
    ds2 = pz.ImageFileDataset(id="other-dog-images", path=dataset2).sem_add_columns(dog_image_cols)
    joined = ds1.sem_join(ds2, condition="breed")
    return joined

def get_task_config(task, dataset, join_dataset=None):
    """Get configuration for a specific task"""
    if task == "paper":
        root_set = build_mit_battery_paper_plan(dataset)
        cols = ["title", "publication_year", "author", "institution", "journal", "funding_agency"]
        stat_path = "profiling-data/paper-profiling.json"
    elif task == "enron":
        root_set = build_enron_plan(dataset)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/enron-profiling.json"
    elif task == "enronGby":
        root_set = enron_gby_plan(dataset)
        cols = ["sender", "count(sender)"]
        stat_path = "profiling-data/egby-profiling.json"
    elif task in ("enronCount", "count"):
        root_set = enron_count_plan(dataset)
        cols = ["count(sender)"]
        stat_path = "profiling-data/ecount-profiling.json"
    elif task in ("enronAvgCount", "average"):
        root_set = enron_average_count_plan(dataset)
        cols = ["average(count(sender))"]
        stat_path = "profiling-data/e-profiling.json"
    elif task == "enronmap":
        root_set = compute_enron_stats(dataset)
        cols = ["sender", "subject", "value"]
        stat_path = "profiling-data/emap-profiling.json"
    elif task == "pdftest":
        root_set = build_test_pdf_plan(dataset)
        cols = ["filename"]
        stat_path = "profiling-data/pdftest-profiling.json"
    elif task == "scitest":
        root_set = build_sci_paper_plan(dataset)
        cols = ["title", "author", "institution", "journal", "funding_agency"]
        stat_path = "profiling-data/scitest-profiling.json"
    elif task == "image":
        root_set = build_image_plan(dataset)
        cols = None
        stat_path = "profiling-data/image-profiling.json"
    elif task == "gbyImage":
        root_set = build_image_agg_plan(dataset)
        cols = ["breed", "count(breed)"]
        stat_path = "profiling-data/gbyImage-profiling.json"
    elif task == "limit":
        root_set = enron_limit_plan(dataset, 5)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/limit-profiling.json"
    elif task == "join":
        root_set = build_join_plan(dataset, join_dataset)
        cols = ["filename", "sender", "subject"]
        stat_path = "profiling-data/join-profiling.json"
    elif task == "joinImage":
        root_set = build_join_image_plan(dataset, join_dataset)
        cols = None
        stat_path = "profiling-data/joinImage-profiling.json"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return root_set, cols, stat_path

def execute_task(task, dataset, policy, join_dataset=None, verbose=False, profile=False, execution_strategy="sequential", optimizer_strategy="pareto"):
    """Execute a task and return results"""
    root_set, cols, stat_path = get_task_config(task, dataset, join_dataset)
    config = pz.QueryProcessorConfig(
        policy=policy,
        verbose=verbose,
        execution_strategy=execution_strategy,
        optimizer_strategy=optimizer_strategy,
    )
    data_record_collection = root_set.run(config)

    if profile:
        os.makedirs("profiling-data", exist_ok=True)
        with open(stat_path, "w") as f:
            json.dump(data_record_collection.execution_stats.to_json(), f)

    return data_record_collection.data_records, data_record_collection.execution_stats, cols

def format_results_table(records: list[DataRecord], cols=None):
    """Format records as a table"""
    records = [record.to_dict(include_bytes=False) for record in records]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols
    final_df = records_df[print_cols] if not records_df.empty else pd.DataFrame(columns=print_cols)
    return tabulate(final_df, headers="keys", tablefmt="psql")
