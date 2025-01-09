#!/usr/bin/env python3
import argparse
import csv
import datetime
import json
import os
import sys
import time

import gradio as gr
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from palimpzest.constants import Cardinality
from palimpzest.corelib.fields import Field
from palimpzest.corelib.schemas import (
    URL,
    Download,
    ImageFile,
    Number,
    PDFFile,
    RawJSONObject,
    Schema,
    TextFile,
    WebPage,
)
from palimpzest.datamanager import DataDirectory
from palimpzest.datasources import UserSource
from palimpzest.elements.groupbysig import GroupBySig
from palimpzest.elements.records import DataRecord
from palimpzest.execution.execute import Execute
from palimpzest.execution.nosentinel_execution import (
    NoSentinelPipelinedParallelExecution,
    NoSentinelPipelinedSingleThreadExecution,
    NoSentinelSequentialSingleThreadExecution,
)
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.sets import Dataset
from PIL import Image
from requests_html import HTMLSession  # for downloading JavaScript content
from tabulate import tabulate


class ScientificPaper(PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""
    title = Field(desc="The title of the paper. This is a natural language title, not a number or letter.")
    publicationYear = Field(desc="The year the paper was published. This is a number.") # noqa
    author = Field(desc="The name of the first author of the paper")
    institution = Field(desc="The institution of the first author of the paper")
    journal = Field(desc="The name of the journal the paper was published in")
    fundingAgency = Field( # noqa
        desc="The name of the funding agency that supported the research",
    )


def build_sci_paper_plan(dataset_id):
    """A dataset-independent declarative description of authors of good papers"""
    return Dataset(dataset_id, schema=ScientificPaper)


def build_test_pdf_plan(dataset_id):
    """This tests whether we can process a PDF file"""
    return Dataset(dataset_id, schema=PDFFile)


def build_mit_battery_paper_plan(dataset_id):
    """A dataset-independent declarative description of authors of good papers"""
    sci_papers = Dataset(dataset_id, schema=ScientificPaper)
    battery_papers = sci_papers.filter("The paper is about batteries")
    mit_papers = battery_papers.filter("The paper is from MIT")

    return mit_papers


class VLDBPaperListing(Schema):
    """VLDBPaperListing represents a single paper from the VLDB conference"""
    title = Field(desc="The title of the paper")
    authors = Field(desc="The authors of the paper")
    pdfLink = Field(desc="The link to the PDF of the paper") # noqa


def vldb_text_file_to_url(candidate: dict):
    url_records = []
    with open(candidate["filename"]) as f:
        for line in f:
            record = {"url": line.strip()}
            url_records.append(record)

    return url_records


def html_to_text_with_links(html):
    # Parse the HTML content
    soup = BeautifulSoup(html, "html.parser")

    # Find all hyperlink tags
    for a in soup.find_all("a"):
        # Check if the hyperlink tag has an 'href' attribute
        if a.has_attr("href"):
            # Replace the hyperlink with its text and URL in parentheses
            a.replace_with(f"{a.text} ({a['href']})")

    # Extract text from the modified HTML
    text = soup.get_text(separator="\n", strip=True)
    return text


def get_page_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    }

    session = HTMLSession()
    response = session.get(url, headers=headers)
    return response.text


def download_html(candidate: dict):
    textcontent = get_page_text(candidate["url"])

    record = {}
    html = textcontent
    tokens = html.split()[:5000]
    record["html"] = " ".join(tokens)

    stripped_html = html_to_text_with_links(textcontent)
    tokens = stripped_html.split()[:5000]
    record["text"] = " ".join(tokens)

    # get current timestamp, in nice ISO format
    record["timestamp"] = datetime.datetime.now().isoformat()
    return record


def download_pdf(candidate: dict):
    print(f"DOWNLOADING: {candidate['pdfLink']}")
    content = requests.get(candidate["pdfLink"]).content
    record["url"] = candidate["pdfLink"]
    record["content"] = content
    record["timestamp"] = datetime.datetime.now().isoformat()
    time.sleep(1)
    return record


def download_vldb_papers(vldb_listing_page_urls_id, output_dir, execution_engine, profile=False):
    """This function downloads a bunch of VLDB papers from an online listing and saves them to disk.  It also saves a CSV file of the paper listings."""
    # 1. Grab the input VLDB listing page(s) and scrape them for paper metadata
    tfs = Dataset(
        vldb_listing_page_urls_id,
        schema=TextFile,
        desc="A file full of URLs of VLDB journal pages",
    )
    urls = tfs.convert(
        output_schema=URL,
        udf=vldb_text_file_to_url,
        desc="The actual URLs of the VLDB pages",
        cardinality=Cardinality.ONE_TO_MANY,
    ).project(["url"])
    html_content = urls.convert(output_schema=WebPage, udf=download_html)
    vldb_paper_listings = html_content.convert(
        output_schema=VLDBPaperListing,
        desc="The actual listings for each VLDB paper",
        cardinality=Cardinality.ONE_TO_MANY,
    )

    listing_records, listing_execution_stats = Execute(
        vldb_paper_listings,
        policy=MaxQuality(),
        nocache=True,
        allow_token_reduction=False,
        allow_code_synth=False,
        execution_engine=execution_engine,
        verbose=True,
    )

    # save the paper listings to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vldbPaperListings.csv")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=listing_records[0].__dict__.keys())
        writer.writeheader()
        for record in listing_records:
            writer.writerow(record.as_dict())

    if profile:
        with open("profiling-data/vldb1-profiling.json", "w") as f:
            json.dump(listing_execution_stats.to_json(), f)

    # 2. Get the PDF URL for each paper that's listed and download it
    pdf_content = vldb_paper_listings.convert(
        output_schema=Download,
        udf=download_pdf,
    ).project(["url", "content", "timestamp"])

    # 3. Save the paper listings to a CSV file and the PDFs to disk
    pdf_records, download_execution_stats = Execute(
        pdf_content,
        policy=MaxQuality(),
        nocache=True,
        allow_token_reduction=False,
        allow_code_synth=False,
        execution_engine=execution_engine,
        verbose=True,
    )

    for idx, pdf_record in enumerate(pdf_records):
        with open(os.path.join(output_dir, str(idx) + ".pdf"), "wb") as f:
            f.write(pdf_record.content)

    if profile:
        with open("profiling-data/vldb2-profiling.json", "w") as f:
            json.dump(download_execution_stats.to_json(), f)


class GitHubUpdate(Schema):
    """GitHubUpdate represents a single commit message from a GitHub repo"""

    commitId = Field(desc="The unique identifier for the commit") # noqa
    reponame = Field(desc="The name of the repository")
    commit_message = Field(desc="The message associated with the commit")
    commit_date = Field(desc="The date the commit was made")
    committer_name = Field(desc="The name of the person who made the commit")
    file_names = Field(desc="The list of files changed in the commit")


def test_user_source(dataset_id: str):
    return Dataset(dataset_id, schema=GitHubUpdate)


class Email(TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = Field(desc="The email address of the sender")
    subject = Field(desc="The subject of the email")


def build_enron_plan(dataset_id):
    emails = Dataset(dataset_id, schema=Email)
    return emails


def compute_enron_stats(dataset_id):
    emails = Dataset(dataset_id, schema=Email)
    subject_line_lengths = emails.convert(Number, desc="The number of words in the subject field")
    return subject_line_lengths


def enron_gby_plan(dataset_id):
    emails = Dataset(dataset_id, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = ["sender"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_emails = emails.groupby(gby_desc)
    return grouped_emails


def enron_count_plan(dataset_id):
    emails = Dataset(dataset_id, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = []
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    count_emails = emails.groupby(gby_desc)
    return count_emails


def enron_average_count_plan(dataset_id):
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
    data = Dataset(dataset_id, schema=Email)
    limit_data = data.limit(limit)
    return limit_data


class DogImage(ImageFile):
    breed = Field(desc="The breed of the dog")


def build_image_plan(dataset_id):
    images = Dataset(dataset_id, schema=ImageFile)
    filtered_images = images.filter("The image contains one or more dogs")
    dog_images = filtered_images.convert(DogImage, desc="Images of dogs")
    return dog_images


def build_image_agg_plan(dataset_id):
    images = Dataset(dataset_id, schema=ImageFile)
    filtered_images = images.filter("The image contains one or more dogs")
    dog_images = filtered_images.convert(DogImage, desc="Images of dogs")
    ops = ["count"]
    fields = ["breed"]
    groupbyfields = ["breed"]
    gby_desc = GroupBySig(groupbyfields, ops, fields)
    grouped_dog_images = dog_images.groupby(gby_desc)
    return grouped_dog_images


def print_table(records, cols=None, gradio=False, plan_str=None):
    records = [{key: record[key] for key in record.get_field_names()} for record in records]
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


if __name__ == "__main__":
    # parse arguments
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--profile", default=False, action="store_true", help="Profile execution")
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument("--task", type=str, help="The task to run")
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

    # The user has to indicate the dataset id and the task
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.task is None:
        print("Please provide a task")
        exit(1)

    # create directory for profiling data
    if args.profile:
        os.makedirs("profiling-data", exist_ok=True)

    datasetid = args.datasetid
    task = args.task
    verbose = args.verbose
    policy = MaxQuality()
    if args.policy == "mincost":
        policy = MinCost()
    elif args.policy == "mintime":
        policy = MinTime()
    elif args.policy == "maxquality":
        policy = MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    execution_engine = None
    executor = args.executor
    if executor == "sequential":
        execution_engine = NoSentinelSequentialSingleThreadExecution
    elif executor == "pipelined":
        execution_engine = NoSentinelPipelinedSingleThreadExecution
    elif executor == "parallel":
        execution_engine = NoSentinelPipelinedParallelExecution
    else:
        print("Executor not supported for this demo")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

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

    elif task == "usersource":
        # register the ephemeral dataset
        datasetid = "githubtest"
        owner = "mikecafarella"
        repo = "palimpzest"
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"

        class GitHubCommitSource(UserSource):
            def __init__(self, dataset_id):
                super().__init__(RawJSONObject, dataset_id)
                per_page = 100
                params = {"per_page": per_page, "page": 1}
                self.commits = []
                while True:
                    response = requests.get(url, params=params)
                    commits = response.json()

                    if not commits or response.status_code != 200:
                        break

                    self.commits.extend(commits)
                    if len(commits) < per_page:
                        break

                    params["page"] += 1
                    time.sleep(1)

                # to make the demo go faster
                self.commits = self.commits[:10]

            def __len__(self):
                return len(self.commits)

            def get_size(self):
                return sum(map(lambda commit: sys.getsizeof(commit), self.commits))

            def get_item(self, idx: int):
                # NOTE: we can make this a streaming demo again by modifying this get_item function
                commit = self.commits[idx]
                commit_str = json.dumps(commit)
                dr = DataRecord(self.schema, source_id=idx)
                dr.json = commit_str

                return dr

        DataDirectory().register_user_source(GitHubCommitSource(datasetid), datasetid)

        root_set = test_user_source(datasetid)
        cols = ["commitId", "reponame", "commit_message"]
        stat_path = "profiling-data/usersource-profiling.json"

    elif task == "gbyImage":
        root_set = build_image_agg_plan(datasetid)
        cols = ["breed", "count(breed)"]
        stat_path = "profiling-data/gbyImage-profiling.json"

    elif task == "image":
        root_set = build_image_plan(datasetid)
        stat_path = "profiling-data/image-profiling.json"

    # NOTE: VLDB seems to rate limit downloads; causing the program to hang
    elif task == "vldb":
        download_vldb_papers(datasetid, "vldbPapers", execution_engine, profile=args.profile)

    elif task == "limit":
        root_set = enron_limit_plan(datasetid, 5)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/limit-profiling.json"

    else:
        print("Unknown task")
        exit(1)

    records, execution_stats = Execute(
        root_set,
        policy=policy,
        nocache=True,
        allow_token_reduction=False,
        allow_code_synth=False,
        execution_engine=execution_engine,
        verbose=verbose,
    )

    print(f"Policy is: {str(policy)}")
    print("Executed plan:")
    plan_str = list(execution_stats.plan_strs.values())[0]
    print(plan_str)
    end_time = time.time()
    print("Elapsed time:", end_time - start_time)

    if args.profile:
        with open(stat_path, "w") as f:
            json.dump(execution_stats.to_json(), f)

    if task == "image":
        imgs, breeds = [], []
        for record in records:
            path = os.path.join("testdata/images-tiny/", record.filename)
            print(record)
            print("Trying to open ", path)
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

            gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()

    else:
        print_table(records, cols=cols, gradio=False, plan_str=plan_str)
