#!/usr/bin/env python3
import palimpzest as pz

from tabulate import tabulate
from PIL import Image

from palimpzest.elements import GroupBySig

import gradio as gr
import numpy as np
import pandas as pd

import argparse
import requests
import json
import time
import os
import csv


class ScientificPaper(pz.PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""

    title = pz.Field(
        desc="The title of the paper. This is a natural language title, not a number or letter.",
        required=True,
    )
    publicationYear = pz.Field(
        desc="The year the paper was published. This is a number.", required=False
    )
    author = pz.Field(desc="The name of the first author of the paper", required=True)
    institution = pz.Field(
        desc="The institution of the first author of the paper", required=True
    )
    journal = pz.Field(
        desc="The name of the journal the paper was published in", required=True
    )
    fundingAgency = pz.Field(
        desc="The name of the funding agency that supported the research",
        required=False,
    )


def buildSciPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    return pz.Dataset(datasetId, schema=ScientificPaper)


def buildTestPDFPlan(datasetId):
    """This tests whether we can process a PDF file"""
    return pz.Dataset(datasetId, schema=pz.PDFFile)


def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    sciPapers = pz.Dataset(datasetId, schema=ScientificPaper)
    batteryPapers = sciPapers.filter("The paper is about batteries")
    mitPapers = batteryPapers.filter("The paper is from MIT")

    return mitPapers


class VLDBPaperListing(pz.Schema):
    """VLDBPaperListing represents a single paper from the VLDB conference"""

    title = pz.Field(desc="The title of the paper", required=True)
    authors = pz.Field(desc="The authors of the paper", required=True)
    pdfLink = pz.Field(desc="The link to the PDF of the paper", required=True)


def downloadVLDBPapers(vldbListingPageURLsId, outputDir, profile=False):
    """This function downloads a bunch of VLDB papers from an online listing and saves them to disk.  It also saves a CSV file of the paper listings."""
    policy = pz.MaxQuality()

    # 1. Grab the input VLDB listing page(s) and scrape them for paper metadata
    tfs = pz.Dataset(
        vldbListingPageURLsId,
        schema=pz.TextFile,
        desc="A file full of URLs of VLDB journal pages",
    )
    urls = tfs.convert(
        pz.URL, desc="The actual URLs of the VLDB pages", cardinality="oneToMany"
    )  # oneToMany=True would be nice here.
    htmlContent = urls.map(pz.DownloadHTMLFunction())
    vldbPaperListings = htmlContent.convert(
        VLDBPaperListing,
        desc="The actual listings for each VLDB paper",
        cardinality="oneToMany",
    )

    # 2. Get the PDF URL for each paper that's listed and download it
    vldbPaperURLs = vldbPaperListings.convert(
        pz.URL, desc="The URLs of the PDFs of the VLDB papers"
    )
    pdfContent = vldbPaperURLs.map(pz.DownloadBinaryFunction())

    # 3. Save the paper listings to a CSV file and the PDFs to disk
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputPath = os.path.join(outputDir, "vldbPaperListings.csv")

    engine = pz.PipelinedParallelExecution
    listingRecords, plan, stats = pz.Execute(rootSet, 
                                policy = policy,
                                nocache=True,
                                execution_engine=engine)

    with open(outputPath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=listingRecords[0].__dict__.keys())
        writer.writeheader()
        for record in listingRecords:
            writer.writerow(record._asDict())

    # if profiling was turned on; capture statistics
    if profile:
        with open("profiling-data/vldb1-profiling.json", "w") as f:
            json.dump(stats.to_dict(), f)

    # TODO
    # physicalTree2 = emitDataset(
    #     pdfContent,
    #     policy,
    #     title="VLDB paper dump",
    #     verbose=True,
    # )
    # for idx, r in enumerate(physicalTree2):
    #     with open(os.path.join(outputDir, str(idx) + ".pdf"), "wb") as f:
    #         f.write(r.content)

    # # if profiling was turned on; capture statistics
    # if profile:
    #     profiling_data = physicalTree2.getProfilingData()
    #     sp = StatsProcessor(profiling_data)

    #     with open("profiling-data/vldb2-profiling.json", "w") as f:
    #         json.dump(sp.profiling_data.to_dict(), f)


class GitHubUpdate(pz.Schema):
    """GitHubUpdate represents a single commit message from a GitHub repo"""

    commitId = pz.Field(desc="The unique identifier for the commit", required=True)
    reponame = pz.Field(desc="The name of the repository", required=True)
    commit_message = pz.Field(
        desc="The message associated with the commit", required=True
    )
    commit_date = pz.Field(desc="The date the commit was made", required=True)
    committer_name = pz.Field(
        desc="The name of the person who made the commit", required=True
    )
    file_names = pz.Field(
        desc="The list of files changed in the commit", required=False
    )


def testStreaming(datasetId: str):
    return pz.Dataset(datasetId, schema=GitHubUpdate)


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)


def buildEnronPlan(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    return emails


def computeEnronStats(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    subjectLineLengths = emails.convert(
        pz.Number, desc="The number of words in the subject field"
    )
    return subjectLineLengths


def enronGbyPlan(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = ["sender"]
    gbyDesc = GroupBySig(groupbyfields, ops, fields)
    groupedEmails = emails.groupby(gbyDesc)
    return groupedEmails


def enronCountPlan(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = []
    gbyDesc = GroupBySig(groupbyfields, ops, fields)
    countEmails = emails.groupby(gbyDesc)
    return countEmails


def enronAverageCountPlan(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    ops = ["count"]
    fields = ["sender"]
    groupbyfields = ["sender"]
    gbyDesc = GroupBySig(groupbyfields, ops, fields)
    groupedEmails = emails.groupby(gbyDesc)
    ops = ["average"]
    fields = ["count(sender)"]
    groupbyfields = []
    gbyDesc = GroupBySig(groupbyfields, ops, fields)
    averageEmailsPerSender = groupedEmails.groupby(gbyDesc)

    return averageEmailsPerSender

def enronLimitPlan(datasetId, limit=5):
    data = pz.Dataset(datasetId, schema=Email)
    limitData = data.limit(limit)
    return limitData



class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required=True)


def buildImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filter("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc="Images of dogs")
    return dogImages


def buildImageAggPlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filter("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc="Images of dogs")
    ops = ["count"]
    fields = ["breed"]
    groupbyfields = ["breed"]
    gbyDesc = GroupBySig(groupbyfields, ops, fields)
    groupedDogImages = dogImages.groupby(gbyDesc)
    return groupedDogImages


def printTable(records, cols=None, gradio=False, query=None, plan_str=None):
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

    if not gradio:
        print(tabulate(records_df[print_cols], headers="keys", tablefmt="psql"))

    else:
        with gr.Blocks() as demo:
            gr.Dataframe(records_df[print_cols])

            if plan_str is not None:
                gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()

# TODO: clean up script; remove UserFunctions
if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--profile", default=False, action="store_true", help="Profile execution"
    )
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument("--task", type=str, help="The task to run")
    parser.add_argument(
        "--engine",
        type=str,
        help='The engine to use. One of sentinel, nosentinel',
        default='sentinel',
    )
    parser.add_argument(
        "--executor",
        type=str,
        help='The plan executor to use. One of sequential, pipelined, parallel',
        default='parallel',
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default='mincost',
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

    execution_engine = None
    engine, executor = args.engine, args.executor
    if engine == "sentinel":
        if executor == "sequential":
            execution_engine = pz.SequentialSingleThreadSentinelExecution
        elif executor == "pipelined":
            execution_engine = pz.PipelinedSingleThreadSentinelExecution
        elif executor == "parallel":
            execution_engine = pz.PipelinedParallelSentinelExecution
        else:
            print("Unknown executor")
            exit(1)
    elif engine == "nosentinel":
        if executor == "sequential":
            execution_engine = pz.SequentialSingleThreadNoSentinelExecution
        elif executor == "pipelined":
            execution_engine = pz.PipelinedSingleThreadNoSentinelExecution
        elif executor == "parallel":
            execution_engine = pz.PipelinedParallelNoSentinelExecution
        else:
            print("Unknown executor")
            exit(1)
    else:
        print("Unknown engine")
        exit(1)
    
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    if task == "paper":
        rootSet = buildMITBatteryPaperPlan(datasetid)
        cols = ["title", "publicationYear", "author", "institution", "journal", "fundingAgency"]
        stat_path = "profiling-data/paper-profiling.json"

    elif task == "enron":
        rootSet = buildEnronPlan(datasetid)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/enron-profiling.json"

    elif task == "enronGby":
        rootSet = enronGbyPlan(datasetid)
        cols = ["sender", "count(sender)"]
        stat_path = "profiling-data/egby-profiling.json"

    elif task in ("enronCount", "count"):
        rootSet = enronCountPlan(datasetid)
        cols = ["count(sender)"]
        stat_path = "profiling-data/ecount-profiling.json"

    elif task in ("enronAvgCount", "average"):
        rootSet = enronAverageCountPlan(datasetid)
        cols = ["average(count(sender))"]
        stat_path = "profiling-data/e-profiling.json"

    elif task == "enronmap":
        rootSet = computeEnronStats(datasetid)
        cols = ["sender", "subject", "value"]
        stat_path = "profiling-data/emap-profiling.json"

    elif task == "pdftest":
        rootSet = buildTestPDFPlan(datasetid)
        cols = ["filename"]
        stat_path = "profiling-data/pdftest-profiling.json"

    elif task == "scitest":
        rootSet = buildSciPaperPlan(datasetid)
        cols = ["title", "author", "institution", "journal", "fundingAgency"]
        stat_path = "profiling-data/scitest-profiling.json"

    # TODO
    elif task == "streaming":
        # register the ephemeral dataset
        datasetid = "githubtest"
        owner = "mikecafarella"
        repo = "palimpzest"
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        blockTime = 5

        class GitHubCommitSource(pz.UserSource):
            def __init__(self, datasetId):
                super().__init__(pz.RawJSONObject, datasetId)

            def userImplementedIterator(self):
                per_page = 100
                params = {"per_page": per_page, "page": 1}
                while True:
                    response = requests.get(url, params=params)
                    commits = response.json()

                    if not commits or response.status_code != 200:
                        break

                    for commit in commits:
                        # Process each commit here
                        commitStr = json.dumps(commit)
                        dr = pz.DataRecord(self.schema)
                        dr.json = commitStr
                        yield dr

                    if len(commits) < per_page:
                        break

                    params["page"] += 1
                    time.sleep(1)

        pz.DataDirectory().registerUserSource(GitHubCommitSource(datasetid), datasetid)

        rootSet = testStreaming(datasetid)
        cols = None
        stat_path = "profiling-data/streaming-profiling.json"

    elif task == "gbyImage":
        rootSet = buildImageAggPlan(datasetid)
        cols = ["breed", "count(breed)"]
        stat_path = "profiling-data/gbyImage-profiling.json"

    elif task == "image":
        rootSet = buildImagePlan(datasetid)
        stat_path = "profiling-data/image-profiling.json"

    # TODO
    elif task == "vldb":
        downloadVLDBPapers(datasetid, "vldbPapers", profile=args.profile)

    elif task == "limit":
        rootSet = enronLimitPlan(datasetid, 5)
        cols = ["sender", "subject"]
        stat_path = "profiling-data/limit-profiling.json"

    else:
        print("Unknown task")
        exit(1)

    records, execution_stats = pz.Execute(rootSet, 
                                    policy = policy,
                                    nocache=True,
                                    allow_token_reduction=False,
                                    allow_code_synth=False,
                                    execution_engine=execution_engine,
                                    verbose=verbose)

    print(f"Policy is: {str(policy)}")
    print("Executed plan:")
    plan_str = list(execution_stats.plan_strs.values())[0]
    print(plan_str)
    endTime = time.time()
    print("Elapsed time:", endTime - startTime)

    if args.profile:
        with open(stat_path, "w") as f:
            json.dump(execution_stats.to_json(), f)

    if task == 'image':
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

    elif task == 'pdftest':
        records = [pz.Number() for r in records] 
        records = [setattr(number, "value", idx) for idx, number in enumerate(records)]
        printTable(records, cols=cols, gradio=True, plan_str=plan_str)        
    else:
        printTable(records, cols=cols, gradio=False, plan_str=plan_str)
