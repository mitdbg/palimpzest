#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from tabulate import tabulate
from PIL import Image


from palimpzest.execution import Execution
from palimpzest.elements import DataRecord

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
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
   author = pz.Field(desc="The name of the first author of the paper", required=True)
   institution = pz.Field(desc="The institution of the first author of the paper", required=True)
   journal = pz.Field(desc="The name of the journal the paper was published in", required=True)
   fundingAgency = pz.Field(desc="The name of the funding agency that supported the research", required=False)

def buildSciPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    return pz.Dataset(datasetId, schema=ScientificPaper)

def buildTestPDFPlan(datasetId):
    """This tests whether we can process a PDF file"""
    pdfPapers = pz.Dataset(datasetId, schema=pz.PDFFile)

    return pdfPapers

def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    sciPapers = pz.Dataset(datasetId, schema=ScientificPaper)
    batteryPapers = sciPapers.filterByStr("The paper is about batteries")
    mitPapers = batteryPapers.filterByStr("The paper is from MIT")

    return mitPapers

class VLDBPaperListing(pz.Schema):
    """VLDBPaperListing represents a single paper from the VLDB conference"""
    title = pz.Field(desc="The title of the paper", required=True)
    authors = pz.Field(desc="The authors of the paper", required=True)
    pdfLink = pz.Field(desc="The link to the PDF of the paper", required=True)

def downloadVLDBPapers(vldbListingPageURLsId, outputDir):
    """ This function downloads a bunch of VLDB papers from an online listing and saves them to disk.  It also saves a CSV file of the paper listings."""
    policy = pz.MaxQuality()

    # 1. Grab the input VLDB listing page(s) and scrape them for paper metadata
    tfs = pz.Dataset(vldbListingPageURLsId, schema=pz.TextFile, desc="A file full of URLs of VLDB journal pages")
    urls = tfs.convert(pz.URL, desc="The actual URLs of the VLDB pages", cardinality="oneToMany")  # oneToMany=True would be nice here.   
    htmlContent = urls.map(pz.DownloadHTMLFunction())
    vldbPaperListings = htmlContent.convert(VLDBPaperListing, desc="The actual listings for each VLDB paper", cardinality="oneToMany")

    # 2. Get the PDF URL for each paper that's listed and download it
    vldbPaperURLs = vldbPaperListings.convert(pz.URL, desc="The URLs of the PDFs of the VLDB papers")
    pdfContent = vldbPaperURLs.map(pz.DownloadBinaryFunction())

    # 3. Save the paper listings to a CSV file and the PDFs to disk
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputPath = os.path.join(outputDir, "vldbPaperListings.csv")

    physicalTree1 = emitDataset(vldbPaperListings, policy, title="VLDB paper dump", verbose=True)
    listingRecords = [r for r in physicalTree1]
    with open(outputPath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=listingRecords[0].__dict__.keys())
        writer.writeheader()
        for record in listingRecords:
            writer.writerow(record.asDict())

    physicalTree2 = emitDataset(pdfContent, policy, title="VLDB paper dump", verbose=True)
    for idx, r in enumerate(physicalTree2):
        with open(os.path.join(outputDir, str(idx) + ".pdf"), "wb") as f:
            f.write(r.content)

#    For debugging
#    physicalTree = emitDataset(vldbPaperListings, policy, title="VLDB papers", verbose=True)
#    listingRecords = [r for r in physicalTree]
#    printTable(listingRecords, gradio=True, plan=physicalTree)


class GitHubUpdate(pz.Schema):
    """GitHubUpdate represents a single commit message from a GitHub repo"""
    commitId = pz.Field(desc="The unique identifier for the commit", required=True)
    reponame = pz.Field(desc="The name of the repository", required=True)
    commit_message = pz.Field(desc="The message associated with the commit", required=True)
    commit_date = pz.Field(desc="The date the commit was made", required=True)
    committer_name = pz.Field(desc="The name of the person who made the commit", required=True)
    file_names = pz.Field(desc="The list of files changed in the commit", required=False)

def testStreaming(datasetId: str):
    return pz.Dataset(datasetId, schema=GitHubUpdate)

def testCount(datasetId):
    files = pz.Dataset(datasetId)
    fileCount = files.aggregate("COUNT")
    return fileCount

def testAverage(datasetId):
    data = pz.Dataset(datasetId)
    average = data.aggregate("AVERAGE")
    return average

def testLimit(datasetId, n):
    data = pz.Dataset(datasetId)
    limitData = data.limit(n)
    return limitData

class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)

def buildEnronPlan(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    return emails

def computeEnronStats(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    subjectLineLengths = emails.convert(pz.Number, desc = "The number of words in the subject field")
    return subjectLineLengths

class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required = True)

def buildImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filterByStr("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc = "Images of dogs")
    return dogImages


def buildNestedStr(node, indent=0, buildStr=""):
        elt, child = node
        indentation = " " * indent
        buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
        if child is not None:
            return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
        else:
            return buildStr

def printTable(records, cols=None, gradio=False, query=None, plan=None):
    records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_')
        }
        for record in records
    ]
    records_df = pd.DataFrame(records)
    print_cols = records_df.columns if cols is None else cols

    if not gradio:
        print(tabulate(records_df[print_cols], headers="keys", tablefmt='psql'))

    else:
        with gr.Blocks() as demo:
            gr.Dataframe(records_df[print_cols])

            if plan is not None:
                plan_str = buildNestedStr(plan.dumpPhysicalTree())
                gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()

def emitDataset(rootSet, policy, title="Dataset", verbose=False, shouldProfile=False):
    def emitNestedTuple(node, indent=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            emitNestedTuple(child, indent=indent+2)

    # print()
    # print()
    # print("# Let's test the basic functionality of the system")

    # Print the syntactic tree
    syntacticElements = rootSet.dumpSyntacticTree()
    # print()
    # print("Syntactic operator tree")
    # emitNestedTuple(syntacticElements)

    # Print the (possibly optimized) logical tree
    logicalTree = rootSet.getLogicalTree()
    logicalElements = logicalTree.dumpLogicalTree()

    # print()
    #print("Logical operator tree")
    #emitNestedTuple(logicalElements)

    # Generate candidate physical plans
    candidatePlans = logicalTree.createPhysicalPlanCandidates(shouldProfile=shouldProfile)

    # print out plans to the user if it is their choice
    if args.policy == "user":
        print("----------")
        for idx, cp in enumerate(candidatePlans):
            print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
            print("Physical operator tree")
            physicalOps = cp[3].dumpPhysicalTree()
            emitNestedTuple(physicalOps)
            print("----------")

    # have policy select the candidate plan to execute
    planTime, planCost, quality, physicalTree = policy.choose(candidatePlans)
    print("----------")
    print(f"Policy is: {str(policy)}")
    print(f"Chose plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
    emitNestedTuple(physicalTree.dumpPhysicalTree())


    #iterate over data
    # print()
    # print("Estimated seconds to complete:", planTime)
    # print("Estimated USD to complete:", planCost)
    # print("Concrete data results")
    return physicalTree

#
# Get battery papers and emit!
#
if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print verbose output')
    parser.add_argument('--profile', default=False, action='store_true', help='Profile execution')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--task' , type=str, help='The task to run')
    parser.add_argument('--policy', type=str, help="One of 'user', 'mincost', 'mintime', 'maxquality', 'harmonicmean'")

    args = parser.parse_args()

    # The user has to indicate the dataset id and the task
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.task is None:
        print("Please provide a task")
        exit(1)

    datasetid = args.datasetid
    task = args.task
    policy = pz.MaxHarmonicMean()
    if args.policy is not None:
        if args.policy == "user":
            policy = pz.UserChoice()
        elif args.policy == "mincost":
            policy = pz.MinCost()
        elif args.policy == "mintime":
            policy = pz.MinTime()
        elif args.policy == "maxquality":
            policy = pz.MaxQuality()
        elif args.policy == "harmonicmean":
            policy = pz.MaxHarmonicMean()

    if os.getenv('OPENAI_API_KEY') is None and os.getenv('TOGETHER_API_KEY') is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")


    if task == "paper":
        rootSet = buildMITBatteryPaperPlan(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Good MIT battery papers written by good authors", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(
            records,
            cols=["title", "publicationYear", "author", "institution", "journal", "fundingAgency"],
            gradio=True,
            plan=physicalTree,
        )

    elif task == "enron":
        rootSet = buildEnronPlan(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Enron emails", verbose=args.verbose, shouldProfile=args.profile)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, cols=["sender", "subject"], gradio=True, plan=physicalTree)

        # if profiling was turned on; capture statistics
        if Profiler.profiling_on():
            profiling_data = physicalTree.getProfilingData()
            sp = StatsProcessor(profiling_data)

            with open('profiling.json', 'w') as f:
                json.dump(sp.profiling_data, f)

    elif task == "enronoptimize":
        rootSet = buildEnronPlan(datasetid)
        execution = pz.Execution(rootSet, policy)
        physicalTree = execution.executeAndOptimize()
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, cols=["sender", "subject"], gradio=True, plan=physicalTree)

    elif task == "enronmap":
        rootSet = computeEnronStats(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Enron subject counts", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, gradio=True, plan=physicalTree)

    elif task == "pdftest":
        rootSet = buildTestPDFPlan(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="PDF files", verbose=args.verbose)
        records = [pz.Number() for r in enumerate(physicalTree)]
        records = [setattr(number, 'value', idx) for idx, number in enumerate(records)]
        print("----------")
        print()
        printTable(records, gradio=True, plan=physicalTree)

    elif task == "scitest":
        rootSet = buildSciPaperPlan(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Scientific files", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, cols=["title", "author", "institution", "journal", "fundingAgency"], gradio=True, plan=physicalTree)

    elif task == "streaming":
        # register the ephemeral dataset
        datasetid = "ephemeral:githubtest"
        owner = "mikecafarella"
        repo = "palimpzest"
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        blockTime = 5

        class GitHubCommitSource(pz.UserSource):
            def __init__(self, datasetId):
                super().__init__(pz.RawJSONObject, datasetId)

            def userImplementedIterator(self):
                per_page = 100
                params = {
                    'per_page': per_page,
                    'page': 1
                }
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

                    params['page'] += 1
                    time.sleep(1)

        pz.DataDirectory().registerUserSource(GitHubCommitSource(datasetid), datasetid)

        rootSet = testStreaming(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Streaming items", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, gradio=True, plan=physicalTree)


    elif task == "image":
        print("Starting image task")
        rootSet = buildImagePlan(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Dogs", verbose=args.verbose)
        records = [r for r in physicalTree]

        print("Obtained records", records)
        imgs, breeds = [], []
        for record in records:
            print("Trying to open ", record.filename)
            img = Image.open(record.filename).resize((128,128))
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

            plan_str = buildNestedStr(physicalTree.dumpPhysicalTree())
            gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()

        # if profiling was turned on; capture statistics
        if Profiler.profiling_on():
            profiling_data = physicalTree.getProfilingData()

            with open('profiling.json', 'w') as f:
                json.dump(profiling_data, f)

    elif task == "vldb":
        downloadVLDBPapers(datasetid, "vldbPapers")

    elif task == "count":
        rootSet = testCount(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Count records", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, gradio=True, plan=physicalTree)

    elif task == "average":
        rootSet = testAverage(datasetid)
        physicalTree = emitDataset(rootSet, policy, title="Average of numbers", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, gradio=True, plan=physicalTree)

    elif task == "limit":
        rootSet = testLimit(datasetid, 5)
        physicalTree = emitDataset(rootSet, policy, title="Limit the set to 5 items", verbose=args.verbose)
        records = [r for r in physicalTree]
        print("----------")
        print()
        printTable(records, gradio=True, plan=physicalTree)

    else:
        print("Unknown task")
        exit(1)

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
