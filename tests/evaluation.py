#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from tabulate import tabulate
from PIL import Image


from palimpzest.constants import Model
from palimpzest.execution import Execution
from palimpzest.elements import DataRecord, GroupBySig

import matplotlib.pyplot as plt
import pandas as pd

import argparse
import json
import shutil
import time
import os


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)
    # to = pz.ListField(element_type=pz.StringField, desc="The email address(es) of the recipient(s)", required=True)
    # cced = pz.ListField(element_type=pz.StringField, desc="The email address(es) CC'ed on the email", required=True)

# TODO: it might not be obvious to a new user how to write/split up a schema for multimodal file data;
#       under our current setup, we have one schema which represents a file (e.g. pz.File), so the equivalent
#       here is to have a schema which represents the different (sets of) files, but I feel like users
#       will naturally just want to define the fields they wish to extract from the underlying (set of) files
#       and have PZ take care of the rest
class RealEstateListing(pz.Schema):
    """The source text and image data for a real estate listing."""
    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.BytesField(desc="The content of the listing's text description", required=True)
    image_contents = pz.ListField(element_type=pz.BytesField, desc="A list of the contents of each image of the listing", required=True)

# TODO: if we want to apply filters based on images, right now we have to get a description and then
#       apply a filter on that description; we want a way to directly query the model for what it is
#       we seek to know about the image in its output description (and possibly filter directly on that)
class ModernNearMITRealEstateListing(RealEstateListing):
    """Represents a real estate listing which consists of text as well as a set of photos."""
    location = pz.StringField(desc="The address of the property")
    sq_ft = pz.NumericField(desc="The square footage (sq. ft.) of the property")
    year_built = pz.NumericField(desc="The year in which the property was built")
    bedrooms = pz.NumericField(desc="The number of bedrooms")
    bathrooms = pz.NumericField(desc="The number of bathrooms")
    image_descriptions = # TODO

class RealEstateListingSource(pz.UserSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListing, datasetId)
        self.listings_dir = listings_dir

    def userImplementedIterator(self):
        for root, _, files in os.walk(self.listings_dir):
            if root == self.listings_dir:
                continue

            # create data record
            dr = pz.DataRecord(self.schema)
            dr.listing = root.split("/")[-1]
            dr.image_contents = []
            for file in files:
                bytes_data = open(os.path.join(root, file), "rb").read()
                if file.endswith('.txt'):
                    dr.text_content = bytes_data
                elif file.endswith('.png'):
                    dr.image_contents.append(bytes_data)
            yield dr


def buildNestedStr(node, indent=0, buildStr=""):
    elt, child = node
    indentation = " " * indent
    buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
    if child is not None:
        return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
    else:
        return buildStr

def compute_label(physicalTree, label_idx):
    """
    Map integer to physical plan.
    """
    physicalOps = physicalTree.dumpPhysicalTree()
    label = buildNestedStr(physicalOps)
    print(f"LABEL {label_idx}: {label}")
    return f"PZ-{label_idx}"


def score_enron_plan(datasetid, records) -> float:
    """
    Computes the F1 score of the enron plan
    """
    # parse records
    records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_')
        }
        for record in records
    ]
    records_df = pd.DataFrame(records)
    if records_df.empty:
        return 0.0, 0.0, 0.0

    pred_filenames = records_df.filename.apply(lambda fn: os.path.basename(fn)).tolist()

    # get groundtruth
    gt_df = (
        pd.read_csv("testdata/groundtruth/enron-eval.csv")
        if datasetid == "enron-eval"
        else pd.read_csv("testdata/groundtruth/enron-eval-tiny.csv")
    )
    target_filenames = list(gt_df[gt_df.label == 1].filename.unique())

    # compute true and false positives
    tp, fp = 0, 0
    for filename in pred_filenames:
        if filename in target_filenames:
            tp += 1
        else:
            fp += 1

    # compute false negatives
    fn = 0
    for filename in target_filenames:
        if filename not in pred_filenames:
            fn += 1

    # compute precision, recall, f1 score
    precision = tp/(tp + fp) if tp + fp > 0 else 0.0
    recall = tp/(tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1_score


def evaluate_enron_baseline(model, datasetid):
    """
    Perform single shot evaluation with the given model
    """
    print("----------------------")
    print(f"Model: {model.value}")
    print("---")
    # construct generator
    doc_schema = str(Email)
    doc_type = Email.className()
    generator = pz.DSPyGenerator(model.value, pz.PromptStrategy.DSPY_COT_BOOL, doc_schema, doc_type, False)

    # initialize metrics
    total_input_tokens, total_output_tokens = 0, 0

    # iterate over files and compute predictions
    ds = pz.DataDirectory().getRegisteredDataset(datasetid)
    start_time = time.time()
    output_records = []
    for record in ds:
        filename = os.path.basename(record.filename)
        content = str(record.contents)
        print(f"running record: {filename}")

        filterCondition = """
        The email meets both of the following criteria:
        1. it refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")
        2. it is not quoting from a news article or an article written by someone outside of Enron
        """
        # 2. it is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name
        answer, gen_stats = generator.generate(content, filterCondition)

        # update token usage
        total_input_tokens += gen_stats.usage["prompt_tokens"]
        total_output_tokens += gen_stats.usage["completion_tokens"]

        # add record to output records if answer is true
        # if answer.lower().strip() == "true":
        if "true" in answer.lower():
            output_records.append(record)

    # compute runtime
    runtime = time.time() - start_time

    # compute USD cost of generation
    usd_per_input_token = pz.MODEL_CARDS[gen_stats.model_name]["usd_per_input_token"]
    usd_per_output_token = pz.MODEL_CARDS[gen_stats.model_name]["usd_per_output_token"]
    total_input_usd = total_input_tokens * usd_per_input_token
    total_output_usd = total_output_tokens * usd_per_output_token
    cost = total_input_usd + total_output_usd

    # compute f1_score
    _, _, f1_score = score_enron_plan(datasetid, output_records)

    # compute label and return
    label = None
    if model == Model.GPT_4:
        label = "GPT-4"
    elif model == Model.GPT_3_5:
        label = "GPT-3.5"
    elif model == Model.MIXTRAL:
        label = "MIXTRAL-7B"

    return runtime, cost, f1_score, label


def run_enron_pz_plan(datasetid, plan, idx):
    """
    I'm placing this in a separate file from evaluate_enron_pz to see if this prevents
    an error where the DSPy calls to Gemini (and other models?) opens too many files.
    My hope is that placing this inside a separate function will cause the file descriptors
    to be cleaned up once the function returns.
    """
    # TODO: eventually get runtime from profiling data
    # execute plan to get records and runtime;
    start_time = time.time()
    records = [r for r in plan]
    runtime = time.time() - start_time

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)
    with open(f'eval-results/enron-profiling-{idx}.json', 'w') as f:
        json.dump(sp.profiling_data.to_dict(), f)

    # score plan based on its output records
    _, _, f1_score = score_enron_plan(datasetid, records)

    cost = 0.0
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        stats = stats.source_op_stats

    # compute label
    label = compute_label(plan, idx)

    return runtime, cost, f1_score, label


def evaluate_enron_pz(datasetid, reoptimize=False, limit=None):
    """
    This creates the PZ set of plans for the Enron email evaluation.

    Make sure to pre-register the dataset with:

    $ pz reg --path testdata/enron-eval --name enron-eval
    """
    # TODO: we can expand this dataset, but it's good enough for now
    emails = pz.Dataset(datasetid, schema=Email)
    emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
    # emails = emails.filterByStr("The email is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name")
    emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")

    logicalTree = emails.getLogicalTree()
    candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, shouldProfile=True)
    results = []
    for idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
        print("----------------------")
        print(f"Plan: {buildNestedStr(plan.dumpPhysicalTree())}")
        print("---")
        runtime, cost, f1_score, label = run_enron_pz_plan(datasetid, plan, idx)

        # add to results
        results.append((runtime, cost, f1_score, label))

        # workaround to disabling cache: delete all cached generations after each plan
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)

    return results


def evaluate_real_estate_pz(datasetid, reoptimize=False, limit=None):
    """
    This creates the PZ set of plans for the Real Estate evaluation.

    Make sure to pre-register the dataset with:

    $ pz reg --path testdata/real-estate-eval --name real-estate-eval
    """
    # TODO: we can expand this dataset, but it's good enough for now
    listings = pz.Dataset(datasetid, schema=RealEstateListing)
    listings = listings.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
    # listings = listings.filterByStr("The email is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name")
    listings = listings.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")

    logicalTree = listings.getLogicalTree()
    candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, shouldProfile=True)
    results = []
    for idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
        print("----------------------")
        print(f"Plan: {buildNestedStr(plan.dumpPhysicalTree())}")
        print("---")
        runtime, cost, f1_score, label = run_enron_pz_plan(datasetid, plan, idx)

        # add to results
        results.append((runtime, cost, f1_score, label))

        # workaround to disabling cache: delete all cached generations after each plan
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)

    return results

def plot_runtime_cost_vs_quality(results):
    # create figure
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    # parse results into fields
    for runtime, cost, f1_score, label in results:
        # set label and color
        color = None
        marker = "*" if "PZ" in label else "^"

        # plot runtime vs. f1_score
        axs[0].scatter(f1_score, runtime, label=label, alpha=0.4, color=color, marker=marker)

        # plot cost vs. f1_score
        axs[1].scatter(f1_score, cost, label=label, alpha=0.4, color=color, marker=marker)

    # savefig
    axs[0].set_title("Runtime and Cost vs. F1 Score")
    axs[0].set_ylabel("runtime (seconds)")
    axs[1].set_ylabel("cost (USD)")
    axs[1].set_xlabel("F1 Score")
    axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig.savefig("eval-results/enron.png", bbox_inches="tight")


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--eval' , type=str, help='The evaluation to run')
    parser.add_argument('--limit' , type=int, help='The number of plans to consider')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs("eval-results", exist_ok=True)

    # The user has to indicate the evaluation to be run
    if args.eval is None:
        print("Please provide an evaluation")
        exit(1)

    if args.eval == "enron":
        # get PZ plan metrics
        print("Running PZ Plans")
        print("----------------")
        results = evaluate_enron_pz(args.datasetid, limit=args.limit)

        # get baseline metrics
        print("Running Baselines")
        print("-----------------")
        all_gpt4_runtime, all_gpt4_cost, all_gpt4_quality, all_gpt4_label = evaluate_enron_baseline(Model.GPT_4, args.datasetid)
        # all_gpt35_runtime, all_gpt35_cost, all_gpt35_quality, all_gpt35_label = evaluate_enron_baseline(Model.GPT_3_5)
        all_mixtral_runtime, all_mixtral_cost, all_mixtral_quality, mixtral_label = evaluate_enron_baseline(Model.MIXTRAL, args.datasetid)

        # plot runtime vs quality and cost vs quality
        baselines = [
            (all_gpt4_runtime, all_gpt4_cost, all_gpt4_quality, all_gpt4_label),
            # (all_gpt35_runtime, all_gpt35_cost, all_gpt35_quality, all_gpt35_label),
            (all_mixtral_runtime, all_mixtral_cost, all_mixtral_quality, mixtral_label),
        ]
        pz_plans = [
            (runtime, cost, f1_score, label)
            for runtime, cost, f1_score, label in results
        ]
        all_results = baselines + pz_plans
        with open("eval-results/enron.json", 'w') as f:
            json.dump(all_results, f)

        plot_runtime_cost_vs_quality(all_results)

    if args.eval == "real-estate":
        # register user data source
        pz.DataDirectory().registerUserSource(GitHubCommitSource(datasetid), datasetid)