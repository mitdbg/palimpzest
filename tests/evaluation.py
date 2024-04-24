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
class RealEstateListingFiles(pz.Schema):
    """The source text and image data for a real estate listing."""
    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.BytesField(desc="The content of the listing's text description", required=True)
    image_contents = pz.ListField(element_type=pz.BytesField, desc="A list of the contents of each image of the listing", required=True)

# TODO: longer-term we will want to support one or more of the following:
#       0. allow use of multimodal models on text + image inputs
#
#       1. allow users to define fields and specify which source fields they
#          should be converted from (e.g. text_content or image_contents);
#          PZ can then re-order these separate conversion steps with downstream
#          filters automatically to minimize execution cost
#      
class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""
    address = pz.StringField(desc="The address of the property")
    sq_ft = pz.NumericField(desc="The square footage (sq. ft.) of the property")
    year_built = pz.NumericField(desc="The year in which the property was built")
    bedrooms = pz.NumericField(desc="The number of bedrooms")
    bathrooms = pz.NumericField(desc="The number of bathrooms")

class FullRealEstateListing(TextRealEstateListing):
    """Represents a real estate listing with specific fields extracted from its text and images."""
    is_modern_and_attractive = pz.BooleanField(desc="True if the home interior is modern and attractive and False otherwise")
    has_natural_sunlight = pz.BooleanField(desc="True if the home interior has lots of natural sunlight and False otherwise")

class RealEstateListingSource(pz.UserSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
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


def score_plan(datasetid, records) -> float:
    """
    Computes the F1 score of the plan
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
    gt_df = None
    if datasetid == "enron-eval":
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval.csv")
    elif datasetid == "enron-eval-tiny":
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval-tiny.csv")
    elif datasetid == "real-estate-eval":
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval.csv")

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


def run_pz_plan(datasetid, plan, idx):
    """
    I'm placing this in a separate file from evaluate_pz_plans to see if this prevents
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

    with open(f'eval-results/{datasetid}-profiling-{idx}.json', 'w') as f:
        json.dump(sp.profiling_data.to_dict(), f)

    # score plan based on its output records
    _, _, f1_score = score_plan(datasetid, records)

    cost, models = 0.0, []
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        if stats.model_name is not None:
            models.append(stats.model_name)
        stats = stats.source_op_stats

    # compute label
    print(f"PLAN {idx}: {buildNestedStr(plan.dumpPhysicalTree())}")

    return runtime, cost, f1_score, models


def evaluate_pz_plans(datasetid, reoptimize=False, limit=None):
    """
    This creates the PZ set of plans for the Enron email evaluation.

    Make sure to pre-register the dataset(s) with:

    $ pz reg --path testdata/enron-eval --name enron-eval

    (Note that the real-estate dataset is registered dynamically.)
    """
    # TODO: we can expand these datasets, but they're good enough for now
    logicalTree = None
    if "enron" in datasetid:
        emails = pz.Dataset(datasetid, schema=Email)
        emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
        # emails = emails.filterByStr("The email is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name")
        emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")
        logicalTree = emails.getLogicalTree()

    elif "real-estate" in datasetid:
        def within_two_miles_of_mit(record):
            # NOTE: I'm using this hard-coded function so that folks w/out a
            #       Geocoding API key from google can still run this example
            far_away_addrs = ["Melcher St", "Sleeper St", "437 D St", "Seaport", "Liberty"]
            if any([street.lower() in record.address.lower() for street in far_away_addrs]):
                return False
            return True

        # TODO: update logical plan creation to consider swapping (pairs of) (convert and filter)
        listings = pz.Dataset(datasetid, schema=RealEstateListingFiles)
        listings = listings.convert(TextRealEstateListing)
        listings = listings.convert(FullRealEstateListing)
        listings = listings.filterByStr(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"]
        )
        listings = listings.filterByFn(within_two_miles_of_mit, depends_on=["address"])
        logicalTree = listings.getLogicalTree()

    # get total number of plans
    num_plans = len(logicalTree.createPhysicalPlanCandidates(max=limit, shouldProfile=True))

    results = []
    for idx in range(num_plans):
    # for idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
        # skip all-Gemini plan which opens too many files
        if idx == 17:
            continue

        # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
        candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, shouldProfile=True)
        _, _, _, plan = candidatePlans[idx]

        # workaround to disabling cache: delete all cached generations after each plan
        bad_files = ["testdata/enron-eval/assertion.log", "testdata/enron-eval/azure_openai_usage.log", "testdata/enron-eval/openai_usage.log"]
        for file in bad_files:
            if os.path.exists(file):
                os.remove(file)

        print("----------------------")
        print(f"Plan: {buildNestedStr(plan.dumpPhysicalTree())}")
        print("---")
        runtime, cost, f1_score, models = run_pz_plan(datasetid, plan, idx)

        # add to results
        result_dict = {"runtime": runtime, "cost": cost, "f1_score": f1_score, "models": models}
        results.append(result_dict)
        with open(f'eval-results/{datasetid}-results-{idx}.json', 'w') as f:
            json.dump(result_dict, f)

        # workaround to disabling cache: delete all cached generations after each plan
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)

    return results



def plot_runtime_cost_vs_quality(results, datasetid):
    # create figure
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    # parse results into fields
    for result_dict in results:
        runtime = result_dict["runtime"]
        cost = result_dict["cost"]
        f1_score = result_dict["f1_score"]
        models = result_dict["models"]
        text = None
        if all([model == "gpt-4-0125-preview" for model in models]):
            # add text for ALL-GPT4
            text = "ALL-GPT4"
        elif all([model == "mistralai/Mixtral-8x7B-Instruct-v0.1" for model in models]):
            # add text for ALL-MIXTRAL
            text = "ALL-MIXTRAL"
        elif datasetid == "enron-eval" and models == ["gpt-4-0125-preview"] * 2 + ["mistralai/Mixtral-8x7B-Instruct-v0.1"]:
            # add text for Mixtral-GPT4
            text = "MIXTRAL-GPT4"
        elif datasetid == "enron-eval" and models == ["gpt-4-0125-preview"] * 2 + ["gemini-1.0-pro-001"]:
            # add text for Gemini-GPT4
            text = "GEMINI-GPT4"

        # set label and color
        color = None
        marker = None
        # marker = "*" if "PZ" in label else "^"

        # plot runtime vs. f1_score
        axs[0].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker)
        if text is not None:
            axs[0].annotate(text, (f1_score, runtime))

        # plot cost vs. f1_score
        axs[1].scatter(f1_score, cost, alpha=0.4, color=color, marker=marker)
        if text is not None:
            axs[1].annotate(text, (f1_score, cost))

    # savefig
    axs[0].set_title("Runtime and Cost vs. F1 Score")
    axs[0].set_ylabel("runtime (seconds)")
    axs[1].set_ylabel("cost (USD)")
    axs[1].set_xlabel("F1 Score")
    # axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig.savefig("eval-results/enron.png", bbox_inches="tight")


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--eval' , type=str, help='The evaluation to run')
    parser.add_argument('--limit' , type=int, help='The number of plans to consider')
    parser.add_argument('--listings-dir', type=str, help='The directory with real-estate listings')

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
        results = evaluate_pz_plans(args.datasetid, limit=args.limit)

        # # get baseline metrics
        # print("Running Baselines")
        # print("-----------------")
        # all_gpt4_runtime, all_gpt4_cost, all_gpt4_quality, all_gpt4_label = evaluate_enron_baseline(Model.GPT_4, args.datasetid)
        # # all_gpt35_runtime, all_gpt35_cost, all_gpt35_quality, all_gpt35_label = evaluate_enron_baseline(Model.GPT_3_5)
        # all_mixtral_runtime, all_mixtral_cost, all_mixtral_quality, mixtral_label = evaluate_enron_baseline(Model.MIXTRAL, args.datasetid)

        # # plot runtime vs quality and cost vs quality
        # baselines = [
        #     (all_gpt4_runtime, all_gpt4_cost, all_gpt4_quality, all_gpt4_label),
        #     # (all_gpt35_runtime, all_gpt35_cost, all_gpt35_quality, all_gpt35_label),
        #     (all_mixtral_runtime, all_mixtral_cost, all_mixtral_quality, mixtral_label),
        # ]
        # pz_plans = [
        #     (runtime, cost, f1_score, label)
        #     for runtime, cost, f1_score, label in results
        # ]
        # all_results = baselines + pz_plans

        # results = []
        # for idx in range(17):
        #     with open(f'eval-results/{args.datasetid}-results-{idx}.json', 'r') as f:
        #         result_dict = json.load(f)
        #         results.append(result_dict)

        with open("eval-results/enron.json", 'w') as f:
            json.dump(results, f)
 
        plot_runtime_cost_vs_quality(results, args.datasetid)

    if args.eval == "real-estate":
        # register user data source
        print("Registering Datasource")
        pz.DataDirectory().registerUserSource(RealEstateListingSource(args.datasetid, args.listings_dir), args.datasetid)
        
        print("Running PZ plans")
        results = evaluate_pz_plans(args.datasetid, limit=args.limit)

        with open("eval-results/real-estate.json", 'w') as f:
            json.dump(results, f)

        plot_runtime_cost_vs_quality(results)
