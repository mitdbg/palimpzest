#!/usr/bin/env python3
import context
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from tabulate import tabulate
from PIL import Image


from palimpzest.constants import Model
from palimpzest.execution import Execution, graphicEmit, flatten_nested_tuples
from palimpzest.elements import DataRecord, GroupBySig

import matplotlib.pyplot as plt
import pandas as pd

import argparse
import json
import shutil
import time
import os
import pandas as pd
import pdb
from sklearn.metrics import precision_recall_fscore_support

IN_DIR= "testdata/biofabric-matching/"
RESULT_PATH = "eval-results/biofabric/"

# def evaluate_matching():

class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""
    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False)
    ethnicity = pz.Field(desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False)
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
    tumor_grade = pz.Field(desc="The tumor grade", required=False)
    tumor_focality = pz.Field(desc="The tumor focality", required=False)
    tumor_largest_dimension_diameter = pz.Field(desc="The tumor largest dimension diameter", required=False)
    primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
    morphology = pz.Field(desc="The morphology", required=False)
    tissue_or_organ_of_origin = pz.Field(desc="The tissue or organ of origin", required=False)
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    filename = pz.Field(desc="The name of the file the record was extracted from", required=False)
    study = pz.Field(desc="The last name of the author of the study, from the table name", required=False)

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


def score_biofabric_plan(records) -> float:
    """
    Computes the F1 score of the biofabric plan
    """
    # parse records
    output_rows = []
    for rec in records:
        dct = rec.asDict()
        output_rows.append(dct) 
    records_df = pd.DataFrame(output_rows)

    if records_df.empty:
        return 0.0, 0.0, 0.0

    output = records_df
    index = [x for x in output.columns if x != "study"]
    target_matching = pd.read_csv(os.path.join(RESULT_PATH, "target_matching.csv"), index_col=0).reindex(index)

    studies = output["study"].unique()
    # Group by output by the "study" column and split it into many dataframes indexed by the "study" column
    df = pd.DataFrame(columns=target_matching.columns, index = index)
    cols = output.columns
    predicted = []
    targets = []

    for study in studies:
        output_study = output[output["study"] == study]
        study = study.split(".xlsx")[0]
        try:
            input_df = pd.read_excel(os.path.join(IN_DIR, f"{study}.xlsx"))
        except:
            targets += [study]*5 
            predicted += ["missing"]*5
            continue
        # for every column in output_study, check which column in input_df is the closest, i.e. the one with the highest number of matching values
        for col in cols:
            if col == "study":
                continue
            max_matches = 0
            max_col = "missing"
            for input_col in input_df.columns:
                try:
                    matches = sum([1 for idx,x in enumerate(output_study[col]) if x == input_df[input_col]
                    [idx]])
                except:
                    pdb.set_trace()
                if matches > max_matches:
                    max_matches = matches
                    max_col = input_col
            df.loc[col, study] = max_col

            # build a matrix that has the study on the columns and the predicted column names on the rows
        df.fillna("missing", inplace=True)

        targets += list(target_matching[study].values)
        predicted += list(df[study].values)

    # print(df)
    # get groundtruth
    p,r,f1,sup = precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)
    
    return p, r, f1



def stub_scores(records_df):

    if records_df.empty:
        return 0.0, 0.0, 0.0

    output = records_df
    index = [x for x in output.columns if x != "study"]
    target_matching = pd.read_csv(os.path.join(RESULT_PATH, "target_matching.csv"), index_col=0).reindex(index)

    studies = output["study"].unique()
    # Group by output by the "study" column and split it into many dataframes indexed by the "study" column
    df = pd.DataFrame(columns=target_matching.columns, index = index)
    cols = output.columns
    predicted = []
    targets = []

    for study in studies:
        output_study = output[output["study"] == study]
        study = study.split(".xlsx")[0]
        try:
            input_df = pd.read_excel(os.path.join(IN_DIR, f"{study}.xlsx"))
        except:
            targets += [study]*5 
            predicted += ["missing"]*5
            continue
        # for every column in output_study, check which column in input_df is the closest, i.e. the one with the highest number of matching values
        for col in cols:
            if col == "study":
                continue
            max_matches = 0
            max_col = "missing"
            for input_col in input_df.columns:
                try:
                    matches = sum([1 for idx,x in enumerate(output_study[col]) if x == input_df[input_col]
                    [idx]])
                except:
                    pdb.set_trace()
                if matches > max_matches:
                    max_matches = matches
                    max_col = input_col
            df.loc[col, study] = max_col

            # build a matrix that has the study on the columns and the predicted column names on the rows
        df.fillna("missing", inplace=True)

        targets += list(target_matching[study].values)
        predicted += list(df[study].values)

    # print(df)
    # get groundtruth
    p,r,f1,sup = precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)

    print(f"Precision {p} recall {r} F1 {f1}")

def evaluate_biofabric_baseline(model, datasetid):
    """
    Perform single shot evaluation with the given model
    """
    print("----------------------")
    print(f"Model: {model.value}")
    print("---")
    # construct generator
    doc_schema = str(CaseData)
    doc_type = CaseData.className()
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
        The file meets the following criteria:
        1. The rows of the table contain the patient age
        """
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
    _, _, f1_score = score_biofabric_plan(datasetid, output_records)

    # compute label and return
    label = None
    if model == Model.GPT_4:
        label = "GPT-4"
    elif model == Model.GPT_3_5:
        label = "GPT-3.5"
    elif model == Model.MIXTRAL:
        label = "MIXTRAL-7B"
    elif model == Model.GEMINI_1:
        label = "GEMINI-1"

    return runtime, cost, f1_score, label

def evaluate_biofabric_pz(datasetid, reoptimize=False, limit=None):
    """
    This creates the PZ set of plans
    """

    records_df = pd.read_csv("eval-results/biofabric/clean_output.csv")
    stub_scores(records_df)

    # pz reg --name biofabric-matching-tiny --path testdata/biofabric-matching-tiny
    # xls = pz.Dataset('biofabric-matching-tiny', schema=pz.XLSFile)
    xls = pz.Dataset(datasetid, schema=pz.XLSFile)
    patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
    patient_tables = patient_tables.filterByStr("The rows of the table contain the patient age")
    case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")

    logicalTree = case_data.getLogicalTree()
    candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, shouldProfile=True)
    results = []
    for idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
        print("----------------------")
        ops = plan.dumpPhysicalTree()
        flatten_ops = flatten_nested_tuples(ops)
        print(f"Plan: {graphicEmit(flatten_ops)}")
        print("---")

        # Running the plan
        start_time = time.time()
        records = [r for r in plan]
        runtime = time.time() - start_time
        profileData = plan.getProfilingData()

        sp = StatsProcessor(profileData)
        # TODO profiling data contains arrays ?
        # with open(f'eval-results/biofabric/profiling-{idx}.json', 'w') as f:
            # json.dump(sp.profiling_data.to_dict(), f)

        # score plan based on its output records
        p, r, f1_score = score_biofabric_plan(records)

        cost = 0.0
        stats = sp.profiling_data
        while stats is not None:
            cost += stats.total_usd
            stats = stats.source_op_stats

        # compute label
        label = compute_label(plan, idx)
        results.append((runtime, cost, f1_score, label))

        # workaround to disabling cache: delete all cached generations after each plan
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)

    return results


def plot_runtime_cost_vs_quality(results):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    for runtime, cost, f1_score, label in results:
        color = None
        marker = "*" if "PZ" in label else "^"
        # plot runtime vs. f1_score
        axs[0].scatter(f1_score, runtime, label=label, alpha=0.4, color=color, marker=marker)
        # plot cost vs. f1_score
        axs[1].scatter(f1_score, cost, label=label, alpha=0.4, color=color, marker=marker)

    axs[0].set_title("Runtime and Cost vs. F1 Score")
    axs[0].set_ylabel("runtime (seconds)")
    axs[1].set_ylabel("cost (USD)")
    axs[1].set_xlabel("F1 Score")
    axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig.savefig("eval-results/biofabric/cost-quality.png", bbox_inches="tight")


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--limit' , type=int, help='The number of plans to consider')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached results', default=False)

    args = parser.parse_args()
    no_cache = args.no_cache
    datasetid = args.datasetid

    if no_cache:
        pz.DataDirectory().clearCache(keep_registry=True)

    datasetid = "biofabric-tiny"
    # get PZ plan metrics
    print("Running PZ Plans")
    print("----------------")
    results = evaluate_biofabric_pz(datasetid, limit=args.limit)

    datasetid = "biofabric-tiny-csv"
    # get baseline metrics
    print("Running Baselines")
    print("-----------------")
    # all_gpt4_runtime, all_gpt4_cost, all_gpt4_quality, all_gpt4_label = evaluate_biofabric_baseline(Model.GPT_4, datasetid)
    # all_gpt35_runtime, all_gpt35_cost, all_gpt35_quality, all_gpt35_label = evaluate_biofabric_baseline(Model.GPT_3_5, datasetid)
    # all_gemini_runtime, all_gemini_cost, all_gemini_quality, gemini_label = evaluate_biofabric_baseline(Model.GEMINI_1, datasetid)

    # plot runtime vs quality and cost vs quality
    # baselines = [
        # (all_gpt4_runtime, all_gpt4_cost, all_gpt4_quality, all_gpt4_label),
        # (all_gpt35_runtime, all_gpt35_cost, all_gpt35_quality, all_gpt35_label),
        # (all_gemini_runtime, all_gemini_cost, all_gemini_quality, gemini_label),
    # ]
    baselines = []
    pz_plans = [
        (runtime, cost, f1_score, label)
        for runtime, cost, f1_score, label in results
    ]
    all_results = baselines + pz_plans
    with open("eval-results/biofabric.json", 'w') as f:
        json.dump(all_results, f)

    plot_runtime_cost_vs_quality(all_results)