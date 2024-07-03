import pytest
from palimpzest.execution.execution import Execute, SequentialSingleThreadExecution
import palimpzest as pz

from palimpzest.utils.model_helpers import getModels
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import json
import shutil
import time
import os
import pdb

from palimpzest.datamanager.datamanager import DataDirectory

def score_biofabric_plans(workload, records, plan_idx, policy_str=None, reopt=False) -> float:
    """
    Computes the results of all biofabric plans
    """
    # parse records
    exclude_keys = ["op_id", "uuid", "parent_uuid", "stats"]
    matching_columns = [
        "age_at_diagnosis",
        "ajcc_pathologic_n",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "case_submitter_id",
        "ethnicity",
        "gender",
        "morphology",
        "primary_diagnosis",
        "race",
        "tissue_or_organ_of_origin",
        "tumor_focality",
        "tumor_grade",
        "tumor_largest_dimension_diameter",
        "vital_status",
    ]
    output_rows = []
    for rec in records:
        dct = {k:v for k,v in rec._asDict().items() if k in matching_columns}
        filename = os.path.basename(rec._asDict()["filename"])
        dct["study"] = os.path.basename(filename).split("_")[0]
        output_rows.append(dct)

    records_df = pd.DataFrame(output_rows)

    # if not reopt:
    #     records_df.to_csv(
    #         f"final-eval-results/{workload}/preds-{plan_idx}.csv", index=False
    #     )
    # else:
    #     records_df.to_csv(
    #         f"final-eval-results/reoptimization/{workload}/{policy_str}.csv",
    #         index=False,
    #     )

    if records_df.empty:
        return 0.0

    output = records_df
    index = [x for x in output.columns if x != "study"]
    # target_matching = pd.read_csv(os.path.join(f'final-eval-results/{opt}/{workload}/', "target_matching.csv"), index_col=0).reindex(index)
    target_matching = pd.read_csv(
        os.path.join(f"testdata/", "target_matching.csv"), index_col=0
    ).reindex(index)

    studies = output["study"].unique()
    # Group by output by the "study" column and split it into many dataframes indexed by the "study" column
    df = pd.DataFrame(columns=target_matching.columns, index=index)
    predicted = []
    targets = []

    for study in studies:
        output_study = output[output["study"] == study]
        try:
            input_df = pd.read_excel(
                os.path.join("testdata/biofabric-matching/", f"{study}.xlsx")
            )
        except:
            print("Cannot find the study", study)
            targets += [study] * 5
            predicted += ["missing"] * 5
            continue
        # for every column in output_study, check which column in input_df is the closest, i.e. the one with the highest number of matching values
        for col in matching_columns:
            max_matches = 0
            max_col = "missing"
            for input_col in input_df.columns:
                matches = sum([1 for idx, x in enumerate(output_study[col]) if x == input_df[input_col][idx]])
                if matches > max_matches:
                    max_matches = matches
                    max_col = input_col
            df.loc[col, study] = max_col

            # build a matrix that has the study on the columns and the predicted column names on the rows
        df.fillna("missing", inplace=True)

        targets += list(target_matching[study].values)
        predicted += list(df[study].values)

    # print(df)
    p, r, f1, sup = precision_recall_fscore_support(
        targets, predicted, average="micro", zero_division=0
    )

    return f1


def score_plan(workload, records, plan_idx, policy_str=None, reopt=False) -> float:
    """
    Computes the F1 score of the plan
    """
    # special handling for biofabric workload
    if workload == "biofabric":
        return score_biofabric_plans(workload, records, plan_idx, policy_str, reopt)

    records_df = pd.DataFrame([rec._asDict() for rec in records])

    # save predictions for this plan
    # if not reopt:
    #     records_df.to_csv(
    #         f"final-eval-results/{workload}/preds-{plan_idx}.csv", index=False
    #     )
    # else:
    #     records_df.to_csv(
    #         f"final-eval-results/reoptimization/{workload}/{policy_str}.csv",
    #         index=False,
    #     )

    if records_df.empty:
        return 0.0

    # get list of predictions
    preds = None
    if workload == "enron":
        preds = records_df.filename.apply(lambda fn: os.path.basename(fn)).tolist()
    elif workload == "real-estate":
        preds = list(records_df.listing)

    # get list of groundtruth answers
    targets = None
    if workload == "enron":
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval.csv")
        targets = list(gt_df[gt_df.label == 1].filename)
    elif workload == "real-estate":
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval-100.csv")
        targets = list(gt_df[gt_df.label == 1].listing)

    # compute true and false positives
    tp, fp = 0, 0
    for pred in preds:
        if pred in targets:
            tp += 1
        else:
            fp += 1

    # compute false negatives
    fn = 0
    for target in targets:
        if target not in preds:
            fn += 1

    # compute precision, recall, f1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return f1_score

@pytest.mark.parametrize("workload", ["real-estate","biofabric", "enron"])
def test_workload(workload, enron_eval, biofabric_eval, real_estate_eval):

    print("Testing on workload:", workload)
    if workload == "enron":
        dataset = enron_eval
    elif workload == "biofabric":
        dataset = biofabric_eval
    elif workload == "real-estate":
        dataset = real_estate_eval

    else:
        raise ValueError(f"Unknown workload: {workload}")
    
    # workload_to_dataset_size = {"enron": 1000, "real-estate": 100, "biofabric": 11}
    workload_to_dataset_size = {"enron": 10, "real-estate": 5, "biofabric": 3}
    dataset_size = workload_to_dataset_size[workload]
    num_samples = int(0.05 * dataset_size) if workload != "biofabric" else 1

    available_models = getModels(include_vision=True)
    records, plan, stats= Execute(dataset, 
                                  policy=pz.MinCost(),
                                  available_models=available_models,
                                  num_samples=num_samples,
                                  nocache=True,
                                  allow_model_selection=True,
                                  allow_bonded_query=True,
                                  allow_code_synth=False,
                                  allow_token_reduction=False,
                                  execution_engine=SequentialSingleThreadExecution)
    
    # print(f"Plan: {result_dict['plan_info']['plan_label']}")
    f1_score = score_plan(workload=workload, records=records, plan_idx=stats.plan_idx)
    print(f"  F1: {f1_score}")
    print(f"  rt: {stats.total_plan_time}")
    print(f"  $$: {stats.total_plan_cost}")
