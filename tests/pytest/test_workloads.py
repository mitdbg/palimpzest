import os

import pandas as pd
import pytest
from sklearn.metrics import precision_recall_fscore_support

from palimpzest.execution.execute import Execute
from palimpzest.execution.nosentinel_execution import (
    NoSentinelPipelinedParallelExecution,
    NoSentinelPipelinedSingleThreadExecution,
    NoSentinelSequentialSingleThreadExecution,
)
from palimpzest.policy import MinCost
from palimpzest.utils.model_helpers import get_models


def score_biofabric_plans(dataset, records, policy_str=None, reopt=False) -> float:
    """
    Computes the results of all biofabric plans
    """
    # parse records
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
        dct = {k: v for k, v in rec.as_dict().items() if k in matching_columns}
        filename = os.path.basename(rec.as_dict()["filename"])
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
    target_matching = pd.read_csv(os.path.join("testdata/", "target_matching.csv"), index_col=0).reindex(index)

    studies = output["study"].unique()
    # Group by output by the "study" column and split it into many dataframes indexed by the "study" column
    df = pd.DataFrame(columns=target_matching.columns, index=index)
    predicted = []
    targets = []

    for study in studies:
        output_study = output[output["study"] == study]
        try:
            input_df = pd.read_excel(os.path.join("testdata/biofabric-matching/", f"{study}.xlsx"))
        except Exception:
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
    p, r, f1, sup = precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)

    return f1


def score_plan(dataset, records, policy_str=None, reopt=False) -> float:
    """
    Computes the F1 score of the plan
    """
    # special handling for biofabric dataset
    if "biofabric" in dataset:
        return score_biofabric_plans(dataset, records, policy_str, reopt)

    records_df = pd.DataFrame([rec.as_dict() for rec in records])

    # save predictions for this plan
    # if not reopt:
    #     records_df.to_csv(
    #         f"final-eval-results/{dataset}/preds-{plan_idx}.csv", index=False
    #     )
    # else:
    #     records_df.to_csv(
    #         f"final-eval-results/reoptimization/{dataset}/{policy_str}.csv",
    #         index=False,
    #     )

    if records_df.empty:
        return 0.0

    # get lists of predictions and groundtruth answers
    preds, targets = None, None
    if "enron" in dataset:
        preds = records_df.filename.apply(lambda fn: os.path.basename(fn)).tolist()
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval-tiny.csv")
        targets = list(gt_df[gt_df.label == 1].filename)
    elif "real-estate" in dataset:
        preds = list(records_df.listing)
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval-tiny.csv")
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
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return f1_score


@pytest.mark.parametrize(
    argnames=("execution_engine"),
    argvalues=[
        pytest.param(NoSentinelSequentialSingleThreadExecution, id="seq-single-thread"),
        pytest.param(NoSentinelPipelinedSingleThreadExecution, id="pipe-single-thread"),
        pytest.param(NoSentinelPipelinedParallelExecution, id="pipe-parallel"),
    ],
)
@pytest.mark.parametrize(
    argnames=("dataset", "workload"),
    argvalues=[
        ("real-estate-eval-tiny", "real-estate-workload"),
        ("biofabric-tiny", "biofabric-workload"),
        ("enron-eval-tiny", "enron-workload"),
    ],
    indirect=True,
)
def test_workload(dataset, workload, execution_engine):
    # workload_to_dataset_size = {"enron": 1000, "real-estate": 100, "biofabric": 11}
    dataset_to_size = {"enron-eval-tiny": 10, "real-estate-eval-tiny": 5, "biofabric-tiny": 3}
    dataset_size = dataset_to_size[dataset]
    num_samples = int(0.05 * dataset_size) if dataset != "biofabric-tiny" else 1

    available_models = get_models(include_vision=True)
    records, stats = Execute(
        workload,
        policy=MinCost(),
        available_models=available_models,
        num_samples=num_samples,
        nocache=True,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        execution_engine=execution_engine,
    )

    # NOTE: f1 score calculation will be low for biofabric b/c the
    #       evaluation function still checks against the full dataset's labels
    # print(f"Plan: {result_dict['plan_info']['plan_label']}")
    f1_score = score_plan(dataset=dataset, records=records)
    print(f"  F1: {f1_score}")
    print(f"  rt: {stats.total_execution_time}")
    print(f"  $$: {stats.total_execution_cost}")
