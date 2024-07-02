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

def score_biofabric_plans(
    workload, records, plan_idx, policy_str=None, reopt=False
) -> float:
    """
    Computes the results of all biofabric plans
    """
    # parse records
    # exclude_keys = ["filename", "op_id", "uuid", "parent_uuid", "stats"]
    include_keys = [
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
        dct = {k: v for k, v in rec.items() if k in include_keys}
        # dct = {k:v for k,v in rec._asDict().items() if k not in exclude_keys}
        # filename = os.path.basename(rec._asDict()["filename"])
        dct["study"] = os.path.basename(rec["filename"]).split("_")[0]
        output_rows.append(dct)

    records_df = pd.DataFrame(output_rows)
    if not reopt:
        records_df.to_csv(
            f"final-eval-results/{workload}/preds-{plan_idx}.csv", index=False
        )
    else:
        records_df.to_csv(
            f"final-eval-results/reoptimization/{workload}/{policy_str}.csv",
            index=False,
        )

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
    cols = output.columns
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
        for col in cols:
            if col == "study":
                continue
            max_matches = 0
            max_col = "missing"
            for input_col in input_df.columns:
                try:
                    matches = sum(
                        [
                            1
                            for idx, x in enumerate(output_study[col])
                            if x == input_df[input_col][idx]
                        ]
                    )
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

    records_df = pd.DataFrame(records)

    # save predictions for this plan
    if not reopt:
        records_df.to_csv(
            f"final-eval-results/{workload}/preds-{plan_idx}.csv", index=False
        )
    else:
        records_df.to_csv(
            f"final-eval-results/reoptimization/{workload}/{policy_str}.csv",
            index=False,
        )

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


def run_pz_plan(
    workload, plan, plan_idx, total_sentinel_cost, total_sentinel_time, sentinel_records
):
    """
    I'm placing this in a separate file from evaluate_pz_plans to see if this prevents
    an error where the DSPy calls to Gemini (and other models?) opens too many files.
    My hope is that placing this inside a separate function will cause the file descriptors
    to be cleaned up once the function returns.
    """
    # TODO: eventually get runtime from profiling data
    # execute plan to get records and runtime;
    start_time = time.time()
    new_records = [r for r in plan]
    runtime = total_sentinel_time + (time.time() - start_time)

    # parse new_records
    new_records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith("_") and key not in ["image_contents"]
        }
        for record in new_records
    ]
    all_records = sentinel_records + new_records

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)

    # TODO: debug profiling issue w/conventional query stats for per-field stats
    # with open(f'eval-results/{datasetid}-profiling-{idx}.json', 'w') as f:
    #     json.dump(sp.profiling_data.to_dict(), f)

    # score plan based on its output records
    f1_score = score_plan(workload, all_records, plan_idx)

    plan_info = {
        "plan_idx": plan_idx,
        "plan_label": compute_label(plan, plan_idx),
        "models": [],
        "op_names": [],
        "generated_fields": [],
        "query_strategies": [],
        "token_budgets": [],
    }
    cost = total_sentinel_cost
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        plan_info["models"].append(stats.model_name)
        plan_info["op_names"].append(stats.op_name)
        plan_info["generated_fields"].append(stats.generated_fields)
        plan_info["query_strategies"].append(stats.query_strategy)
        plan_info["token_budgets"].append(stats.token_budget)
        stats = stats.source_op_stats

    # construct and return result_dict
    result_dict = {
        "runtime": runtime,
        "cost": cost,
        "f1_score": f1_score,
        "plan_info": plan_info,
    }

    return result_dict

def evaluate_pz_plan(sentinel_data, workload, plan_idx):
    # if os.path.exists(f'final-eval-results/{opt}/{workload}/results-{plan_idx}.json'):
    #     return
    if os.path.exists(f"final-eval-results/{workload}/results-{plan_idx}.json"):
        return

    # unpack sentinel data
    (
        total_sentinel_cost,
        total_sentinel_time,
        all_cost_estimate_data,
        sentinel_records,
        num_samples,
    ) = sentinel_data

    # get logicalTree
    logicalTree = get_logical_tree(workload, nocache=True, scan_start_idx=num_samples)

    # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
    plans = logicalTree.createPhysicalPlanCandidates(
        min=20,
        cost_estimate_sample_data=all_cost_estimate_data,
        allow_model_selection=True,
        allow_codegen=True,
        allow_token_reduction=True,
        # pareto_optimal=False if opt in ["codegen", "token-reduction"] and workload == "enron" else True,
        pareto_optimal=True,
        include_baselines=True,
        shouldProfile=True,
    )
    _, _, _, plan, _ = plans[plan_idx]
    plan.setPlanIdx(plan_idx)

    # workaround to disabling cache: delete all cached generations after each plan
    bad_files = [
        "testdata/enron-eval/assertion.log",
        "testdata/enron-eval/azure_openai_usage.log",
        "testdata/enron-eval/openai_usage.log",
    ]
    for file in bad_files:
        if os.path.exists(file):
            os.remove(file)

    # display the plan output
    print("----------------------")
    ops = plan.dumpPhysicalTree()
    flatten_ops = flatten_nested_tuples(ops)
    print(f"Plan {plan_idx}:")
    graphicEmit(flatten_ops)
    print("---")

    # run the plan
    result_dict = run_pz_plan(
        workload,
        plan,
        plan_idx,
        total_sentinel_cost,
        total_sentinel_time,
        sentinel_records,
    )

    # write result json object
    with open(f"final-eval-results/{workload}/results-{plan_idx}.json", "w") as f:
        json.dump(result_dict, f)


def run_reoptimize_eval(workload, policy_str, parallel: bool = False):
    workload_to_fixed_cost = {
        "enron": 20.0,
        "real-estate": 3.0,
        "biofabric": 2.0,
    }
    workload_to_fixed_runtime = {
        "enron": 10000,
        "real-estate": 600,
        "biofabric": 1000,
    }
    workload_to_fixed_quality = {
        "enron": 0.8,
        "real-estate": 0.8,
        "biofabric": 0.40,
    }

    policy = pz.MaxHarmonicMean()
    if policy_str is not None:
        if policy_str == "max-quality-at-fixed-cost":
            policy = pz.MaxQualityAtFixedCost(
                fixed_cost=workload_to_fixed_cost[workload]
            )
        elif policy_str == "max-quality-at-fixed-runtime":
            policy = pz.MaxQualityAtFixedRuntime(
                fixed_runtime=workload_to_fixed_runtime[workload]
            )
        elif policy_str == "min-runtime-at-fixed-quality":
            policy = pz.MinRuntimeAtFixedQuality(
                fixed_quality=workload_to_fixed_quality[workload]
            )
        elif policy_str == "min-cost-at-fixed-quality":
            policy = pz.MinCostAtFixedQuality(
                fixed_quality=workload_to_fixed_quality[workload]
            )

    # TODO: in practice could move this inside of get_logical_tree w/flag indicating sentinel run;
    #       for now just manually set to make sure evaluation is accurate
    # set samples and size of dataset
    workload_to_dataset_size = {"enron": 1000, "real-estate": 100, "biofabric": 11}
    dataset_size = workload_to_dataset_size[workload]
    num_samples = int(0.05 * dataset_size) if workload != "biofabric" else 1

    # run sentinels
    start_time = time.time()
    output = run_sentinel_plans(
        workload, num_samples, policy_str=policy_str, parallel=parallel
    )
    total_sentinel_cost, _, all_cost_estimate_data, sentinel_records = output

    # # get cost estimates given current candidate plans
    # for plan_idx in range(num_plans):
    #     totalTimeInitEst, totalCostInitEst, qualityInitEst, plan, fullPlanCostEst = candidatePlans[plan_idx]

    #     models = get_models_from_physical_plan(plan)
    #     result_dict = {
    #         "plan_idx": plan_idx,
    #         "plan_label": compute_label(plan, plan_idx),
    #         "runtime": totalTimeInitEst,
    #         "cost": totalCostInitEst,
    #         "quality": qualityInitEst,
    #         "models": models,
    #         "full_plan_cost_est": fullPlanCostEst,
    #     }
    #     estimates[f"estimate_{estimate_iter}"].append(result_dict)

    # with open(f"final-eval-results/reoptimization/{opt}/{workload}/estimates.json", 'w') as f:
    #     estimates = dict(estimates)
    #     json.dump(estimates, f)

    # create new plan candidates based on current estimate data
    logicalTree = get_logical_tree(workload, nocache=True, scan_start_idx=num_samples)
    candidatePlans = logicalTree.createPhysicalPlanCandidates(
        cost_estimate_sample_data=all_cost_estimate_data,
        allow_model_selection=True,
        allow_codegen=True,
        allow_token_reduction=True,
        pareto_optimal=True,
        shouldProfile=True,
    )

    # choose best plan and execute it
    (_, _, _, plan, _), plan_idx = policy.choose(candidatePlans, return_idx=True)

    # display the plan output
    print("----------------------")
    ops = plan.dumpPhysicalTree()
    flatten_ops = flatten_nested_tuples(ops)
    print(f"Final Plan:")
    graphicEmit(flatten_ops)
    print("---")

    # run the plan
    new_records = [r for r in plan]
    runtime = time.time() - start_time

    # parse new_records
    new_records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith("_") and key not in ["image_contents"]
        }
        for record in new_records
    ]
    all_records = sentinel_records + new_records

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)

    # workaround to disabling cache: delete all cached generations after each plan
    dspy_cache_dir = os.path.join(
        os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/"
    )
    if os.path.exists(dspy_cache_dir):
        shutil.rmtree(dspy_cache_dir)

    plan_info = {
        "plan_idx": None,
        "plan_label": compute_label(plan, plan_idx),
        "models": [],
        "op_names": [],
        "generated_fields": [],
        "query_strategies": [],
        "token_budgets": [],
    }
    cost = total_sentinel_cost
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        plan_info["models"].append(stats.model_name)
        plan_info["op_names"].append(stats.op_name)
        plan_info["generated_fields"].append(stats.generated_fields)
        plan_info["query_strategies"].append(stats.query_strategy)
        plan_info["token_budgets"].append(stats.token_budget)
        stats = stats.source_op_stats

    # score plan
    f1_score = score_plan(
        workload, all_records, None, policy_str=policy_str, reopt=True
    )

    # construct and return result_dict
    result_dict = {
        "runtime": runtime,
        "cost": cost,
        "f1_score": f1_score,
        "plan_info": plan_info,
    }

    fp = (
        f"final-eval-results/reoptimization/{workload}/{policy_str}.json"
        if not parallel
        else f"final-eval-results/reoptimization/{workload}/parallel-{policy_str}.json"
    )

    with open(fp, "w") as f:
        json.dump(result_dict, f)



@pytest.mark.parametrize("workload", ["biofabric", "enron", "real-estate"])
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
    workload_to_dataset_size = {"enron": 10, "real-estate": 5, "biofabric": 11}
    dataset_size = workload_to_dataset_size[workload]
    num_samples = int(0.05 * dataset_size) if workload != "biofabric" else 1

    available_models = getModels(include_vision=True)
    records, plan, stats= Execute(dataset, 
                                  policy=pz.MinCost(),
                                  available_models=available_models,
                                  num_samples=num_samples,
                                  nocache=True,
                                  allow_model_selection=True,
                                  allow_bonded_query=False,
                                  allow_code_synth=False,
                                  allow_token_reduction=False,
                                  execution_engine=SequentialSingleThreadExecution)
    
    import pdb; pdb.set_trace()
    # print(f"Plan: {result_dict['plan_info']['plan_label']}")
    # print(f"  F1: {result_dict['f1_score']}")
    # print(f"  rt: {result_dict['runtime']}")
    # print(f"  $$: {result_dict['cost']}")
    # print("---")


    # _ = pool.starmap(
    #     evaluate_pz_plan,
    #     [(sentinel_data, workload, plan_idx) for plan_idx in range(num_plans)],
    # )
