from palimpzest.constants import Model
from palimpzest.datamanager import DataDirectory
from palimpzest.operators import ConvertFromCandidateOp
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.profiler import StatsProcessor
from palimpzest.sets import Set

# for those of us who resist change and are still on Python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise

from concurrent.futures import ThreadPoolExecutor
from typing import List

import os
import shutil


def flatten_nested_tuples(nested_tuples):
    """
    This function takes a nested iterable of the form (4,(3,(2,(1,())))) and flattens it to (1, 2, 3, 4).
    """
    result = []

    def flatten(item):
        if isinstance(item, tuple):
            if item:  # Check if not an empty list
                flatten(item[0])  # Process the head
                flatten(item[1])  # Process the tail
        else:
            result.append(item)

    flatten(nested_tuples)
    result = list(result)
    result.reverse()
    return result[1:]


def graphicEmit(flatten_ops):
    start = flatten_ops[0]
    print(f" 0. {type(start).__name__} -> {start.outputSchema.__name__} \n")

    for idx, (left, right) in enumerate(pairwise(flatten_ops)):
        in_schema = left.outputSchema
        out_schema = right.outputSchema
        print(
            f" {idx+1}. {in_schema.__name__} -> {type(right).__name__} -> {out_schema.__name__} ",
            end="",
        )
        # if right.desc is not None:
        #     print(f" ({right.desc})", end="")
        # check if right has a model attribute
        if right.is_hardcoded():
            print(f"\n    Using hardcoded function", end="")
        elif hasattr(right, "model"):
            print(f"\n    Using {right.model}", end="")
            if hasattr(right, "filter"):
                filter_str = (
                    right.filter.filterCondition
                    if right.filter.filterCondition is not None
                    else str(right.filter.filterFn)
                )
                print(f'\n    Filter: "{filter_str}"', end="")
            if hasattr(right, "token_budget"):
                print(f"\n    Token budget: {right.token_budget}", end="")
            if hasattr(right, "query_strategy"):
                print(f"\n    Query strategy: {right.query_strategy}", end="")
        print()
        print(
            f"    ({','.join(in_schema.fieldNames())[:15]}...) -> ({','.join(out_schema.fieldNames())[:15]}...)"
        )
        print()


class Execute:
    """
    Class for executing plans w/sentinels as described in the paper. Will refactor in PZ 1.0.
    """

    @staticmethod
    def compute_label(physicalTree, label_idx):
        """
        Map integer to physical plan.
        """
        physicalOps = physicalTree.dumpPhysicalTree()
        flat = flatten_nested_tuples(physicalOps)
        ops = [op for op in flat if not op.is_hardcoded()]
        label = "-".join([
            f"{repr(op.model)}_{op.query_strategy if isinstance(op, ConvertFromCandidateOp) else None}_{op.token_budget if isinstance(op, ConvertFromCandidateOp) else None}"
            for op in ops
        ])
        return f"PZ-{label_idx}-{label}"

    @staticmethod
    def run_sentinel_plan(args_tuple):
        # parse input tuple
        dataset, plan_idx, num_samples, verbose = args_tuple

        # create logical plan from dataset
        logicalTree = dataset.getLogicalTree(num_samples=num_samples, nocache=True)

        # compute number of plans
        sentinel_plans = logicalTree.createPhysicalPlanCandidates(sentinels=True)
        plan = sentinel_plans[plan_idx]

        # display the plan output
        if verbose:
            print("----------------------")
            ops = plan.dumpPhysicalTree()
            flatten_ops = flatten_nested_tuples(ops)
            print(f"Sentinel Plan {plan_idx}:")
            graphicEmit(flatten_ops)
            print("---")

        # run the plan
        records = [r for r in plan]

        # get profiling data for plan and compute its cost
        profileData = plan.getProfilingData()
        sp = StatsProcessor(profileData)
        cost_estimate_sample_data = sp.get_cost_estimate_sample_data()

        plan_info = {
            "plan_idx": plan_idx,
            "plan_label": Execute.compute_label(plan, f"s{plan_idx}"),
            "models": [],
            "op_names": [],
            "generated_fields": [],
            "query_strategies": [],
            "token_budgets": [],
        }
        cost = 0.0
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
            "runtime": None,
            "cost": cost,
            "f1_score": None,
            "plan_info": plan_info,
        }

        return records, result_dict, cost_estimate_sample_data

    @classmethod
    def run_sentinel_plans(cls, dataset: Set, num_samples: int, verbose: bool = False):
        # create logical plan from dataset
        logicalTree = dataset.getLogicalTree(num_samples=num_samples, nocache=True)

        # compute number of plans
        sentinel_plans = logicalTree.createPhysicalPlanCandidates(sentinels=True)
        num_sentinel_plans = len(sentinel_plans)

        all_cost_estimate_data, return_records = [], []
        # with Pool(processes=num_sentinel_plans) as pool:
        #     results = pool.starmap(Execute.run_sentinel_plan, [(dataset, plan_idx, num_samples, verbose) for plan_idx in range(num_sentinel_plans)])
        with ThreadPoolExecutor(max_workers=num_sentinel_plans) as executor:
            results = list(
                executor.map(
                    Execute.run_sentinel_plan,
                    [
                        (dataset, plan_idx, num_samples, verbose)
                        for plan_idx in range(num_sentinel_plans)
                    ],
                )
            )

            # write out result dict and samples collected for each sentinel
            for records, result_dict, cost_est_sample_data in results:
                # aggregate sentinel est. data
                all_cost_estimate_data.extend(cost_est_sample_data)

                # TODO: turn into utility function in proper utils file
                # set return_records to be records from champion model
                champion_model = None
                if os.environ.get("OPENAI_API_KEY", None) is not None:
                    champion_model = Model.GPT_4.value
                elif os.environ.get("TOGETHER_API_KEY", None) is not None:
                    champion_model = Model.MIXTRAL.value
                elif os.environ.get("GOOGLE_API_KEY", None) is not None:
                    champion_model = Model.GEMINI_1.value
                else:
                    raise Exception(
                        "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"
                    )

                # find champion model plan records and add those to all_records
                if all(
                    [
                        model is None
                        or model in [champion_model, "gpt-4-vision-preview"]
                        for model in result_dict["plan_info"]["models"]
                    ]
                ) and any(
                    [
                        model == champion_model
                        for model in result_dict["plan_info"]["models"]
                    ]
                ):
                    return_records = records

        return all_cost_estimate_data, return_records

    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int = 20,
        nocache: bool = False,
        verbose: bool = False,
    ):
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if nocache:
            dspy_cache_dir = os.path.join(
                os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/"
            )
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)

            # remove codegen samples from previous dataset from cache
            cache = DataDirectory().getCacheService()
            cache.rmCache()

        # TODO: if nocache=False and there is a cached result; don't run sentinels
        # run sentinel plans
        all_cost_estimate_data, sentinel_records = cls.run_sentinel_plans(
            dataset, num_samples, verbose
        )

        # create new plan candidates based on current estimate data
        logicalTree = dataset.getLogicalTree(
            nocache=nocache, scan_start_idx=num_samples
        )
        candidatePlans = logicalTree.createPhysicalPlanCandidates(
            cost_estimate_sample_data=all_cost_estimate_data,
            allow_model_selection=True,
            allow_codegen=True,
            allow_token_reduction=True,
            pareto_optimal=True,
            shouldProfile=True,
        )

        # choose best plan and execute it
        (_, _, _, plan, _) = policy.choose(candidatePlans)

        # display the plan output
        if verbose:
            print("----------------------")
            ops = plan.dumpPhysicalTree()
            flatten_ops = flatten_nested_tuples(ops)
            print(f"Final Plan:")
            graphicEmit(flatten_ops)
            print("---")

        # run the plan
        new_records = [r for r in plan]
        all_records = sentinel_records + new_records

        return all_records, plan


class NewExecute:
    @staticmethod
    def get_champion_model():
        # TODO:? turn into utility function in proper utils file
        champion_model = None
        if os.environ.get("OPENAI_API_KEY", None) is not None:
            champion_model = Model.GPT_4.value
        elif os.environ.get("TOGETHER_API_KEY", None) is not None:
            champion_model = Model.MIXTRAL.value
        elif os.environ.get("GOOGLE_API_KEY", None) is not None:
            champion_model = Model.GEMINI_1.value
        else:
            raise Exception(
                "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"
            )

        return champion_model

    @staticmethod
    def run_sentinel_plan(args):
        # parse args
        plan, plan_idx, verbose = args

        # display the plan output
        if verbose:
            print("----------------------")
            ops = plan.dumpPhysicalTree()
            flatten_ops = flatten_nested_tuples(ops)
            print(f"Sentinel Plan {plan_idx}:")
            graphicEmit(flatten_ops)
            print("---")

        # run the plan
        records, _ = plan.execute()

        return records

    @classmethod
    def run_sentinel_plans(
        cls, sentinel_plans: List[PhysicalPlan], verbose: bool = False
    ):
        # compute number of plans
        num_sentinel_plans = len(sentinel_plans)

        all_sample_execution_data, return_records = [], []
        with ThreadPoolExecutor(max_workers=num_sentinel_plans) as executor:
            results = list(
                executor.map(
                    NewExecute.run_sentinel_plan,
                    [(plan, idx, verbose) for idx, plan in enumerate(sentinel_plans)],
                )
            )

            # write out result dict and samples collected for each sentinel
            for records, plan in zip(results, sentinel_plans):
                # aggregate sentinel est. data
                sample_execution_data = plan.getExecutionData()
                all_sample_execution_data.extend(sample_execution_data)

                # set return_records to be records from champion model
                champion_model = NewExecute.get_champion_model()

                # find champion model plan records and add those to all_records
                if champion_model in plan.getModels():
                    return_records = records

        return all_sample_execution_data, return_records # TODO: make sure you capture cost of sentinel plans.

    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int = 20,
        nocache: bool = False,
        verbose: bool = False,
    ):
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if nocache:
            dspy_cache_dir = os.path.join(
                os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/"
            )
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)

            # remove codegen samples from previous dataset from cache
            cache = DataDirectory().getCacheService()
            cache.rmCache()

        # only run sentinels if there isn't a cached result already
        uid = (
            dataset.universalIdentifier()
        )  # TODO: I think we may need to get uid from source?
        run_sentinels = nocache or not DataDirectory().hasCachedAnswer(uid)

        all_sample_execution_data, sentinel_records = None, None
        if run_sentinels:
            # initialize logical and physical planner
            logical_planner = LogicalPlanner(nocache)
            physical_planner = PhysicalPlanner(num_samples, scan_start_idx=0)

            # get sentinel plans
            sentinel_plans = []
            for logical_plan in logical_planner.generate_plans(dataset, sentinels=True):
                for sentinel_plan in physical_planner.generate_plans(
                    logical_plan, sentinels=True
                ):
                    sentinel_plans.append(sentinel_plan)

            # run sentinel plans
            all_sample_execution_data, sentinel_records = cls.run_sentinel_plans(
                sentinel_plans, verbose
            )

        # (re-)initialize logical and physical planner
        scan_start_idx = num_samples if run_sentinels else 0
        logicalPlanner = LogicalPlanner(nocache)
        physicalPlanner = PhysicalPlanner(
            scan_start_idx=scan_start_idx,
            sample_execution_data=all_sample_execution_data,
        )

        # create all possible physical plans
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)

        # TODO
        # compute per-operator estimates of runtime, cost, and quality
        operator_estimates = physical_planner.compute_operator_estimates()

        # TODO
        # estimate the cost of each plan
        plans = physical_planner.estimate_plan_costs(all_physical_plans, operator_estimates)

        # choose best plan and execute it
        (_, _, _, plan, _) = policy.choose(plans)

        # display the plan output
        if verbose:
            print("----------------------")
            ops = plan.dumpPhysicalTree()
            flatten_ops = flatten_nested_tuples(ops)
            print(f"Final Plan:")
            graphicEmit(flatten_ops)
            print("---")

        # run the plan
        new_records, stats = plan.execute()
        # new_records = []
        # source, phys_operators = plan[0], plan[1:] # TODO: maybe add __iter__ to plan to iterate over source records
        # for record in source:
        #     for phy_op in phys_operators:
        #         instantiated_op = phy_op()
        #         with Profiler: # or however the Stat collection works:
        #             record = instantiated_op(record)
        #     new_records.append(record)

        all_records = sentinel_records + new_records

        return all_records
