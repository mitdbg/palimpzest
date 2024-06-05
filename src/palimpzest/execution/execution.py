from palimpzest.constants import Model
from palimpzest.datamanager import DataDirectory
from palimpzest.operators import InduceFromCandidateOp
from palimpzest.policy import Policy, MaxQuality, UserChoice
from palimpzest.profiler import StatsProcessor
from palimpzest.sets import Set

# for those of us who resist change and are still on Python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise

def emitNestedTuple(node, indent=0):
    elt, child = node
    print(" " * indent, elt)
    if child is not None:
        emitNestedTuple(child, indent=indent+2)

import os
import concurrent
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
        print(f" {idx+1}. {in_schema.__name__} -> {type(right).__name__} -> {out_schema.__name__} ", end="")
        # if right.desc is not None:
        #     print(f" ({right.desc})", end="")
        # check if right has a model attribute
        if right.is_hardcoded():
            print(f"\n    Using hardcoded function", end="")
        elif hasattr(right, 'model'):
            print(f"\n    Using {right.model}", end="")
            if hasattr(right, 'filter'):
                filter_str = right.filter.filterCondition if right.filter.filterCondition is not None else str(right.filter.filterFn)
                print(f'\n    Filter: "{filter_str}"', end="")
            if hasattr(right, 'token_budget'):
                print(f'\n    Token budget: {right.token_budget}', end="")
            if hasattr(right, 'query_strategy'):
                print(f'\n    Query strategy: {right.query_strategy}', end="")
        print()
        print(f"    ({','.join(in_schema.fieldNames())[:15]}...) -> ({','.join(out_schema.fieldNames())[:15]}...)")
        print()

class Execution:
    """An Execution is responsible for completing a query on a given Set.
    Right now we assume the query is always SELECT * FROM set."""
    def __init__(self, rootset: Set, policy: Policy) -> None:
        self.rootset = rootset
        self.policy = policy

    def executeAndOptimize(self, verbose: bool=False, shouldProfile: bool=False):
        """An execution of the rootset, subject to user-given policy."""
        logicalTree = self.rootset.getLogicalTree()

        # Crude algorithm:
        # 1. Generate a logical plan
        # 2. If there's no previously-created reliable data, then get some via sampling on a high-quality plan
        # 3. Once the sample is available, either (1) declare victory because the work is done, or (2) 
        #    use the sample to reoptimize according to user preferences

        cache = DataDirectory().getCacheService()
        cachedInfo = cache.getCachedData("querySamples", self.rootset.universalIdentifier())
        if cachedInfo is not None:
            sampleOutputs, profileData = cachedInfo
        else:
            # Obtain a sample of the first 'sampleSize' records.
            # We need the output examples as well as the performance data.
            sampleSize = 4
            limitSet = self.rootset.limit(sampleSize)
            logicalTree = limitSet.getLogicalTree()
            candidatePlans = logicalTree.createPhysicalPlanCandidates(shouldProfile=True)
            if verbose:
                print("----- PRE-SAMPLE PLANS -----")
                for idx, cp in enumerate(candidatePlans):
                    print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
                    print("Physical operator tree")
                    physicalOps = cp[3].dumpPhysicalTree()
                    emitNestedTuple(physicalOps)
                    print("----------")

            planTime, planCost, quality, physicalTree, _ = MaxQuality().choose(candidatePlans)

            if verbose:
                print("----------")
                print(f"Policy is: Maximum Quality")
                print(f"Chose plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
                emitNestedTuple(physicalTree.dumpPhysicalTree())

            # Execute the physical plan and cache the results
            sampleOutputs = [r for r in physicalTree]
            profileData = physicalTree.getProfilingData()

            # We put this into an ephemeral cache
            cache.putCachedData("querySamples", self.rootset.universalIdentifier(), (sampleOutputs, profileData))

        # TODO: remove
        if verbose:
            import json
            import os
            if not os.path.exists('profiling-data'):
                os.makedirs('profiling-data')
            with open('profiling-data/eo-raw_profiling.json', 'w') as f:
                sp = StatsProcessor(profileData)
                json.dump(sp.profiling_data.to_dict(), f)

        # process profileData with StatsProcessor
        sp = StatsProcessor(profileData)
        cost_estimate_sample_data = sp.get_cost_estimate_sample_data()

        # TODO: remove
        if verbose:
            import json
            with open('profiling-data/eo-cost-estimate.json', 'w') as f:
                json.dump(cost_estimate_sample_data, f)

        # Ok now reoptimize the logical plan, this time with the sample data.
        # (The data is not currently being used; let's see if this method can work first)
        logicalTree = self.rootset.getLogicalTree()
        candidatePlans = logicalTree.createPhysicalPlanCandidates(cost_estimate_sample_data=cost_estimate_sample_data, shouldProfile=shouldProfile)
        if type(self.policy) == UserChoice or verbose:
            print("----- POST-SAMPLE PLANS -----")
            for idx, cp in enumerate(candidatePlans):
                print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
                print("Physical operator tree")
                physicalOps = cp[3].dumpPhysicalTree()
                emitNestedTuple(physicalOps)
                print("----------")

        planTime, planCost, quality, physicalTree, _ = self.policy.choose(candidatePlans)

        if verbose:
            print("----------")
            print(f"Policy is: {self.policy}")
            print(f"Chose plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
            emitNestedTuple(physicalTree.dumpPhysicalTree())

        return physicalTree


class SamplePlansExecution(Execution):

    def __init__(self, rootset: Set, policy: Policy, n_plans: int) -> None:
        self.rootset = rootset
        self.policy = policy
        self.n_plans = n_plans

    def executeAndOptimize(self, verbose: bool=False, shouldProfile: bool=False):
        """An execution of the rootset, subject to user-given policy."""
        logicalTree = self.rootset.getLogicalTree()

        # TODO:
        # 1. enumerate plans
        # 2. select N to run for k samples
        # 3. gather execution data
        # 4. provide all cost estimates to physical operators

class SimpleExecution(Execution):
    """
    This simple execution does not pre-sample the data and does not use cache nor profiling.
    It just runs the query and returns the results.
    """

    def executeAndOptimize(self, verbose: bool=False, shouldProfile: bool=False):
        """An execution of the rootset, subject to user-given policy."""
        logicalTree = self.rootset.getLogicalTree()

        # Ok now reoptimize the logical plan, this time with the sample data.
        # (The data is not currently being used; let's see if this method can work first)
        logicalTree = self.rootset.getLogicalTree()
        candidatePlans = logicalTree.createPhysicalPlanCandidates(shouldProfile=shouldProfile)
        if type(self.policy) == UserChoice:
            print("-----AVAILABLE PLANS -----")
            for idx, cp in enumerate(candidatePlans):
                print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
                print("Physical operator tree")
                physicalOps = cp[3].dumpPhysicalTree()
                emitNestedTuple(physicalOps)
                print("----------")
                # flatten_ops = flatten_nested_tuples(physicalOps)               
                # graphicEmit(flatten_ops)
                # print("----------")

        planTime, planCost, quality, physicalTree, _ = self.policy.choose(candidatePlans)

        if verbose:
            print("----------")
            print(f"Policy is: {self.policy}")
            print(f"Chosen plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
            ops = physicalTree.dumpPhysicalTree()
            flatten_ops = flatten_nested_tuples(ops)
            graphicEmit(flatten_ops)
            # emitNestedTuple(ops)

        return physicalTree


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
            f"{repr(op.model)}_{op.query_strategy if isinstance(op, InduceFromCandidateOp) else None}_{op.token_budget if isinstance(op, InduceFromCandidateOp) else None}"
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
            "token_budgets": []
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
    def run_sentinel_plans(cls, dataset: Set, num_samples: int, verbose: bool=False):
        # create logical plan from dataset
        logicalTree = dataset.getLogicalTree(num_samples=num_samples, nocache=True)

        # compute number of plans
        sentinel_plans = logicalTree.createPhysicalPlanCandidates(sentinels=True)
        num_sentinel_plans = len(sentinel_plans)

        all_cost_estimate_data, return_records = [], []
        # with Pool(processes=num_sentinel_plans) as pool:
        #     results = pool.starmap(Execute.run_sentinel_plan, [(dataset, plan_idx, num_samples, verbose) for plan_idx in range(num_sentinel_plans)])
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sentinel_plans) as executor:
            results = list(executor.map(Execute.run_sentinel_plan, [(dataset, plan_idx, num_samples, verbose) for plan_idx in range(num_sentinel_plans)]))

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
                    raise Exception("No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]")

                # find champion model plan records and add those to all_records
                if (
                    all([model is None or model in [champion_model, "gpt-4-vision-preview"] for model in result_dict['plan_info']['models']])
                    and any([model == champion_model for model in result_dict['plan_info']['models']])
                ):
                    return_records = records

        return all_cost_estimate_data, return_records


    def __new__(cls, dataset: Set, policy: Policy, num_samples: int=20, nocache: bool=False, verbose: bool=False):
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if nocache:
            dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)
            
            # remove codegen samples from previous dataset from cache
            cache = DataDirectory().getCacheService()
            cache.rmCache()

        # TODO: if nocache=False and there is a cached result; don't run sentinels
        # run sentinel plans
        all_cost_estimate_data, sentinel_records = cls.run_sentinel_plans(dataset, num_samples, verbose)

        # create new plan candidates based on current estimate data
        logicalTree = dataset.getLogicalTree(nocache=nocache, scan_start_idx=num_samples)
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
    def run_sentinel_plan(plan):
        # run the plan
        records = [r for r in plan]

        # get profiling data for plan and compute its cost
        profileData = plan.getStats()
        sp = StatsProcessor(profileData)
        cost_estimate_sample_data = sp.get_cost_estimate_sample_data()

        return records, cost_estimate_sample_data


    def __new__(cls, dataset: Set, policy: Policy, num_samples: int=20, nocache: bool=False, verbose: bool=False):
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if nocache:
            # delete/disable DSPy cachedir
            # remove codegen samples from previous dataset from cache
            pass

        # get sentinel plans
        sentinelPlans = []
        logicalPlanner = LogicalPlanner(nocache, num_samples, scan_start_idx=0, sentinels=True)
        logicalPlan = logicalPlanner.generate_plans(dataset):
        for sentinelPlan in PhysicalPlanner().generate_plans(logicalPlan, sentinel=True):
            sentinelPlans.append(sentinelPlan)

        # run sentinel plans
        all_cost_estimate_data, sentinel_records = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sentinelPlans)) as executor:
            results = list(executor.map(NewExecute.run_sentinel_plan, sentinelPlans))

            # write out result dict and samples collected for each sentinel
            for records, cost_est_sample_data in results:
                # aggregate sentinel est. data
                all_cost_estimate_data.extend(cost_est_sample_data)

                champion_model = None # compute champion model
                if champion_model:
                    sentinel_records = records

        # create new plan candidates based on current estimate data
        physicalPlanCandidates = []
        logicalPlanner = LogicalPlanner(nocache, scan_start_idx=num_samples)
        for logicalPlan in logicalPlanner.generate_plans(dataset):
            for physicalPlan in PhysicalPlanner().generate_plans(logicalPlan, cost_estimate_sample_data=all_cost_estimate_data):
                physicalPlanCandidates.append(physicalPlan)
    
        # choose best plan and execute it
        (_, _, _, plan, _) = policy.choose(physicalPlanCandidates)

        # run the plan
        new_records = [r for r in plan]
        all_records = sentinel_records + new_records

        return all_records
