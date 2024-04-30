from palimpzest.sets import Set
from palimpzest.policy import Policy, MaxQuality, UserChoice
from palimpzest.profiler import StatsProcessor
from palimpzest.datamanager import DataDirectory
import itertools 

def emitNestedTuple(node, indent=0):
    elt, child = node
    print(" " * indent, elt)
    if child is not None:
        emitNestedTuple(child, indent=indent+2)


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

    for idx, (left, right) in enumerate(itertools.pairwise(flatten_ops)):
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
                print(f'\n    Filter: "{right.filter.filterCondition}"', end="")
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

