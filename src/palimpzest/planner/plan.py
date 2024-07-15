from __future__ import annotations

from palimpzest.operators import FilteredScan, LogicalOperator, PhysicalOperator
from palimpzest.operators.physical import PhysicalOperator

# backwards compatability for users who are still on Python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise # type: ignore

from typing import List



class Plan:
    """A generic Plan is a graph of nodes (#TODO a list for now).
    The main subclasses are a LogicalPlan, which is composed of logical Operators, and a PhysicalPlan, which is composed of physical Operators.
    Plans are typically generated by objects of class Planner, and consumed by several objects, e.g., Execution, CostEstimator, Optimizer, etc. etc.
    """

    operators = []

    def __init__(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.operators)

    def __next__(self):
        return next(iter(self.operators))

    def __len__(self):
        return len(self.operators)

    def __str__(self):
        if self.operators:
            return f"{self.__class__.__name__}:\n" + "\n".join(
                map(str, [f"{idx}. {str(op)}" for idx, op in enumerate(self.operators)])
            )
        else:
            return f"{self.__class__.__name__}: No operator tree."

class LogicalPlan(Plan):
    
    def __init__(self, 
                 operators: List[LogicalOperator] = [],
                 datasetIdentifier: str = None,):
        self.operators = operators
        self.datasetIdentifier = datasetIdentifier


    @staticmethod
    def fromOpsAndSubPlan(ops: List[LogicalOperator], subPlan: LogicalPlan) -> LogicalPlan:
        # create copies of all logical operators
        copySubPlan = [op.copy() for op in subPlan.operators]
        copyOps = [op.copy() for op in ops]

        # construct full set of operators
        copySubPlan.extend(copyOps)
        fullOperators = copySubPlan
        # make input and output schemas internally consistent
        for idx, op in enumerate(fullOperators):
            # if this op is a filter, set its outputSchema equal to its inputSchema
            if isinstance(op, FilteredScan):
                op.outputSchema = op.inputSchema

            # set next op's inputSchema to be this op's outputSchema
            if idx + 1 < len(fullOperators):
                nextOp = fullOperators[idx + 1]
                nextOp.inputSchema = op.outputSchema

        # return the LogicalPlan
        return LogicalPlan(fullOperators, subPlan.datasetIdentifier)


class PhysicalPlan(Plan):

    def __init__(self, 
                 operators: List[PhysicalOperator],
                 datasetIdentifier: str = None,):
        self.operators = operators
        self.total_time = None
        self.total_cost = None
        self.quality = None
        self.datasetIdentifier = datasetIdentifier
        # MR: Maybe we should keep the plan stats tied to the plan object for easy reference?
        #     Even if it's just by adding a setter method which pz.Execute can call after it's
        #     finished doing it's thing? I'm not sure what the use-case will be, but I imagine
        #     some day in the future when people are doing crazy things w/PZ, folks may want to
        #     just reference the plan.plan_stats (or just plan.stats) in their program.
        # self.plan_stats = PlanStats(plan_id=self.plan_id())

    @staticmethod
    def fromOpsAndSubPlan(ops: List[PhysicalOperator], subPlan: PhysicalPlan) -> PhysicalPlan:
        # create copies of all logical operators
        copySubPlan = [op.copy() for op in subPlan.operators]
        copyOps = [op.copy() for op in ops]

        # construct full set of operators
        copySubPlan.extend(copyOps)

        # return the PhysicalPlan
        return PhysicalPlan(operators=copySubPlan, datasetIdentifier=subPlan.datasetIdentifier)

    def __repr__(self) -> str:
        """Computes a string representation for this plan."""
        ops = [op for op in self.operators if not op.is_hardcoded()]
        label = "-".join([str(op) for op in ops])
        return f"PZ-{label}"

    # GV: Should we generate a unique ID for each plan in the __init__ ?
    # MR: I think that's a good idea (also in fromOpsAndSubPlan)
    def plan_id(self) -> str:
        return self.__repr__()

    def getPlanModelNames(self) -> List[str]:
        model_names = []
        for op in self.operators:
            model = getattr(op, "model", None)
            if model is not None:
                model_names.append(model.value)

        return model_names

    def printPlan(self) -> None:
        """Print the physical plan."""
        print_ops = self.operators[1:]
        if len(print_ops) == 0:
            print("Empty plan: ", self.plan_id())
            return

        start = print_ops[0]
        print(f" 0. {start.inputSchema.__name__} -> {type(start).__name__} -> {start.outputSchema.__name__} \n")

        for idx, (left, right) in enumerate(pairwise(print_ops)):
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
                if hasattr(right, "prompt_strategy"):
                    print(f"\n    Prompt strategy: {right.prompt_strategy}", end="")
            print()
            print(
                f"    ({','.join(in_schema.fieldNames())[:15]}...) -> ({','.join(out_schema.fieldNames())[:15]}...)"
            )
            print()
