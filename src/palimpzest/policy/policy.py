# from palimpzest.planner import legacy_PhysicalPlan as PhysicalPlan
from palimpzest.planner import PhysicalPlan
from typing import List, Tuple, Union

import numpy as np


class Policy:
    """
    Base class for policies that can choose a best plan from a set of
    candidate plans based on some selection criteria.
    """

    def __init__(self):
        pass

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        pass


class MaxQuality(Policy):
    """
    This policy selects the plan with the maximum quality along the
    pareto-optimal curve of candidate plans.
    """

    def __str__(self):
        return "Maximum Quality"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        return sorted(candidatePlans, key=lambda plan: plan.quality)[-1]


class MaxQualityAtFixedCost(Policy):
    def __init__(self, max_cost: float):
        self.max_cost = max_cost

    def __str__(self):
        return "MaxQuality@MinCost"

    def choose(
        self, candidatePlans: List[PhysicalPlan], return_idx: bool = False
    ) -> Union[PhysicalPlan, Tuple[PhysicalPlan, int]]:
        best_plan, best_plan_idx, max_quality, max_quality_runtime = None, -1, 0, np.inf
        for idx, plan in enumerate(candidatePlans):
            # if plan is too expensive, skip
            if plan.total_cost > self.max_cost:
                continue

            # if plan is above best current max quality, this is new best plan
            if plan.quality > max_quality:
                best_plan = plan
                best_plan_idx = idx
                max_quality = plan.quality
                max_quality_runtime = plan.total_time

            # if plan is tied w/current max quality -- and has lower runtime -- this is new best plan
            elif plan.quality == max_quality and plan.total_time < max_quality_runtime:
                best_plan = plan
                best_plan_idx = idx
                max_quality = plan.quality
                max_quality_runtime = plan.total_time

        # if no plan was below fixed cost; return cheapest plan
        if best_plan is None:
            print("NO PLAN FOUND BELOW FIXED COST; PICKING MIN. COST PLAN INSTEAD")
            min_cost = np.inf
            for idx, plan in enumerate(candidatePlans):
                if plan.total_cost < min_cost:
                    best_plan = plan
                    best_plan_idx = idx
                    min_cost = plan.total_cost

        return best_plan if not return_idx else (best_plan, best_plan_idx)


class MaxQualityAtFixedRuntime(Policy):
    def __init__(self, max_runtime: float):
        self.max_runtime = max_runtime

    def __str__(self):
        return "MaxQuality@MinRuntime"

    def choose(
        self, candidatePlans: List[PhysicalPlan], return_idx: bool = False
    ) -> Union[PhysicalPlan, Tuple[PhysicalPlan, int]]:
        best_plan, best_plan_idx, max_quality, max_quality_cost = None, -1, 0, np.inf
        for idx, plan in enumerate(candidatePlans):
            # if plan is too long, skip
            if plan.total_time > self.max_runtime:
                continue

            # if plan is above best current max quality, this is new best plan
            if plan.quality > max_quality:
                best_plan = plan
                best_plan_idx = idx
                max_quality = plan.quality
                max_quality_cost = plan.total_cost

            # if plan is tied w/current max quality -- and has lower cost -- this is new best plan
            elif plan.quality == max_quality and plan.total_cost < max_quality_cost:
                best_plan = plan
                best_plan_idx = idx
                max_quality = plan.quality
                max_quality_cost = plan.total_cost

        # if no plan was below fixed runtime; return shortest plan
        if best_plan is None:
            print("NO PLAN FOUND BELOW FIXED COST; PICKING MIN. RUNTIME PLAN INSTEAD")
            min_runtime = np.inf
            for idx, plan in enumerate(candidatePlans):
                if plan.total_time < min_runtime:
                    best_plan = plan
                    best_plan_idx = idx
                    min_runtime = plan.total_time

        return best_plan if not return_idx else (best_plan, best_plan_idx)


class MinCostAtFixedQuality(Policy):
    def __init__(self, min_quality: float):
        self.min_quality = min_quality

    def __str__(self):
        return "MinCost@FixedQuality"

    def choose(
        self, candidatePlans: List[PhysicalPlan], return_idx: bool = False
    ) -> Union[PhysicalPlan, Tuple[PhysicalPlan, int]]:
        best_plan, best_plan_idx, min_cost, min_cost_runtime = None, -1, np.inf, np.inf
        for idx, plan in enumerate(candidatePlans):
            # if plan is too low quality, skip
            if plan.quality < self.min_quality:
                continue

            # if plan is below best current min cost, this is new best plan
            if plan.total_cost < min_cost:
                best_plan = plan
                best_plan_idx = idx
                min_cost = plan.total_cost
                min_cost_runtime = plan.total_time

            # if plan is tied w/current min cost -- and has lower runtime -- this is new best plan
            elif plan.total_cost == min_cost and plan.total_time < min_cost_runtime:
                best_plan = plan
                best_plan_idx = idx
                min_cost = plan.total_cost
                min_cost_runtime = plan.total_time

        # if no plan was above fixed quality; return best plan
        if best_plan is None:
            print(
                "NO PLAN FOUND ABOVE FIXED QUALITY; PICKING MAX. QUALITY PLAN INSTEAD"
            )
            max_quality = 0
            for idx, plan in enumerate(candidatePlans):
                if plan.quality > max_quality:
                    best_plan = plan
                    best_plan_idx = idx
                    max_quality = plan.quality

        return best_plan if not return_idx else (best_plan, best_plan_idx)


class MinRuntimeAtFixedQuality(Policy):
    def __init__(self, min_quality: float):
        self.min_quality = min_quality

    def __str__(self):
        return "MinRuntime@FixedQuality"

    def choose(
        self, candidatePlans: List[PhysicalPlan], return_idx: bool = False
    ) -> Union[PhysicalPlan, Tuple[PhysicalPlan, int]]:
        best_plan, best_plan_idx, min_runtime, min_runtime_cost = (
            None,
            -1,
            np.inf,
            np.inf,
        )
        for idx, plan in enumerate(candidatePlans):
            # if plan is too low quality, skip
            if plan.quality < self.min_quality:
                continue

            # if plan is below best current min cost, this is new best plan
            if plan.total_time < min_runtime:
                best_plan = plan
                best_plan_idx = idx
                min_runtime = plan.total_time
                min_runtime_cost = plan.total_cost

            # if plan is tied w/current min runtime -- and has lower cost -- this is new best plan
            elif plan.total_time == min_runtime and plan.total_cost < min_runtime_cost:
                best_plan = plan
                best_plan_idx = idx
                min_runtime = plan.total_time
                min_runtime_cost = plan.total_cost

        # if no plan was above fixed quality; return best plan
        if best_plan is None:
            print(
                "NO PLAN FOUND ABOVE FIXED QUALITY; PICKING MAX. QUALITY PLAN INSTEAD"
            )
            max_quality = 0
            for idx, plan in enumerate(candidatePlans):
                if plan.quality > max_quality:
                    best_plan = plan
                    best_plan_idx = idx
                    max_quality = plan.quality

        return best_plan if not return_idx else (best_plan, best_plan_idx)


class MaxQualityMinRuntime(Policy):
    """
    This policy selects the plan with the maximum quality along the
    pareto-optimal curve of candidate plans. It then breaks ties by
    selecting the plan with the minimum runtime.
    """

    def __str__(self):
        return "(Maximum Quality, Minimum Runtime)"

    def choose(
        self, candidatePlans: List[PhysicalPlan], return_idx: bool = False
    ) -> Union[PhysicalPlan, Tuple[PhysicalPlan, int]]:
        best_plan, best_plan_idx, max_quality, max_quality_runtime = None, -1, 0, np.inf
        for idx, plan in enumerate(candidatePlans):
            if plan.quality > max_quality:
                best_plan = plan
                best_plan_idx = idx
                max_quality = plan.quality
                max_quality_runtime = plan.total_time
            elif plan.quality == max_quality and plan.total_time < max_quality_runtime:
                best_plan = plan
                best_plan_idx = idx
                max_quality = plan.quality
                max_quality_runtime = plan.total_time

        return best_plan if not return_idx else (best_plan, best_plan_idx)


class MinTime(Policy):
    """
    This policy selects the plan with the minimal execution time along the
    pareto-optimal curve of candidate plans.
    """

    def __str__(self):
        return "Minimum Time"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        return sorted(candidatePlans, key=lambda plan: plan.total_time)[0]


class MinCost(Policy):
    """
    This policy selects the plan with the minimal cost along the pareto-optimal
    curve of candidate plans.
    """

    def __str__(self):
        return "Minimum Cost"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        return sorted(candidatePlans, key=lambda plan: plan.total_cost)[0]


class MaxHarmonicMean(Policy):
    """
    This policy selects the plan with the maximum harmonic mean of cost, time, and quality
    along the pareto-optimal curve of candidate plans.
    """

    def __init__(
        self, max_time: float = 600.0, max_cost: float = 1.0, max_quality: float = 1.0
    ):
        self.max_cost = max_cost
        self.max_time = max_time
        self.max_quality = max_quality

    def __str__(self):
        return "Maximum Harmonic Mean"

    def choose(
        self, candidatePlans: List[PhysicalPlan], return_idx: bool = False
    ) -> Union[PhysicalPlan, Tuple[PhysicalPlan, int]]:
        epsilon = 1e-3
        bestPlan, bestHarmonicMean, bestPlanIdx = None, 0.0, -1
        for idx, plan in enumerate(candidatePlans):
            # scale time and cost into [0, 1]
            scaled_time = (self.max_time - plan.total_time) / self.max_time
            scaled_cost = (self.max_cost - plan.total_cost) / self.max_cost
            scaled_quality = (plan.quality) / self.max_quality
            scaled_time = min(max(scaled_time, 0.0), 1.0)
            scaled_cost = min(max(scaled_cost, 0.0), 1.0)
            scaled_quality = min(max(scaled_quality, 0.0), 1.0)

            harmonicMean = 3.0 / (
                (1.0 / (scaled_time + epsilon))
                + (1.0 / (scaled_cost + epsilon))
                + (1.0 / (scaled_quality + epsilon))
            )

            if harmonicMean > bestHarmonicMean:
                bestHarmonicMean = harmonicMean
                bestPlan = plan
                bestPlanIdx = idx

        return bestPlan if not return_idx else (bestPlan, bestPlanIdx)


class UserChoice(Policy):
    """
    This policy asks the user to decide which of the pareto-optimal
    candidate plans to execute.
    """

    def __str__(self):
        return "User Choice"

    def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
        print("Please select a plan from the following options:")
        for idx, plan in enumerate(candidatePlans):
            print(f"[{idx}] {plan}")
        user_choice = input(f"Please select a plan in [0-{len(candidatePlans) - 1}]: ")
        user_choice = int(user_choice)
        if user_choice not in range(len(candidatePlans)):
            print(
                f"Error: user choice {user_choice} was not a number in the specified range. Please try again."
            )
            return self.choose(candidatePlans)

        return candidatePlans[user_choice]
