import json

from palimpzest.core.data.dataclasses import PlanCost


class Policy:
    """
    Base class for a policy. Each policy has two methods: constraint() and chooose().
    The first method determines whether the given cost, runtime, and quality for a plan
    (or sub-plan) satisfy the policy's constraint(s). The second method takes in the 
    (cost, runtime, quality) tuples for two plans (or subplans) and returns True if the
    first plan is better than the second one and False otherwise.
    """

    def __init__(self):
        pass

    def get_primary_metric(self) -> str:
        """
        Returns one of ["cost", "time", "quality"]; whichever corresponds to the
        maximization / minimization goal of the policy.

        Eventually we may make policies more general by allowing users to optimize
        some function: f(cost, time, quality). In that case, we may deprecate this
        method and update its callers.
        """
        raise NotImplementedError("Calling this method from an abstract base class.")

    def get_dict(self) -> dict:
        """
        Returns a dict representation of the policy which specifies how much weight
        (in [0,1]) should be given to each metric.
        """
        raise NotImplementedError("Calling this method from an abstract base class.")

    def constraint(self, plan: PlanCost) -> bool:
        """
        Return True if the given (cost, runtime, quality) for a plan (or subplan)
        satisfy the policy's constraint(s). Otherwise, return False.
        """
        raise NotImplementedError("Calling this method from an abstract base class.")

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan is better than other_plan and return False otherwise.
        """
        raise NotImplementedError("Calling this method from an abstract base class.")

    def to_json_str(self) -> str:
        """Convert policy configuration to a JSON-serializable dictionary."""
        return json.dumps({
            "type": self.__class__.__name__,
            "config": self.get_dict()
        }, indent=2)

class MaxQuality(Policy):
    """
    This policy has no constraints and computes the best plan as the one with
    the higher quality.
    """

    def __str__(self):
        return "Maximum Quality"

    def get_primary_metric(self) -> str:
        return "quality"

    def get_dict(self) -> dict:
        return {"cost": 0.0, "time": 0.0, "quality": 1.0}

    def constraint(self, plan: PlanCost) -> bool:
        """There is no constraint."""
        return True

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has higher quality than other_plan and return False otherwise.
        Use cost and then runtime as tiebreakers.
        """
        if plan.quality == other_plan.quality:
            if plan.cost == other_plan.cost:
                return plan.time < other_plan.time
            return plan.cost < other_plan.cost

        return plan.quality > other_plan.quality


class MinCost(Policy):
    """
    This policy has no constraints and computes the best plan as the one with
    the lower cost.
    """

    def __str__(self):
        return "Minimum Cost"

    def get_primary_metric(self) -> str:
        return "cost"

    def get_dict(self) -> dict:
        return {"cost": 1.0, "time": 0.0, "quality": 0.0}

    def constraint(self, plan: PlanCost) -> bool:
        """There is no constraint."""
        return True

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has lower cost than other_plan and return False otherwise.
        Use quality and then runtime as tiebreakers.
        """
        if plan.cost == other_plan.cost:
            if plan.quality == other_plan.quality:
                return plan.time < other_plan.time
            return plan.quality > other_plan.quality

        return plan.cost < other_plan.cost


class MinTime(Policy):
    """
    This policy has no constraints and computes the best plan as the one with
    the lower runtime.
    """

    def __str__(self):
        return "Minimum Time"

    def get_primary_metric(self) -> str:
        return "time"

    def get_dict(self) -> dict:
        return {"cost": 0.0, "time": 1.0, "quality": 0.0}

    def constraint(self, plan: PlanCost) -> bool:
        """There is no constraint."""
        return True

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has lower runtime than other_plan and return False otherwise.
        Use quality and then cost as tiebreakers.
        """
        if plan.time == other_plan.time:
            if plan.quality == other_plan.quality:
                return plan.cost < other_plan.cost
            return plan.quality > other_plan.quality

        return plan.time < other_plan.time


class MaxQualityAtFixedCost(Policy):
    """
    This policy applies a constraint (upper bound) on the cost of the plan
    and tries to maximize quality subject to that constraint.
    """

    def __init__(self, max_cost: float):
        self.max_cost = max_cost

    def __str__(self):
        return "MaxQuality@FixedCost"

    def get_primary_metric(self) -> str:
        return "quality"

    def get_dict(self) -> dict:
        return {"cost": 0.5, "time": 0.0, "quality": 0.5}

    def constraint(self, plan: PlanCost) -> bool:
        return plan.cost < self.max_cost

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has higher quality than other_plan and return False otherwise.
        Use cost and then runtime as a tie-breaker.
        """
        if plan.quality == other_plan.quality:
            if plan.cost == other_plan.cost:
                return plan.time < other_plan.time
            return plan.cost < other_plan.cost

        return plan.quality > other_plan.quality


class MaxQualityAtFixedTime(Policy):
    """
    This policy applies a constraint (upper bound) on the runtime of the plan
    and tries to maximize quality subject to that constraint.
    """

    def __init__(self, max_time: float):
        self.max_time = max_time

    def __str__(self):
        return "MaxQuality@FixedTime"

    def get_primary_metric(self) -> str:
        return "quality"

    def get_dict(self) -> dict:
        return {"cost": 0.0, "time": 0.5, "quality": 0.5}

    def constraint(self, plan: PlanCost) -> bool:
        return plan.time < self.max_time

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has higher quality than other_plan and return False otherwise.
        Use runtime and then cost as a tie-breaker.
        """
        if plan.quality == other_plan.quality:
            if plan.time == other_plan.time:
                return plan.cost < other_plan.cost
            return plan.time < other_plan.time

        return plan.quality > other_plan.quality


class MinCostAtFixedQuality(Policy):
    """
    This policy applies a constraint (lower bound) on the quality of the plan
    and tries to minimize cost subject to that constraint.
    """

    def __init__(self, min_quality: float):
        self.min_quality = min_quality

    def __str__(self):
        return "MinCost@FixedQuality"

    def get_primary_metric(self) -> str:
        return "cost"

    def get_dict(self) -> dict:
        return {"cost": 0.5, "time": 0.0, "quality": 0.5}

    def constraint(self, plan: PlanCost) -> bool:
        return plan.quality > self.min_quality

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has lower cost than other_plan and return False otherwise.
        Use quality and then runtime as a tie-breaker.
        """
        if plan.cost == other_plan.cost:
            if plan.quality == other_plan.quality:
                return plan.time < other_plan.time
            return plan.quality > other_plan.quality

        return plan.cost < other_plan.cost


class MinTimeAtFixedQuality(Policy):
    """
    This policy applies a constraint (lower bound) on the quality of the plan
    and tries to minimize runtime subject to that constraint.
    """

    def __init__(self, min_quality: float):
        self.min_quality = min_quality

    def __str__(self):
        return "MinTime@FixedQuality"

    def get_primary_metric(self) -> str:
        return "time"

    def get_dict(self) -> dict:
        return {"cost": 0.0, "time": 0.5, "quality": 0.5}

    def constraint(self, plan: PlanCost) -> bool:
        return plan.quality > self.min_quality

    def choose(self, plan: PlanCost, other_plan: PlanCost) -> float:
        """
        Return True if plan has lower runtime than other_plan and return False otherwise.
        Use quality and then cost as a tie-breaker.
        """
        if plan.time == other_plan.time:
            if plan.quality == other_plan.quality:
                return plan.cost < other_plan.cost
            return plan.quality > other_plan.quality

        return plan.time < other_plan.time


# TODO: add this back in a way which allows users to select a plan from a small pareto optimal set at the end of
# query optimization
# class UserChoice(Policy):
#     """
#     This policy asks the user to decide which of the pareto-optimal
#     candidate plans to execute.
#     """

#     def __str__(self):
#         return "User Choice"

#     def choose(self, candidatePlans: List[PhysicalPlan]) -> PhysicalPlan:
#         print("Please select a plan from the following options:")
#         for idx, plan in enumerate(candidatePlans):
#             print(f"[{idx}] {plan}")
#         user_choice = input(f"Please select a plan in [0-{len(candidatePlans) - 1}]: ")
#         user_choice = int(user_choice)
#         if user_choice not in range(len(candidatePlans)):
#             print(
#                 f"Error: user choice {user_choice} was not a number in the specified range. Please try again."
#             )
#             return self.choose(candidatePlans)

#         return candidatePlans[user_choice]
