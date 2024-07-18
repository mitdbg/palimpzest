from palimpzest.cost_estimator import CostEstimator
# from palimpzest.planner import PhysicalPlan
# from palimpzest.policy import Policy


class PruningStrategy:

    def __init__(self, *args, **kwargs):
        pass

    def prune_plan(self, plan) -> bool:
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")


class ParetoPruningStrategy(PruningStrategy):

    def __init__(self, cost_estimator: CostEstimator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_estimator = cost_estimator
        self.frontier_plans = []

    def prune_plan(self, plan) -> bool:
        """
        Function which returns True if the plan should be pruned and false otherwise.
        """
        # cost the plan
        plan_total_cost, plan_total_time, plan_quality = self.cost_estimator.estimate_plan_cost(plan)

        # check if the plan is on the pareto frontier
        dominated_indices = set()
        for idx, (total_cost, total_time, quality) in enumerate(self.frontier_plans):
            # if plan is dominated by frontier plan, prune the plan
            if (
                total_time <= plan_total_time
                and total_cost <= plan_total_cost
                and quality >= plan_quality
            ):
                # import pdb; pdb.set_trace()
                return True

            # check if the plan dominates the frontier plan
            if (
                plan_total_time <= total_time
                and plan_total_cost <= total_cost
                and plan_quality >= quality
            ):
                dominated_indices.add(idx)

        # reaching this point means the plan is on the frontier; remove any dominated plans
        self.frontier_plans = [plan for idx, plan in enumerate(self.frontier_plans) if idx not in dominated_indices]

        # add plan to frontier
        self.frontier_plans.append((plan_total_cost, plan_total_time, plan_quality))

        # do not prune plan
        # import pdb; pdb.set_trace()
        return False

    def clear_frontier(self) -> None:
        self.frontier_plans = []


class ParetoPlusPolicyPruningStrategy(PruningStrategy):

    def __init__(self, cost_estimator: CostEstimator, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_estimator = cost_estimator
        self.policy = policy
        self.frontier_plans = []

    def prune_plan(self, plan) -> bool:
        # cost the plan
        total_cost, total_time, quality = self.cost_estimator.estimate_plan_cost(plan)

        raise NotImplementedError("need to finish this function")
