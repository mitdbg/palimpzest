from .plan import Plan
from typing import List
class Planner:
    """
    A Planner is responsible for generating a set of possible plans.
    The fundamental abstraction is, given an input of a graph (of datasets, or logical operators), it generates a set of possible graphs which correspond to the plan.
    These plans can be consumed from the planner using the __iter__ method.
    """

    def __init__(self):
        self.plans = []

    def generate_plans(self) -> List[Plan]:
        return NotImplementedError

    def __iter__(self):
        return iter(self.plans)

    def __next__(self):
        return next(iter(self.plans))

    def __len__(self):
        return len(self.plans)