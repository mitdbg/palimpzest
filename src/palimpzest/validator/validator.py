# from abc import ABC, abstractmethod
from typing import Callable


class BaseValidator:
    """
    The Validator is used during optimization to score the output of physical operator(s) and physical plan(s).

    The core function of the Validator is to take a (set of) input(s) and a (set of) output(s)
    - LLM validation vs. Non-LLM validation
    - operator-level validation vs. plan-level validation
    - LLM validation may only make sense at the operator-level
    - Non-LLM Validation may work at the operator and/or plan-level

    TODO: start with non-llm based operator level validation; port over code from Sentinel executor
    TODO: allow Validator to come with its own source Dataset(s)
    TODO: try to eliminate need for source_idx
    """
    def __init__(self, eval_fn: Callable) -> None:
        self.eval_fn = eval_fn


class Validator(BaseValidator):
    """
    """
    pass


class LLMValidator(BaseValidator):
    """
    """
    pass
