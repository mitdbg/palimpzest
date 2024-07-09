from palimpzest.operators.physical import PhysicalOperator
from abc import ABC, abstractmethod


class PhysicalOpStrategy(ABC):
    """A PhysicalOpStrategy is a strategy that is based on a physical operation.
    It acts as a factory class for logical operation to automatically implement parametrized physical operators.
    At its core, a strategy must declare:
    - the logical operation it is based on
    - the parameters that exposes for execution/planning
    - the physical operator TYPE it will generate
    - A function that takes as input the strategy parameter, and it returns a physical operator CLASS with a parametrized __call__ method

    For an easiest example, consider checking the LLMFilterStrategy class in this directory.
    """

    logical_op_class = None
    physical_op_class = None
    
    @abstractmethod
    def __new__(cls) -> PhysicalOperator:
        raise NotImplementedError("This is an abstract class. Please implement a concrete strategy that accepts a specific parameter new class.")
