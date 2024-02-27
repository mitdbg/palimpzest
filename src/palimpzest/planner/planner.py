from palimpzest.elements import Filter
from palimpzest.operators import LogicalOperator, PhysicalOp
from palimpzest.sets import Set

from typing import List, Union

import palimpzest as pz

# DEFINITIONS
Node = Union[LogicalOperator, PhysicalOp, Set]


class Plan:
    """
    This base class takes in the 
    """

    def __init__(self, dataset_id: str=None, source: str=None):
        self.dataset_id = dataset_id
        self.source = source

    def emitNestedTuple(self, node: Node, indent: int=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            self.emitNestedTuple(child, indent=indent + 2)

    def sources(self) -> Union[pz.Dataset, List[pz.Dataset]]:
        """
        User-defined sources.
        """
        pass

    def filters(self) -> Union[Filter, List[Filter]]:
        """
        User-defined filters.
        """
        pass

    def plan(self) -> pz.Dataset:
        """
        Function which enables users to imperatively define their logical plan.
        """
        pass

    def buildLogicalPlan(self) -> pz.Dataset:
        """
        Build logical plan using sources, filters, transforms, aggregates, etc.
        """
        pass

    def buildPhysicalPlan(self, verbose: bool=False) -> PhysicalOp:
        """
        Build physical plan from logical plan.
        """
        user_plan_root_dataset = self.plan()
        rootDataset = self.buildLogicalPlan() if user_plan_root_dataset is None else user_plan_root_dataset

        # Print the syntactic tree
        syntacticElements = rootDataset.dumpSyntacticTree()
        if verbose:
            print("Syntactic operator tree")
            self.emitNestedTuple(syntacticElements)

        # Print the (possibly optimized) logical tree
        logicalTree = rootDataset.getLogicalTree()
        logicalElements = logicalTree.dumpLogicalTree()
        if verbose:
            print()
            print("Logical operator tree")
            self.emitNestedTuple(logicalElements)

        # Print the physical operators that will be executed
        planTime, planCost, estimatedCardinality, physicalTree = logicalTree.createPhysicalPlan()
        if verbose:
            print()
            print("Physical operator tree")
            physicalOps = physicalTree.dumpPhysicalTree()
            print()
            print("estimated costs:", physicalTree.estimateCost())
            self.emitNestedTuple(physicalOps)

            #iterate over data
            print()
            print("Estimated seconds to complete:", planTime)
            print("Estimated USD to complete:", planCost)
            print("Estimated cardinality:", estimatedCardinality)
            print("Concrete data results")

        return physicalTree

    def execute(self, verbose: bool=False) -> None:
        """
        Execute the physical plan.

        TODO: Currently prints out all records, we may want to return these in a list instead.
        """
        physicalTree = self.buildPhysicalPlan(verbose)
        for r in physicalTree:
            print(r)
