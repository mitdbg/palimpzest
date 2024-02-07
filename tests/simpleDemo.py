#!/usr/bin/env python3
import palimpzest as pz

import os
import argparse


class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)

def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    testRepo1 = pz.ConcreteDataset(pz.File, datasetId, desc="The dataset Mike downloaded on Jan 30")
    sciPapers = pz.Set(ScientificPaper, input=testRepo1, desc="Scientific papers")
    batteryPapers = sciPapers.addFilterStr("The paper is about batteries")
    mitPapers = batteryPapers.addFilterStr("The paper is from MIT")
    goodAuthorPapers = mitPapers.addFilterStr("Paper where the title begins with the letter X")

    return goodAuthorPapers

def emitDataset(datasetid, title="Dataset", verbose=False):
    def emitNestedTuple(node, indent=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            emitNestedTuple(child, indent=indent+2)

    rootSet = buildMITBatteryPaperPlan(datasetid)

    print()
    print()
    print("# Let's test the basic functionality of the system")

    # Print the syntactic tree
    syntacticElements = rootSet.dumpSyntacticTree()
    print()
    print("Syntactic operator tree")
    emitNestedTuple(syntacticElements)

    # Print the (possibly optimized) logical tree
    logicalTree = rootSet.getLogicalTree()
    logicalElements = logicalTree.dumpLogicalTree()
    print()
    print("Logical operator tree")
    emitNestedTuple(logicalElements)

    # Print the physical operators that will be executed
    planTime, planPrice, estimatedCardinality, physicalTree = logicalTree.createPhysicalPlan()    
    print()
    print("Physical operator tree")
    physicalOps = physicalTree.dumpPhysicalTree()
    print()
    print("estimated costs:", physicalTree.estimateCost())
    emitNestedTuple(physicalOps)

    #iterate over data
    print()
    print("Estimated seconds to complete:", planTime)
    print("Estimated USD to complete:", planPrice)
    print("Estimated cardinality:", estimatedCardinality)
    print("Concrete data results")
    for r in physicalTree:
        print(r)


#
# Get battery papers and emit!
#
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print verbose output')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--task' , type=str, help='The task to run')

    args = parser.parse_args()

    # The user has to indicate the dataset id and the task
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.task is None:
        print("Please provide a task")
        exit(1)

    datasetid = args.datasetid
    task = args.task

    config = pz.Config(os.getenv("PZ_DIR"))
    if task == "paper":
        emitDataset(datasetid, title="Good MIT battery papers written by good authors", verbose=args.verbose)
    else:
        print("Unknown task")
        exit(1)
