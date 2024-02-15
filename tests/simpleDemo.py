#!/usr/bin/env python3
import palimpzest as pz

import os
import argparse
import time


class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)

def buildTestPDFPlan(datasetId):
    """This tests whether we can process a PDF file"""
    testRepo1 = pz.ConcreteDataset(pz.File, datasetId, desc="A small test inputset")
    pdfPapers = pz.Set(pz.PDFFile, input=testRepo1, desc="PDFs")

    return pdfPapers

def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    testRepo1 = pz.ConcreteDataset(pz.File, datasetId, desc="The dataset Mike downloaded on Jan 30")
    sciPapers = pz.Set(ScientificPaper, input=testRepo1, desc="Scientific papers")
    batteryPapers = sciPapers.addFilterStr("The paper is about batteries")
    mitPapers = batteryPapers.addFilterStr("The paper is from MIT")
    goodAuthorPapers = mitPapers.addFilterStr("Paper where the title begins with the letter X")

    return goodAuthorPapers

class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)

def buildEnronPlan(datasetId):
    testRepo1 = pz.ConcreteDataset(pz.File, datasetId, desc="A collection of files")
    textFiles = pz.Set(pz.TextFile, input=testRepo1, desc="Text files")
    emails = pz.Set(Email, input=textFiles, desc="Emails")

    return emails

class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required = True)

def buildImagePlan(datasetId):
    testRepo1 = pz.ConcreteDataset(pz.File, datasetId, desc="A collection of images")
    images = pz.Set(pz.ImageFile, input=testRepo1, desc="Cast as images")
    filteredImages = images.addFilterStr("The image contains one or more dogs")
    dogImages = pz.Set(DogImage, input=filteredImages, desc = "Images of dogs")
    return dogImages


def emitDataset(rootSet, title="Dataset", verbose=False):
    def emitNestedTuple(node, indent=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            emitNestedTuple(child, indent=indent+2)


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
    return physicalTree

#
# Get battery papers and emit!
#
if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
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

    if task == "paper":
        rootSet = buildMITBatteryPaperPlan(datasetid)
        physicalTree = emitDataset(rootSet, title="Good MIT battery papers written by good authors", verbose=args.verbose)
        for r in physicalTree:
            print(r)
    elif task == "enron":
        rootSet = buildEnronPlan(datasetid)
        physicalTree = emitDataset(rootSet, title="Good Enron emails", verbose=args.verbose)
        for r in physicalTree:
            print(r.sender, r.subject)
            print()
        #planTime, planPrice, estimatedCardinality, physicalTree = rootSet.getLogicalTree().createPhysicalPlan()
        #for email in physicalTree:
        #    print(email.sender, email.subject)
    elif task == "pdftest":
        rootSet = buildTestPDFPlan(datasetid)
        physicalTree = emitDataset(rootSet, title="PDF files", verbose=args.verbose)

        for idx, r in enumerate(physicalTree):
            print("Extracted pdf", idx)
    elif task == "image":
        rootSet = buildImagePlan(datasetid)
        physicalTree = emitDataset(rootSet, title="Dogs", verbose=args.verbose)
        for r in physicalTree:
            print("\n\nGOT RESULT:")
            print(r.filename)
            print(r.breed)
            print()
    else:
        print("Unknown task")
        exit(1)


    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
