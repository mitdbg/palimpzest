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
    pdfPapers = pz.getData(datasetId, basicElt=pz.PDFFile)

    return pdfPapers

def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    sciPapers = pz.getData(datasetId, basicElt=ScientificPaper)
    batteryPapers = sciPapers.filterByStr("The paper is about batteries")
    mitPapers = batteryPapers.filterByStr("The paper is from MIT")

    return mitPapers


def testCount(datasetId):
    files = pz.getData(datasetId)
    fileCount = files.aggregate("COUNT")
    return fileCount

def testAverage(datasetId):
    data = pz.getData(datasetId)
    average = data.aggregate("AVERAGE")
    return average

def testLimit(datasetId, n):
    data = pz.getData(datasetId)
    limitData = data.limit(n)
    return limitData

class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)

def buildEnronPlan(datasetId):
    emails = pz.getData(datasetId, basicElt=Email)
    filteredEmails = emails.filterByStr("The email was written to a woman")
    return filteredEmails

def computeEnronStats(datasetId):
    emails = pz.getData(datasetId, basicElt=Email)
    #filteredEmails = emails.filterByStr("The email is about someone taking a vaction")
    subjectLineLengths = emails.convert(pz.Number, desc = "The number of words in the subject field")
    #return subjectLineLengths.aggregate("AVERAGE")
    return subjectLineLengths
    #return filteredEmails


class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required = True)

def buildImagePlan(datasetId):
    images = pz.getData(datasetId, basicElt=pz.ImageFile)
    filteredImages = images.filterByStr("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc = "Images of dogs")
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
    planTime, planCost, estimatedCardinality, physicalTree = logicalTree.createPhysicalPlan()    
    print()
    print("Physical operator tree")
    physicalOps = physicalTree.dumpPhysicalTree()
    emitNestedTuple(physicalOps)

    #iterate over data
    print()
    print("Estimated seconds to complete:", planTime)
    print("Estimated USD to complete:", planCost)
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
        #rootSet = computeEnronStats(datasetid)
        physicalTree = emitDataset(rootSet, title="Good Enron emails", verbose=args.verbose)
        for r in physicalTree:
            print(r)
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
            print(r.filename, r.breed)
    elif task == "count":
        rootSet = testCount(datasetid)
        physicalTree = emitDataset(rootSet, title="Count records", verbose=args.verbose)
        for r in physicalTree:
            print(r)
    elif task == "average":
        rootSet = testAverage(datasetid)
        physicalTree = emitDataset(rootSet, title="Average of numbers", verbose=args.verbose)
        for r in physicalTree:
            print(r)
    elif task == "limit":
        rootSet = testLimit(datasetid, 5)
        physicalTree = emitDataset(rootSet, title="Limit the set to 5 items", verbose=args.verbose)
        for r in physicalTree:
            print(r)
    else:
        print("Unknown task")
        exit(1)


    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
