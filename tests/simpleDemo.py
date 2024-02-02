import palimpzest as pz

import os

class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)

def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    testRepo1 = pz.ConcreteDataset(pz.File, datasetId, desc="The dataset Mike downloaded on Jan 30")
    sciPapers = pz.Set(ScientificPaper, input=testRepo1, desc="Scientific papers")
    mitPapers = sciPapers.addFilterStr("The paper is from MIT")
    batteryPapers = mitPapers.addFilterStr("The paper is about batteries")
    goodAuthorPapers = batteryPapers.addFilterStr("Paper where the title begins with the letter X")

    return goodAuthorPapers


def emitDataset(title="Dataset"):
    def emitNestedTuple(node, indent=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            emitNestedTuple(child, indent=indent+2)

    dataset1 = "concretedata-01"
    rootSet = buildMITBatteryPaperPlan(dataset1)

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
    physicalTree = logicalTree.getPhysicalTree()
    print()
    print("Physical operator tree")
    physicalOps = physicalTree.dumpPhysicalTree()
    emitNestedTuple(physicalOps)

    #iterate over data
    print()
    print("Concrete data results")
    for r in physicalTree:
        print(r)


#
# Get battery papers and emit!
#
config = pz.Config(os.getenv("PZ_DIR"))
#srcDataDir = "./testFileDirectory"
#pz.DataDirectory().registerLocalDirectory(srcDataDir, "concretedataset-01")
#pz.DataDirectory().registerLocalDirectory(srcDataDir, "concretedataset-02")

emitDataset(title="Good MIT battery papers written by good authors")
