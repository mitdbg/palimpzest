import palimpzest as pz

import json

class ScientificPaper(pz.File):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)

def getMITBatteryPapers():
    """A dataset-independent declarative description of authors of good papers"""
    sciPapers = pz.Set(ScientificPaper)
    mitPapers = sciPapers.addFilterStr("The paper is from MIT")
    batteryPapers = mitPapers.addFilterStr("The paper is about batteries")
    goodAuthorPapers = batteryPapers.addFilterStr("Papers where the author list contains at least one highly-respected author",
                                            targetFn=lambda x: x.authors)
    return goodAuthorPapers

def emitDataset(rootSet, srcDataDir, title="Dataset"):
    def emitNestedTuple(node, indent=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            emitNestedTuple(child, indent=indent+2)

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

    # Print the JSON schema that will be populated
    jsonSchema = logicalTree.outputElementType.jsonSchema()
    jsonSchema["title"]=title
    print()
    print("JSON SCHEMA")
    print(json.dumps(jsonSchema, indent=2))

    # Print the physical operators that will be executed
    dataSrc = pz.DirectorySource(srcDataDir)
    physicalTree = logicalTree.getPhysicalTree()
    physicalTree.finalize(dataSrc)
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
rootSet = getMITBatteryPapers()
emitDataset(rootSet, "./testFileDirectory", title="Good MIT battery papers written by good authors")
