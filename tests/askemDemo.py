import palimpzest as pz
# from tests.simpleDemo import emitDataset

from palimpzest import ImageFile, DataRecord
from palimpzest.elements.pzlist import ListField
from tests.simpleDemo import emitDataset


# class DocumentSchema(Schema):
#     """A document element that can contain a list of either Equation Images or Plot Images."""
#     images = pz.List(Any([EquationImage, PlotImage]), desc="Images in the document that are either equations or plots", cardinality="0..*")

class ScientificPaper(pz.PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""
    title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
    publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
    textualContent = pz.Field(desc="The text of the paper", required=True)
    images = ListField(pz.ImageFile, desc="Images in the paper", cardinality="0..*")

class PandemicPaper(ScientificPaper):
    """A scientific paper about modeling pandemics"""
    disease = pz.Field(desc="The disease being modeled", required=True)
    replicationRate = pz.Field(desc="The replication rate of the disease", required=False)
    coderepo = pz.Field(desc="The URL of the code repo that accompanies the paper", required=False)
    dataplots = ListField(pz.PlotImage, desc="Plots in the paper", cardinality="0..*")
    equationImage = ListField(pz.EquationImage, desc="Equation images in the paper", cardinality="0..*")


class Equation(pz.Schema):
    """A mathematical equation"""
    text = pz.Field(desc="The text of the equation", required=True)
    image_content = pz.BytesField(desc="An image of the equation", required=True)
    variables = ListField(pz.Field, desc="The variables in the equation", required=True, cardinality="1..*")

def buildPandemicPaperPlan(datasetId):
    """Fetches pandemic papers from a dataset"""
    pandemicPapers = pz.Dataset(datasetId, schema=PandemicPaper)
    return pandemicPapers

def buildEquationPlan(datasetId):
    """Fetches equations from pandemic papers in a dataset"""
    pandemicPapers = buildPandemicPaperPlan(datasetId)
    equations = pandemicPapers.oneToManySet(Equation, desc="Equations in the papers")
    stockAndFlowEquations = equations.filterByStr("The equation describes a stock and flow model")
    return stockAndFlowEquations

def testEmit(datasetId = "askem-testset-02072024"):
    # Assuming you have a dataset ID to work with


    # Build the plan for fetching stock and flow equations from pandemic papers
    stockAndFlowEquations = buildEquationPlan(datasetId)

    # Emit the dataset for stock and flow equations
    # physicalTree = emitDataset(stockAndFlowEquations, title="Stock and Flow Equations in Pandemic Papers")

    # for dataRecord in physicalTree:
    #     print(dataRecord.text, dataRecord.variables, dataRecord.parent.title, dataRecord.parent.publicationYear)

def buildEquationImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filterByStr("The image contains an equation")
    equationImages = filteredImages.convert(pz.EquationImage, desc = "Image that contains an equation")
    return equationImages

if __name__ == "__main__":
    datasetid = "equation-tiny"
    rootSet = buildEquationImagePlan(datasetid)
    physicalTree = emitDataset(rootSet, title="equations", verbose=True)
    for r in physicalTree:
        print(r.filename, r.equation_text)


# Define a ListField (pz.List) to hold EquationImage instances
# equation_images = ListField(element_types=EquationImage, desc="List of equation images")
#
# # Create some EquationImage instances
# eq_img1 = EquationImage(filename="eq1.png", equation_text="x^2 + y^2 = r^2")
# eq_img2 = EquationImage(filename="eq2.png", equation_text="E = mc^2")
#
# # Append EquationImage instances to the list
# equation_images.append(eq_img1)
# equation_images.append(eq_img2)
#
# # Iterate through the list and print details of each EquationImage
# for eq_img in equation_images:
#     print(f"Filename: {eq_img.filename}, Equation Text: {eq_img.equation_text}")