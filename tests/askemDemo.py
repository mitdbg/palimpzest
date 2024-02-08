import palimpzest as pz

class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
   textualContent = pz.Field(desc="The text of the paper", required=True)
   images = pz.List(pz.Image, desc="Images in the paper", cardinality="0..*")

class PandemicPaper(ScientificPaper):
    """A scientific paper about modeling pandemics"""
    disease = pz.Field(desc="The disease being modeled", required=True)
    replicationRate = pz.Field(desc="The replication rate of the disease", required=False)
    coderepo = pz.Field(desc="The URL of the code repo that accompanies the paper", required=False)
    dataplots = pz.List(pz.PlotImage, "Plots in the paper", cardinality="0..*")
    equationImage = pz.List(pz.Image, "Equation images in the paper", cardinality="0..*")

class Equation(pz.Element):
    """A mathematical equation"""
    text = pz.Field(desc="The text of the equation", required=True)
    image = pz.Image(desc="An image of the equation", required=True)
    variables = pz.List(pz.Field, desc="The variables in the equation", required=True, cardinality="1..*")

askemTestPapers = pz.ConcreteDataset(pz.File, "askem-testset-02072024", desc="The Wisconsin Corpus of PDFs")
pandemicPapers = pz.Set(PandemicPaper, input=askemTestPapers, desc="Scientific papers about pandemics")
equations = pandemicPapers.oneToManySet(Equation, desc="Equations in the papers")
stockAndFlowEquations = equations.addFilterStr("The equation describes a stock and flow model")

logicalTree = stockAndFlowEquations.getLogicalTree()
planTime, planPrice, estimatedCardinality, physicalTree = logicalTree.createPhysicalPlan()

for dataRecord in physicalTree:
    print(dataRecord.text, dataRecord.variables, dataRecord.parent.title, datarecord.parent.publicationYear)