import os
import modal

stub = modal.Stub()
pipPacks = ["papermage", 
            "tqdm", 
            "transformers", 
            "pdf2image", 
            "pdfplumber==0.7.4", 
            "requests", 
            "numpy>=1.23.2", 
            "scipy>=1.9.0", 
            "pandas<2", 
            "ncls==0.0.68", 
            "necessary>=0.3.2", 
            "grobid-client-python==0.0.5", 
            "charset-normalizer", 
            "torch>=1.10.0", 
            "smashed", 
            "layoutparser", 
            "pysbd",
            "decontext",
            "vila"]

pdfProcessingImage = modal.Image.debian_slim(python_version="3.11").apt_install(
    ["ffmpeg", "pkg-config", "libpoppler-cpp-dev", "poppler-utils"]).pip_install(
    ["torch==2.1.1", "pkgconfig", "python-poppler"] + pipPacks)

@stub.function(image=pdfProcessingImage)
def processPdf(pdfBytesDocs: list[bytes]):
    """Process a PDF file and return the text contents."""
    import papermage
    import os
    import json
    from papermage.recipes import CoreRecipe

    os.makedirs("/tmp", exist_ok=True)
    results = []
    for pdfBytes in pdfBytesDocs:
        recipe = CoreRecipe()

        with open("/tmp/papermage.pdf", "wb") as file:
            file.write(pdfBytes)

        doc = recipe.run("/tmp/papermage.pdf")

        os.remove("/tmp/papermage.pdf")

        # print all the attrs in doc
        abstracts = []
        sentences = []
        titles = []
        for p in doc.pages:
            for s in p.sentences:
                sentences.append(s.text)

        #doc.abstracts
        for a in doc.abstracts:
            for s in a.sentences:
                abstracts.append(s.text)

        #doc.titles
        for t in doc.titles:
            for s in t.sentences:
                titles.append(s.text)
        #doc.authors
        #doc.equations
        #doc.figures
        #doc.tables
        #results.append((titles, abstracts, sentences))
        results.append(json.dumps(doc.to_json()))

    return results



@stub.local_entrypoint()
def main():
    import json
    from papermage import Document
    pdfBytes1 = open("test.pdf", "rb").read()

    results = processPdf.remote([pdfBytes1])
    for idx, r in enumerate(results):
        docdict = json.loads(r)
        doc = Document.from_json(docdict)
        print(idx, doc)



