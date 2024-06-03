import modal

app = modal.App("palimpzest.tools")
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

@app.function(image=pdfProcessingImage)
def processPapermagePdf(pdfBytesDocs: list[bytes]):
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

        results.append(json.dumps(doc.to_json()))

    return results



@app.local_entrypoint()
def main():
    import json
    from papermage import Document
    pdfBytes1 = open("test.pdf", "rb").read()

    results = processPapermagePdf.local([pdfBytes1])
    for idx, r in enumerate(results):
        docdict = json.loads(r)
        doc = Document.from_json(docdict)
        print(idx, doc)
        for p in doc.pages:
            print(p)



