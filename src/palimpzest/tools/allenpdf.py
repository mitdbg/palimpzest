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
pdfProcessingImage = modal.Image.debian_slim(python_version="3.11").apt_install(["ffmpeg", "pkg-config", "libpoppler-cpp-dev", "poppler-utils"]).pip_install("torch==2.1.1").pip_install("pkgconfig").pip_install(pipPacks).pip_install("python-poppler")

@stub.function(image=pdfProcessingImage)
def processPdf(pdfBytes):
    """Process a PDF file and return the text contents."""
    import papermage
    import os
    from papermage.recipes import CoreRecipe
    recipe = CoreRecipe()

    os.makedirs("/tmp", exist_ok=True)
    with open("/tmp/papermage.pdf", "wb") as file:
        file.write(pdfBytes)

    doc = recipe.run("/tmp/papermage.pdf")

    os.remove("/tmp/papermage.pdf")

    print("HERE IS THE DOC", doc)

    return str(doc.abstracts[0].sentences[0])



@stub.local_entrypoint()
def main():
    pdfBytes = open("test.pdf", "rb").read()
    text = processPdf.remote(pdfBytes)
    print("TEXT", text)

