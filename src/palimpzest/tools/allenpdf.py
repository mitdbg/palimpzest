import os
import modal

stub = modal.Stub()

#    Image.from_registry(
#        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
#    )
#pdfProcessingImage = modal.Image.debian_slim().pip_install("tqdm", 

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
#pdfProcessingImage = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").conda_install(["poppler"]).pip_install("torch==2.1.1").pip_install(pipPacks)
 #                                                          "pdfplumber==0.7.4", 
 #                                                          "requests", 
 #                                                          "numpy>=1.23.2",
 #                                                          "scipy>=1.9.0",
 #                                                          "pandas<2",
 #                                                          "ncls==0.0.68",
 #                                                          "necessary>=0.3.2",
 #                                                          "grobid-client-python==0.0.5",
 #                                                          "charset-normalizer",
 #                                                          "torch>=1.10.0")

@stub.function(image=pdfProcessingImage)
def processPdf(pdfBytes):
    """Process a PDF file and return the text contents."""
    import papermage

    from papermage.recipes import CoreRecipe
    recipe = CoreRecipe()

    os.makedirs("/tmp", exist_ok=True)
    with open("/tmp/papermage.pdf", "wb") as file:
        file.write(pdfBytes)

    doc = recipe.run("/tmp/papermage.pdf")

    os.remove("/tmp/papermage.pdf")

    return doc.abstracts[0].sentences[0]



@stub.local_entrypoint()
def main():
    pdfBytes = open("test.pdf", "rb").read()
    text = processPdf.remote(pdfBytes)
    #text = "foo"
    print("TEXT", text)


