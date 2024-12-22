import modal

app = modal.App("palimpzest.tools")
pip_packs = [
    "papermage",
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
    "vila",
]

pdf_processing_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["ffmpeg", "pkg-config", "libpoppler-cpp-dev", "poppler-utils"])
    .pip_install(["torch==2.1.1", "pkgconfig", "python-poppler"] + pip_packs)
)


@app.function(image=pdf_processing_image)
def process_papermage_pdf(pdf_bytes_docs: list[bytes]):
    """Process a PDF file and return the text contents."""
    import json
    import os

    from papermage.recipes import CoreRecipe

    os.makedirs("/tmp", exist_ok=True)
    results = []
    for pdf_bytes in pdf_bytes_docs:
        recipe = CoreRecipe()

        with open("/tmp/papermage.pdf", "wb") as file:
            file.write(pdf_bytes)

        doc = recipe.run("/tmp/papermage.pdf")

        os.remove("/tmp/papermage.pdf")

        results.append(json.dumps(doc.to_json()))

    return results


@app.local_entrypoint()
def main():
    import json

    from papermage import Document

    with open("test.pdf", "rb") as file:
        pdf_bytes_1 = file.read()

    results = process_papermage_pdf.local([pdf_bytes_1])
    for idx, r in enumerate(results):
        docdict = json.loads(r)
        doc = Document.from_json(docdict)
        print(idx, doc)
        for p in doc.pages:
            print(p)
