import modal

app = modal.App()

#    Image.from_registry(
#        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
#    )
#pdfProcessingImage = modal.Image.debian_slim().pip_install("tqdm", 

pipPacks = ["papermage", "tqdm", "transformers", "pdf2image", "pdfplumber==0.7.4", "requests", "numpy>=1.23.2", "scipy>=1.9.0", "pandas<2", "ncls==0.0.68", "necessary>=0.3.2", "grobid-client-python==0.0.5", "charset-normalizer", "torch>=1.10.0", "smashed"]
pdfProcessingImage = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("torch==2.1.1").pip_install(pipPacks)
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
