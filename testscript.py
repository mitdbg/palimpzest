import os 
from PyPDF2 import PdfReader, PdfWriter

outdir = "testdata/bdf-usecase3-tiny/"
refdir = "testdata/bdf-usecase3-references-pdffull/"
pdfdir = "testdata/bdf-usecase3-pdf/"

for pdffile in os.listdir(pdfdir):
    pdfpath = os.path.join(pdfdir, pdffile)
    outpath = os.path.join(outdir, pdffile)
    refpath = os.path.join(refdir, pdffile)

    reader = PdfReader(pdfpath)
    reader.getPage(0)


    pdf_writer = PdfWriter()

    paper_reader = PdfReader(pdfpath)
    first_page = paper_reader.getPage(0)
    pdf_writer.addPage(first_page)

    ref_reader = PdfReader(refpath)    
    for page_num in range(ref_reader.getNumPages()):
        page = ref_reader.getPage(page_num)
        pdf_writer.addPage(page)
    
    # Write the merged PDF to a new file
    pdf_writer.write(outpath)