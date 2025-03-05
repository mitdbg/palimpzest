import fitz  # PyMuPDF

def extract_highlighted_text(pdf_path):
    doc = fitz.open(pdf_path)
    highlighted_text = []

    for page_number, page in enumerate(doc, start=1):
        annot = page.first_annot
        while annot:
            if annot.type[0] == 8:
                try:
                    text = annot.get_text("text")
                    if text.strip():
                        highlighted_text.append((page_number, text.strip()))
                except AttributeError:
                    print(f"get_text() not available for annotations in this PyMuPDF version.")
            annot = annot.next
    if highlighted_text:
        for idx, (page_num, text) in enumerate(highlighted_text, start=1):
            print(f"Highlight {idx} (Page {page_num}): {text}\n")
    else:
        print("No highlighted text found in the PDF.")


# Example usage
pdf_file = "/Users/yashaga/palimpzest/testdata/matsci_pdfs/Synthesis_ArcMelting_Cava_Nature367252a0.pdf"
extract_highlighted_text(pdf_file)
