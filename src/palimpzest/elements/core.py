from .elements import *

###################################################################################
# "Core" useful Element types. These are Elements that almost everyone will need.
# File, TextFile, Image, PDF, etc.
###################################################################################
class File(MultipartElement):
    """A File is a record that comprises a filename and the contents of the file."""
    filename = Field(desc="The UNIX-style name of the file", required=True)
    contents = BytesField(desc="The contents of the file", required=True)

class TextFile(File):
    """A text file is a File that contains only text. No binary data."""

class PDFFile(File):
    """A PDF file is a File that is a PDF. It has specialized fields, font information, etc."""
    # This class is currently very impoverished. It needs a lot more fields before it can correctly represent a PDF.
