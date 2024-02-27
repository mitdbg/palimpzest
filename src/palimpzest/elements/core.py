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
    text_contents = Field(desc="The text-only contents of the PDF", required=True)

class ImageFile(File):
    """A file that contains an image."""
    text_description = Field(desc="A text description of the image", required=False)

class Number(Element):
    """Just a number. Often used for aggregates"""
    value = Field(desc="A single number", required=True)


class EquationImage(ImageFile):
    """An image that contains a mathematical equation."""
    equation_text = Field(desc="The text representation of the equation in the image", required=True)

class PlotImage(ImageFile):
    """An image that contains a plot, such as a graph or chart."""
    plot_description = Field(desc="A description of the plot", required=True)
