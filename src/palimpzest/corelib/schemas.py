from palimpzest.elements import BytesField, Schema, StringField, File

###################################################################################
# "Core" useful Schemas. These are Schemas that almost everyone will need.
# File, TextFile, Image, PDF, etc.
###################################################################################
class PDFFile(File):
    """A PDF file is a File that is a PDF. It has specialized fields, font information, etc."""
    # This class is currently very impoverished. It needs a lot more fields before it can correctly represent a PDF.
    text_contents = StringField(desc="The text-only contents of the PDF", required=True)

class ImageFile(File):
    """A file that contains an image."""
    text_description = StringField(desc="A text description of the image", required=False)

class EquationImage(ImageFile):
    """An image that contains a mathematical equation."""
    equation_text = StringField(desc="The text representation of the equation in the image", required=True)

class PlotImage(ImageFile):
    """An image that contains a plot, such as a graph or chart."""
    plot_description = StringField(desc="A description of the plot", required=True)

class URL(Schema):
    """A URL is a string that represents a web address."""
    url = StringField(desc="A URL", required=True)

class Download(Schema):
    """A download is a URL and the contents of the download."""
    url = StringField(desc="The URL of the download", required=True)
    content = BytesField(desc="The contents of the download", required=True)
    timestamp = StringField(desc="The timestamp of the download", required=True)

class WebPage(Schema):
    """A web page is a URL and the contents of the page."""
    url = StringField(desc="The URL of the web page", required=True)
    text = StringField(desc="The text contents of the web page", required=True)
    html = StringField(desc="The html contents of the web page", required=True)
    timestamp = StringField(desc="The timestamp of the download", required=True)