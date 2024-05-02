from palimpzest.constants import MAX_ROWS
from palimpzest.elements import BytesField, Schema, StringField, File, NumericField, ListField
import json

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

class XLSFile(File):
    """An XLS file is a File that contains one or more Excel spreadsheets."""
    number_sheets = NumericField(desc="The number of sheets in the Excel file", required=True)
    sheet_names = ListField(element_type=NumericField, desc="The names of the sheets in the Excel file", required=True)

# class TabularRow(Schema):  
    # """A Row is a list of cells. For simplicity, we assume that all cell values are strings."""
    # cells = ListField(element_type=StringField, desc="The cells in the row", required=True)

class Table(Schema):
    """A Table is an object composed of a header and rows."""
    filename = StringField(desc="The name of the file the table was extracted from", required=False)
    name = StringField(desc="The name of the table", required=False)
    header = ListField(element_type=StringField, desc="The header of the table", required=True)
    # TODO currently no support for nesting data records on data records
    rows = ListField(element_type=ListField, desc="The rows of the table", required=True)

    def asJSON(self, value_dict, *args, **kwargs) -> str:
        """Return a JSON representation of an instantiated object of this Schema"""
        # Take the value_dict for the rows and make them into comma separated strings
        dct = value_dict

        rows = []
        for i, row in enumerate(dct["rows"][:MAX_ROWS]): # only sample the first MAX_ROWS
            rows += [",".join(row) + "\n"]
        dct["rows"] = rows

        header = ",".join(dct["header"])
        dct["header"] = header

        return super(Table, self).asJSON(dct, *args, **kwargs)