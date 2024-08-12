from palimpzest.constants import MAX_ROWS
from palimpzest.corelib.fields import *
from typing import Any, Dict, List

import json


class SchemaMetaclass(type):
    """
    This is a metaclass for our Schema class.
    """

    def __str__(cls) -> str:
        """
        Emit a string that contains the names of all the class members that are Fields.
        """
        # get attributes that are Fields
        fields = SchemaMetaclass.fieldNames(cls)

        return f"{cls.__name__}({', '.join(fields)})"

    def __eq__(cls, other) -> bool:
        """
        Equality function for the Schema which checks that the ordered fields and class names are the same.
        """
        cls_schema = SchemaMetaclass.getDesc(cls)
        other_schema = SchemaMetaclass.getDesc(other)

        return cls_schema == other_schema

    def __hash__(cls) -> int:
        """Hash function for the Schema which is a simple hash of its ordered Fields and class name."""
        ordered = SchemaMetaclass.getDesc(cls)

        return hash(ordered.encode())

    def fieldNames(cls) -> List[str]:
        """Return a list of the fields in this Schema"""
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith("__")]
        fields = [attr for attr in attributes if isinstance(getattr(cls, attr), Field)]

        return fields

    def getDesc(cls) -> str:
        """Return a description of the schema"""
        fields = SchemaMetaclass.fieldNames(cls)
        d = {k: hash(getattr(cls, k)) for k in fields}

        # TODO: this causes an exception why trying to use Schema in a type definition
        # e.g. TaskDescriptor = Tuple[str, Union[tuple, None], Schema, Schema]
        # will throw the following exception:
        #
        # File "/Users/matthewrusso/palimpzest/src/palimpzest/elements/elements.py", line 168, in getDesc
        #     d["__class__"] = o.__name__
        # AttributeError: '_SpecialForm' object has no attribute '__name__'
        #
        d["__class__"] = cls.__class__.__name__

        return json.dumps(d, sort_keys=True)

    def jsonSchema(cls) -> Dict[str, Any]:
        """The JSON representation of the Schema"""
        fields = SchemaMetaclass.fieldNames(cls)

        schema = {
            "properties": {},
            "required": [],
            "type": "object",
            "description": cls.__doc__,
        }
        for k in fields:
            if k.startswith("_"):
                continue
            v = getattr(cls, k)
            if v is None:
                continue

            schema["properties"][k] = v.jsonSchema()

            if v.required:
                schema["required"].append(k)
        return schema


# TODO: should we put the SchemaMetaclass functionality into Schema and make it a @dataclass?
class Schema(metaclass=SchemaMetaclass):
    """
    A Schema is defined by a set of named Fields. Much of the class is implemented
    in the SchemaMetaclass (which I need to ask Mike more about). Because Schema is a MetaClass,
    its fields are defined similar to how they are defined in a Python dataclass.

    For example, if you wanted to define a schema for research papers, you could define a schema
    with fields representing the paper's title, publication year, and publishing journal:

    ```python
    class ResearchPaper(Schema):
        paper_title = Field(desc="The title of a scientific paper", required=True)
        paper_year = Field(desc="The year the paper was published", required=True)
        paper_journal = Field(desc="The name of the journal that published the paper", required=False)
    ```

    Note that because not all papers are published in journals, this field might be optional (`required=False`).
    """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(desc={self._desc})"

    def asJSONStr(
        self, record_dict: Dict[str, Any], include_data_cols: bool = True
    ) -> str:
        """Return a JSON representation of a data record with this Schema"""
        if include_data_cols:
            record_dict["data type"] = str(self.__class__.__name__)
            record_dict["data type description"] = str(self.__class__.__doc__)

        return json.dumps(record_dict, indent=2)

    # TODO move logic from metaclass to here
    @classmethod
    def className(cls) -> str:
        """Return the name of this class"""
        return cls.__name__


# TODO: Under the definition of __eq__ in SchemaMetaclass, I think that an equality check like
#       Any([TextFile, PDFFile]) == TextFile will return `False`. I believe this is the behavior
#       that we want (these schemas are not the same), but I need to make sure that an equality
#       check between two instances of these schemas will not return `False` if they are both the
#       the same object (e.g. the same file of text), but one has Schema == Any([TextFile, PDFFile])
#       and the other has Schema = TextFile
#
#####################################################
# A Schema that can be one of multiple other kinds of Schemas.
# For example, I might want to process Any([PDF, WordDoc, TextFile])
#####################################################
class Any(Schema):
    """
    This represents ANY of the specified Schemas. For example, you may not know if a document
    is a PDF or a Word document, but you know it's one of those two.
    """

    def __init__(self, possibleSchemas: List[Schema], desc: str):
        super().__init__(desc=desc)
        self._possibleSchemas = possibleSchemas

    @property
    def children(self) -> List[Schema]:
        return self._possibleSchemas


class OperatorDerivedSchema(Schema):
    """Schema defined by an operator, e.g., a join or a group by"""


###################################################################################
# "Core" useful Schemas. These are Schemas that almost everyone will need.
# File, TextFile, Image, PDF, etc.
###################################################################################
class SourceRecord(Schema):
    """
    Schema used inside of Execution.execute_dag to produce a candidate for operators
    which implement the BaseScan or CacheScan logical operators.
    """
    idx = NumericField(desc="The scan index of the record", required=True)
    get_item_fn = CallableField(desc="The get_item() function from the DataSource", required=True)
    cardinality = StringField(desc="The cardinality of the datasource", required=True)


class File(Schema):
    """
    A File is defined by two Fields:
    - the filename (string)
    - the contents of the file (bytes)
    """

    filename = StringField(desc="The UNIX-style name of the file", required=True)
    contents = BytesField(desc="The contents of the file", required=True)


class Number(Schema):
    """Just a number. Often used for aggregates"""

    value = NumericField(desc="A single number", required=True)


class TextFile(File):
    """A text file is a File that contains only text. No binary data."""


class RawJSONObject(Schema):
    """A JSON object, which is a dictionary of key-value pairs."""

    json = StringField(desc="String representation of a JSON object", required=True)


class PDFFile(File):
    """A PDF file is a File that is a PDF. It has specialized fields, font information, etc."""

    # This class is currently very impoverished. It needs a lot more fields before it can correctly represent a PDF.
    text_contents = StringField(desc="The text-only contents of the PDF", required=True)


class ImageFile(File):
    """A file that contains an image."""

    text_description = StringField(
        desc="A text description of the image", required=False
    )


class EquationImage(ImageFile):
    """An image that contains a mathematical equation."""

    equation_text = StringField(
        desc="The text representation of the equation in the image", required=True
    )


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

    number_sheets = NumericField(
        desc="The number of sheets in the Excel file", required=True
    )
    sheet_names = ListField(
        element_type=NumericField,
        desc="The names of the sheets in the Excel file",
        required=True,
    )


class Table(Schema):
    """A Table is an object composed of a header and rows."""

    filename = StringField(
        desc="The name of the file the table was extracted from", required=False
    )
    name = StringField(desc="The name of the table", required=False)
    header = ListField(
        element_type=StringField, desc="The header of the table", required=True
    )
    # TODO currently no support for nesting data records on data records
    rows = ListField(
        element_type=ListField, desc="The rows of the table", required=True
    )

    def asJSONStr(self, record_dict: Dict[str, Any], *args, **kwargs) -> str:
        """Return a JSON representation of an instantiated object of this Schema"""
        # Take the rows in the record_dict and turn them into comma separated strings
        rows = []
        # only sample the first MAX_ROWS
        for i, row in enumerate(record_dict["rows"][:MAX_ROWS]):
            rows += [",".join(map(str, row)) + "\n"]
        record_dict["rows"] = rows
        record_dict["rows"] = ""
        header = ",".join(record_dict["header"])
        record_dict["header"] = header

        return super(Table, self).asJSONStr(record_dict, *args, **kwargs)
