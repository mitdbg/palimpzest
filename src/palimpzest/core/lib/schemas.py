from __future__ import annotations

import json
from typing import Any as TypingAny

from palimpzest.constants import MAX_ROWS
from palimpzest.core.lib.fields import BytesField, CallableField, Field, ListField, NumericField, StringField


class SchemaMetaclass(type):
    """
    This is a metaclass for our Schema class.
    """

    # TODO: test whether we can move this into Schema
    def __str__(cls) -> str:
        """
        Emit a string that contains the names of all the class members that are Fields.
        """
        # get attributes that are Fields
        fields = SchemaMetaclass.field_names(cls)

        return f"{cls.__name__}({', '.join(fields)})"


    # TODO: test whether we can move this into Schema
    def __eq__(cls, other) -> bool:
        """
        Equality function for the Schema which checks that the ordered fields and class names are the same.
        """
        cls_schema = SchemaMetaclass.get_desc(cls)
        other_schema = SchemaMetaclass.get_desc(other)

        return cls_schema == other_schema


    # TODO: test whether we can move this into Schema
    def __hash__(cls) -> int:
        """Hash function for the Schema which is a simple hash of its ordered Fields and class name."""
        ordered = SchemaMetaclass.get_desc(cls)

        return hash(ordered.encode())


    # TODO: test whether we can move this into Schema
    def field_names(cls, unique=False, id="") -> list[str]:
        """
        Return a list of the fields in this Schema. The `unique` argument is used to determine if the
        class name should be prefixed to the field name for unique identification. The `id` argument is
        used to provide a unique identifier for the class name.
        """
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith("__")]
        prefix = f"{cls.__name__}.{id}." if unique else ""
        fields = [prefix + attr for attr in attributes if isinstance(getattr(cls, attr), Field)]
        return fields


    # TODO: test whether we can move this into Schema
    def field_desc_map(cls, unique=False, id="") -> dict[str, str]:
        """
        Return a mapping from field names to their descriptions. The `unique` argument is used to determine if the
        class name should be prefixed to the field name for unique identification. The `id` argument is
        used to provide a unique identifier for the class name.
        """
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith("__")]
        prefix = f"{cls.__name__}.{id}." if unique else ""
        field_desc_map = {prefix + attr: attr.desc for attr in attributes if isinstance(getattr(cls, attr), Field)}
        return field_desc_map


    # TODO: test whether we can move this into Schema
    def field_map(cls, unique=False, id="") -> dict[str, Field]:
        """
        Return a mapping from field names to their field types. The `unique` argument is used to determine if the
        class name should be prefixed to the field name for unique identification. The `id` argument is used to
        provide a unique identifier for the class name.
        """
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith("__")]
        prefix = f"{cls.__name__}.{id}." if unique else ""
        field_map = {prefix + attr: attr for attr in attributes if isinstance(getattr(cls, attr), Field)}
        return field_map


    # TODO: test whether we can move this into Schema
    def get_desc(cls) -> str:
        """Return a description of the schema"""
        fields = SchemaMetaclass.field_names(cls)
        d = {k: hash(getattr(cls, k)) for k in fields}
        d["__class__"] = cls.__class__.__name__

        return json.dumps(d, sort_keys=True)

    # TODO: test whether we can move this into Schema
    def json_schema(cls) -> dict[str, TypingAny]:
        """The JSON representation of the Schema"""
        fields = SchemaMetaclass.field_names(cls)

        schema = {
            "fields": {},
            "type": "object",
            "description": cls.__doc__,
        }
        for k in fields:
            if k.startswith("_"):
                continue
            v = getattr(cls, k)
            if v is None:
                continue

            schema["fields"][k] = v.json_schema()

        return schema


# TODO: should we put the SchemaMetaclass functionality into Schema and make it a @dataclass?
class Schema(metaclass=SchemaMetaclass):
    """
    A Schema is defined by a set of named Fields. Much of the class is implemented in the SchemaMetaclass.
    Because Schema is a MetaClass, its fields are defined similar to how they are defined in a Python dataclass.

    For example, if you wanted to define a schema for research papers, you could define a schema
    with fields representing the paper's title, publication year, and publishing journal:

    ```python
    class ResearchPaper(Schema):
        paper_title = Field(desc="The title of a scientific paper")
        paper_year = Field(desc="The year the paper was published")
        paper_journal = Field(desc="The name of the journal that published the paper")
    ```
    """

    def __init__(self, desc: str | None = None):
        self._desc = "" if desc is None else desc

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(desc={self._desc})"

    def field_to_json(self, field_name: str, field_value: TypingAny) -> TypingAny:
        """Return a representation of the specified field which will be used in its conversion to JSON"""
        return field_value

    def union(self, other_schema: Schema, keep_duplicates: bool = False) -> Schema:
        """Return the union of this schema with the other_schema"""
        # construct the new schema name
        schema_name = self.__class__.__name__
        other_schema_name = other_schema.class_name()
        new_schema_name = f"Union[{schema_name}, {other_schema_name}]"

        # construct new schema description
        new_desc = (
            f"The union of {schema_name} and {other_schema_name}\n\n"
            f"{schema_name}:\n{self._desc}\n\n"
            f"{other_schema_name}:\n{other_schema._desc}"
        )

        # construct new lists of field names, types, and descriptions
        # NOTE: we don't need to use unique field names because they will be injected with an ID at runtime
        new_field_names, new_field_types, new_field_descs = [], [], []
        this_field_map = self.__class__.field_map()
        for field_name, field in this_field_map.items():
            new_field_names.append(field_name)
            new_field_types.append(field)
            new_field_descs.append(field._desc)

        other_field_map = other_schema.__class__.field_map()
        for field_name, field in other_field_map.items():
            new_field_names.append(field_name)
            new_field_types.append(field)
            new_field_descs.append(field._desc)

        # rename duplicate fields if we are keeping duplicates
        if keep_duplicates:
            dup_new_field_names = []
            for left_idx, left_field_name in enumerate(new_field_names):
                # see if there's a duplicate field name
                matching_field = False
                for right_idx in range(left_idx + 1, len(new_field_names)):
                    right_field_name = new_field_names[right_idx]
                    if left_field_name == right_field_name:
                        matching_field = True
                        break
                
                # if theres a matching field, add them both with their schema names
                if matching_field:
                    dup_new_field_names.append(schema_name + "_" + left_field_name)
                    dup_new_field_names.append(other_schema_name + "_" + left_field_name)
                else:
                    dup_new_field_names.append(left_field_name)

        # Generate the schema class dynamically
        attributes = {"__doc__": new_desc}
        for field_name, field_type, field_desc in zip(new_field_names, new_field_types, new_field_descs):
            attributes[field_name] = field_type(desc=field_desc)

        # Create the class dynamically
        return type(new_schema_name, (Schema,), attributes)


    def project(self, project_cols: list[str]) -> Schema:
        """Return a projection of this schema with only the project_cols"""
        # construct the new schema name
        schema_name = self.__class__.__name__
        new_schema_name = f"Project[{schema_name}]"

        # construct new schema description
        new_desc = f"A projection of {schema_name} which only contains the fields {project_cols}"

        # make sure projection column names are shortened
        project_cols = [field_name.split(".")[-1] for field_name in project_cols]

        # construct new lists of field names, types, and descriptions
        # NOTE: we don't need to use unique field names because they will be injected with an ID at runtime
        new_field_names, new_field_types, new_field_descs = [], [], []
        for field_name, field in self.__class__.field_map().items():
            if field_name in project_cols:
                new_field_names.append(field_name)
                new_field_types.append(field)
                new_field_descs.append(field._desc)

        # Generate the schema class dynamically
        attributes = {"__doc__": new_desc}
        for field_name, field_type, field_desc in zip(new_field_names, new_field_types, new_field_descs):
            attributes[field_name] = field_type(desc=field_desc)

        # Create the class dynamically
        return type(new_schema_name, (Schema,), attributes)

    @classmethod
    def class_name(cls) -> str:
        """Return the name of this class"""
        return cls.__name__


###################################################################################
# "Core" useful Schemas. These are Schemas that almost everyone will need.
# File, TextFile, Image, PDF, etc.
###################################################################################

# First-level Schema's
class Download(Schema):
    """A download is a URL and the contents of the download."""

    url = StringField(desc="The URL of the download")
    content = BytesField(desc="The contents of the download")
    timestamp = StringField(desc="The timestamp of the download")


class File(Schema):
    """
    A File is defined by two Fields:
    - the filename (string)
    - the contents of the file (bytes)
    """

    filename = StringField(desc="The UNIX-style name of the file")
    contents = BytesField(desc="The contents of the file")


class Number(Schema):
    """Just a number. Often used for aggregates"""

    value = NumericField(desc="A single number")


class OperatorDerivedSchema(Schema):
    """Schema defined by an operator, e.g., a join or a group by"""


class RawJSONObject(Schema):
    """A JSON object, which is a dictionary of key-value pairs."""

    json = StringField(desc="String representation of a JSON object")


class SourceRecord(Schema):
    """
    Schema used inside of Execution.execute_dag to produce a candidate for operators
    which implement the BaseScan or CacheScan logical operators.
    """

    idx = NumericField(desc="The scan index of the record")
    get_item_fn = CallableField(desc="The get_item() function from the DataSource")


class Table(Schema):
    """A Table is an object composed of a header and rows."""

    filename = StringField(desc="The name of the file the table was extracted from")
    name = StringField(desc="The name of the table")
    header = ListField(element_type=StringField, desc="The header of the table")
    # TODO currently no support for nesting data records on data records
    rows = ListField(element_type=ListField, desc="The rows of the table")

    def field_to_json(self, field_name: str, field_value: TypingAny) -> TypingAny:
        """Return a truncated JSON representation for `rows` and a string representation for `header`"""
        # take the first MAX_ROWS rows in the record_dict and turn them into comma separated strings
        if field_name == "rows":
            return [",".join(map(str, row)) + "\n" for row in field_value[:MAX_ROWS]]

        elif field_name == "header":
            return ",".join(field_value)

        return field_value


class URL(Schema):
    """A URL is a string that represents a web address."""

    url = StringField(desc="A URL")


class WebPage(Schema):
    """A web page is a URL and the contents of the page."""

    # url = StringField(desc="The URL of the web page")
    text = StringField(desc="The text contents of the web page")
    html = StringField(desc="The html contents of the web page")
    timestamp = StringField(desc="The timestamp of the download")


# Second-level Schemas
class ImageFile(File):
    """A file that contains an image."""

    text_description = StringField(desc="A text description of the image")


class PDFFile(File):
    """A PDF file is a File that is a PDF. It has specialized fields, font information, etc."""

    # This class is currently very impoverished. It needs a lot more fields before it can correctly represent a PDF.
    text_contents = StringField(desc="The text-only contents of the PDF")


class TextFile(File):
    """A text file is a File that contains only text. No binary data."""


class XLSFile(File):
    """An XLS file is a File that contains one or more Excel spreadsheets."""

    number_sheets = NumericField(desc="The number of sheets in the Excel file")
    sheet_names = ListField(
        element_type=NumericField,
        desc="The names of the sheets in the Excel file",
    )


# Third-level Schemas
class EquationImage(ImageFile):
    """An image that contains a mathematical equation."""

    equation_text = StringField(desc="The text representation of the equation in the image")


class PlotImage(ImageFile):
    """An image that contains a plot, such as a graph or chart."""

    plot_description = StringField(desc="A description of the plot")
