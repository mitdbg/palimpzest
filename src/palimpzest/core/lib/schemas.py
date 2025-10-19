from __future__ import annotations

import sys
from typing import Any, TypeAliasType

import pandas as pd
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from palimpzest.utils.hash_helpers import hash_for_serialized_dict

# DEFINITIONS
PANDAS_DTYPE_TO_PYDANTIC = {
    "object": str,
    "bool": bool,
    "int64": int,
    "float64": float,
}

# IMAGE TYPES
ImageFilepath = TypeAliasType('ImageFilepath', str)
ImageBase64 = TypeAliasType('ImageBase64', str)
ImageURL = TypeAliasType('ImageURL', str)

# AUDIO TYPES
AudioFilepath = TypeAliasType('AudioFilepath', str)
AudioBase64 = TypeAliasType('AudioBase64', str)

IMAGE_LIST_FIELD_TYPES = [
    list[ImageBase64],
    list[ImageFilepath],
    list[ImageURL],
    list[ImageBase64] | None,
    list[ImageFilepath] | None,
    list[ImageURL] | None,
    list[ImageBase64] | Any,
    list[ImageFilepath] | Any,
    list[ImageURL] | Any,
]
IMAGE_FIELD_TYPES = IMAGE_LIST_FIELD_TYPES + [
    ImageBase64, ImageFilepath, ImageURL,
    ImageBase64 | None, ImageFilepath | None, ImageURL | None,
    ImageBase64 | Any, ImageFilepath | Any, ImageURL | Any,
]
AUDIO_LIST_FIELD_TYPES = [
    list[AudioBase64],
    list[AudioFilepath],
    list[AudioBase64] | None,
    list[AudioFilepath] | None,
    list[AudioBase64] | Any,
    list[AudioFilepath] | Any,
]
AUDIO_FIELD_TYPES = AUDIO_LIST_FIELD_TYPES + [
    AudioBase64, AudioFilepath,
    AudioBase64 | None, AudioFilepath | None,
    AudioBase64 | Any, AudioFilepath | Any,
]


def get_schema_field_names(schema: type[BaseModel], id: str | None = None) -> list[str]:
    """Return the field names of a Pydantic model."""
    return list(schema.model_fields) if id is None else [f"{schema.__name__}.{id}.{field_name}" for field_name in schema.model_fields]


def _create_pickleable_model(fields: dict[str, tuple[type, FieldInfo]]) -> type[BaseModel]:
    """Create a Pydantic model that can be pickled."""
    # create unique name for the unioned model
    new_schema_name = f"Schema{sorted(fields.keys())}"
    new_schema_id = hash_for_serialized_dict({
        field_name: {"annotation": str(annotation), "default": str(field.default), "description": field.description}
        for field_name, (annotation, field) in fields.items()
    })

    # if this class already exists, get it from the module and return
    module = sys.modules[__name__]
    if hasattr(module, new_schema_id):
        return getattr(module, new_schema_id)

    # create the class dynamically
    new_model = create_model(new_schema_name, **fields)

    # register it in the module's namespace so pickle can find it
    module = sys.modules[__name__]
    setattr(module, new_schema_id, new_model)
    new_model.__module__ = module.__name__

    return new_model


def relax_schema(model: type[BaseModel]) -> type[BaseModel]:
    """Updates the type annotation for every field in the BaseModel to include typing.Any"""
    fields = {}
    for field_name, field in model.model_fields.items():
        fields[field_name] = (field.annotation | Any, field)

    return _create_pickleable_model(fields)


def project(model: type[BaseModel], project_fields: list[str]) -> type[BaseModel]:
    """Project a Pydantic model to only the specified columns."""
    # make sure projection column names are shortened
    project_fields = [field_name.split(".")[-1] for field_name in project_fields]

    # build up the fields for the new schema
    fields = {}
    for field_name, field in model.model_fields.items():
        if field_name in project_fields:
            fields[field_name] = (field.annotation, field)

    # create and return the new schema
    return _create_pickleable_model(fields)


def create_schema_from_fields(fields: list[dict]) -> type[BaseModel]:
    """Create a Pydantic model from a list of fields."""
    fields_ = {}
    for field in fields:
        assert "name" in field, "fields must contain a 'name' key"
        assert "type" in field, "fields must contain a 'type' key"
        assert "desc" in field or "description" in field, "fields must contain a 'description' key"

        # for backwards compatability, rename "desc" to "description"
        if "desc" in field:
            field["description"] = field.pop("desc")
        field_name = field["name"]
        field_type = field["type"]
        fields_[field_name] = (field_type, Field(**{k: v for k, v in field.items() if k not in ["name", "type"]}))

    return _create_pickleable_model(fields_)


def create_schema_from_df(df: pd.DataFrame) -> type[BaseModel]:
    """Create a Pydantic model from a Pandas DataFrame."""
    fields = {}
    for column, dtype in zip(df.columns, df.dtypes):
        column = f"column_{column}" if isinstance(column, int) else column
        field_desc = f"The {column} column from an input DataFrame"
        annotation = PANDAS_DTYPE_TO_PYDANTIC.get(str(dtype), Any)
        fields[column] = (annotation, Field(description=field_desc))

    # create and return the new schema
    return _create_pickleable_model(fields)


def union_schemas(models: list[type[BaseModel]], join: bool = False, on: list[str] | None = None) -> type[BaseModel]:
    """Union multiple Pydantic models into a single model."""
    # convert on to empty list if None
    if on is None:
        on = []

    # build up the fields for the new schema
    fields = {}
    for model in models:
        for field_name, field in model.model_fields.items():
            # for non-join unions, make sure duplicate fields have the same type
            if not join and field_name in fields:
                assert fields[field_name][0] == field.annotation, f"Field {field_name} has different types in different models"

            # for joins with "on" specified, no need to rename fields in "on"
            elif join and field_name in on and field_name in fields:
                continue

            # otherwise, rename duplicate fields by appending _right
            elif join and field_name in fields:
                while field_name in fields:
                    field_name = f"{field_name}_right"

            # add the field to the new schema
            fields[field_name] = (field.annotation, field)

    # create and return the new schema
    return _create_pickleable_model(fields)

###################################################################################
# "Core" useful Schemas. These are Schemas that almost everyone will need.
# File, TextFile, Image, PDF, etc.
###################################################################################


# First-level Schema's
class DefaultSchema(BaseModel):
    """Store context data."""
    value: Any = Field(description="The value of the input data")

class Download(BaseModel):
    """A download is a URL and the contents of the download."""
    url: str = Field(description="The URL of the download")
    content: bytes = Field(description="The contents of the download")
    timestamp: str = Field(description="The timestamp of the download")

class File(BaseModel):
    """
    A File is defined by two Fields:
    - the filename (string)
    - the contents of the file (bytes)
    """
    filename: str = Field(description="The UNIX-style name of the file")
    contents: bytes = Field(description="The contents of the file")

class TextFile(BaseModel):
    """A text file is a File that contains only text. No binary data."""
    filename: str = Field(description="The UNIX-style name of the file")
    contents: str = Field(description="The contents of the file")

class Average(BaseModel):
    average: float = Field(description="The average value of items in the dataset")

class Count(BaseModel):
    count: int = Field(description="The count of items in the dataset")

class Sum(BaseModel):
    sum: int = Field(description="The summation of items in the dataset")

class Min(BaseModel):
    min: int | float = Field(description="The minimum value of some items in the dataset")

class Max(BaseModel):
    max: int | float = Field(description="The maximum value of some items in the dataset")

class OperatorDerivedSchema(BaseModel):
    """Schema defined by an operator, e.g., a join or a group by"""

class Table(BaseModel):
    """A Table is an object composed of a header and rows."""
    filename: str = Field(description="The name of the file the table was extracted from")
    name: str = Field(description="The name of the table")
    header: list[str] = Field(description="The header of the table")
    rows: list[list] = Field(description="The rows of the table")

class URL(BaseModel):
    """A URL is a string that represents a web address."""
    url: str = Field(description="A URL")

class WebPage(BaseModel):
    """A web page is a URL and the contents of the page."""
    text: str = Field(description="The text contents of the web page")
    html: str = Field(description="The html contents of the web page")
    timestamp: str = Field(description="The timestamp of the download")
    filename: str = Field(description="The name of the file the web page was downloaded from")

# Second-level Schemas
class ImageFile(File):
    """A file that contains an image."""
    contents: ImageBase64 = Field(description="The contents of the image encoded as a base64 string")

class AudioFile(File):
    """A file that contains audio."""
    contents: AudioBase64 = Field(description="The contents of an audio recording encoded as a base64 string")

class PDFFile(File):
    """A PDF file is a File that is a PDF. It has specialized fields, font information, etc."""
    # This class is currently very impoverished. It needs a lot more fields before it can correctly represent a PDF.
    text_contents: str = Field(description="The text-only contents of the PDF")

class XLSFile(File):
    """An XLS file is a File that contains one or more Excel spreadsheets."""
    number_sheets: int = Field(description="The number of sheets in the Excel file")
    sheet_names: list[str] = Field(description="The names of the sheets in the Excel file")

# Third-level Schemas
class EquationImage(ImageFile):
    """An image that contains a mathematical equation."""
    equation_text: str = Field(description="The text representation of the equation in the image")

class PlotImage(ImageFile):
    """An image that contains a plot, such as a graph or chart."""
    plot_description: str = Field(description="A description of the plot")
