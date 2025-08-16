from typing import Any

import pytest
from pydantic import BaseModel, Field

from palimpzest.core.lib.schemas import ImageFilepath, TextFile


### SCHEMAS ###
@pytest.fixture
def email_schema():
    class Email(TextFile):
        """Represents an email, which in practice is usually from a text file"""
        sender: str = Field(description="The email address of the sender")
        subject: str = Field(description="The subject of the email")

    return Email


@pytest.fixture
def real_estate_listing_files_schema():
    class RealEstateListingFiles(BaseModel):
        """The source text and image data for a real estate listing."""
        listing: str = Field(description="The name of the listing")
        text_content: str = Field(description="The content of the listing's text description")
        image_filepaths: list[ImageFilepath] = Field(description="A list of the filepaths for each image of the listing")

    return RealEstateListingFiles


@pytest.fixture
def text_real_estate_listing_schema(real_estate_listing_files_schema):
    class TextRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text."""
        address: str = Field(description="The address of the property")
        price: int | float = Field(description="The listed price of the property")

    return TextRealEstateListing


@pytest.fixture
def image_real_estate_listing_schema(real_estate_listing_files_schema):
    class ImageRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text and images."""

        is_modern_and_attractive: bool = Field(
            description="True if the home interior design is modern and attractive and False otherwise"
        )
        has_natural_sunlight: bool = Field(
            description="True if the home interior has lots of natural sunlight and False otherwise"
        )

    return ImageRealEstateListing


@pytest.fixture
def room_real_estate_listing_schema(real_estate_listing_files_schema):
    class RoomRealEstateListing(real_estate_listing_files_schema):
        """Represents a room shown in the image of a real estate listing."""

        room: str = Field(
            description='The room shown in an image. Room can be one of ["living_room", "kitchen", "bedroom", "other"]',
        )

    return RoomRealEstateListing


@pytest.fixture
def case_data_schema():
    class CaseData(BaseModel):
        """An individual row extracted from a table containing medical study data."""

        case_submitter_id: Any = Field(description="The ID of the case")
        age_at_diagnosis: Any = Field(description="The age of the patient at the time of diagnosis")
        race: Any = Field(
            description="An arbitrary classification of a taxonomic group that is a division of a species.",
        )
        ethnicity: Any = Field(
            description="Whether an individual describes themselves as Hispanic or Latino or not.",
        )
        gender: Any = Field(description="Text designations that identify gender.")
        vital_status: Any = Field(description="The vital status of the patient")
        ajcc_pathologic_t: Any = Field(description="The AJCC pathologic T")
        ajcc_pathologic_n: Any = Field(description="The AJCC pathologic N")
        ajcc_pathologic_stage: Any = Field(description="The AJCC pathologic stage")
        tumor_grade: Any = Field(description="The tumor grade")
        tumor_focality: Any = Field(description="The tumor focality")
        tumor_largest_dimension_diameter: Any = Field(description="The tumor largest dimension diameter")
        primary_diagnosis: Any = Field(description="The primary diagnosis")
        morphology: Any = Field(description="The morphology")
        tissue_or_organ_of_origin: Any = Field(description="The tissue or organ of origin")
        filename: Any = Field(description="The name of the file the record was extracted from")
        study: Any = Field(
            description="The last name of the author of the study, from the table name",
        )

    return CaseData

@pytest.fixture
def foobar_schema():
    class FooBar(BaseModel):
        foo: Any = Field("foo")
        bar: Any = Field("bar")

    return FooBar

@pytest.fixture
def baz_schema():
    class Baz(BaseModel):
        baz: Any = Field("baz")

    return Baz
