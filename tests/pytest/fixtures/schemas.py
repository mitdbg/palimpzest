import pytest

from palimpzest.core.lib.fields import BooleanField, Field, ImageFilepathField, ListField, NumericField, StringField
from palimpzest.core.lib.schemas import Schema, TextFile


### SCHEMAS ###
@pytest.fixture
def email_schema():
    class Email(TextFile):
        """Represents an email, which in practice is usually from a text file"""

        sender = Field(desc="The email address of the sender")
        subject = Field(desc="The subject of the email")

    return Email


@pytest.fixture
def real_estate_listing_files_schema():
    class RealEstateListingFiles(Schema):
        """The source text and image data for a real estate listing."""

        listing = StringField(desc="The name of the listing")
        text_content = StringField(desc="The content of the listing's text description")
        image_filepaths = ListField(
            element_type=ImageFilepathField,
            desc="A list of the filepaths for each image of the listing",
        )

    return RealEstateListingFiles


@pytest.fixture
def text_real_estate_listing_schema(real_estate_listing_files_schema):
    class TextRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text."""

        address = StringField(desc="The address of the property")
        price = NumericField(desc="The listed price of the property")

    return TextRealEstateListing


@pytest.fixture
def image_real_estate_listing_schema(real_estate_listing_files_schema):
    class ImageRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text and images."""

        is_modern_and_attractive = BooleanField(
            desc="True if the home interior design is modern and attractive and False otherwise"
        )
        has_natural_sunlight = BooleanField(
            desc="True if the home interior has lots of natural sunlight and False otherwise"
        )

    return ImageRealEstateListing


@pytest.fixture
def room_real_estate_listing_schema(real_estate_listing_files_schema):
    class RoomRealEstateListing(real_estate_listing_files_schema):
        """Represents a room shown in the image of a real estate listing."""

        room = StringField(
            desc='The room shown in an image. Room can be one of ["living_room", "kitchen", "bedroom", "other"]',
        )

    return RoomRealEstateListing


@pytest.fixture
def case_data_schema():
    class CaseData(Schema):
        """An individual row extracted from a table containing medical study data."""

        case_submitter_id = Field(desc="The ID of the case")
        age_at_diagnosis = Field(desc="The age of the patient at the time of diagnosis")
        race = Field(
            desc="An arbitrary classification of a taxonomic group that is a division of a species.",
        )
        ethnicity = Field(
            desc="Whether an individual describes themselves as Hispanic or Latino or not.",
        )
        gender = Field(desc="Text designations that identify gender.")
        vital_status = Field(desc="The vital status of the patient")
        ajcc_pathologic_t = Field(desc="The AJCC pathologic T")
        ajcc_pathologic_n = Field(desc="The AJCC pathologic N")
        ajcc_pathologic_stage = Field(desc="The AJCC pathologic stage")
        tumor_grade = Field(desc="The tumor grade")
        tumor_focality = Field(desc="The tumor focality")
        tumor_largest_dimension_diameter = Field(desc="The tumor largest dimension diameter")
        primary_diagnosis = Field(desc="The primary diagnosis")
        morphology = Field(desc="The morphology")
        tissue_or_organ_of_origin = Field(desc="The tissue or organ of origin")
        # tumor_code = Field(desc="The tumor code")
        filename = Field(desc="The name of the file the record was extracted from")
        study = Field(
            desc="The last name of the author of the study, from the table name",
        )

    return CaseData

@pytest.fixture
def foobar_schema():
    class FooBar(Schema):
        foo = Field("foo")
        bar = Field("bar")

    return FooBar

@pytest.fixture
def baz_schema():
    class Baz(Schema):
        baz = Field("baz")

    return Baz
