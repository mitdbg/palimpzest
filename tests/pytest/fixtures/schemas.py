import pytest
import palimpzest as pz

### SCHEMAS ###
@pytest.fixture
def email_schema():
    class Email(pz.TextFile):
        """Represents an email, which in practice is usually from a text file"""
        sender = pz.Field(desc="The email address of the sender", required=True)
        subject = pz.Field(desc="The subject of the email", required=True)
    
    return Email

@pytest.fixture
def real_estate_listing_files_schema():
    class RealEstateListingFiles(pz.Schema):
        """The source text and image data for a real estate listing."""
        listing = pz.StringField(desc="The name of the listing", required=True)
        text_content = pz.StringField(
            desc="The content of the listing's text description", required=True
        )
        image_filepaths = pz.ListField(
            element_type=pz.StringField,
            desc="A list of the filepaths for each image of the listing",
            required=True,
        )

    return RealEstateListingFiles

@pytest.fixture
def text_real_estate_listing_schema(real_estate_listing_files_schema):
    class TextRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text."""
        address = pz.StringField(desc="The address of the property")
        price = pz.NumericField(desc="The listed price of the property")

    return TextRealEstateListing

@pytest.fixture
def image_real_estate_listing_schema(real_estate_listing_files_schema):
    class ImageRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text and images."""
        is_modern_and_attractive = pz.BooleanField(
            desc="True if the home interior design is modern and attractive and False otherwise"
        )
        has_natural_sunlight = pz.BooleanField(
            desc="True if the home interior has lots of natural sunlight and False otherwise"
        )

    return ImageRealEstateListing

@pytest.fixture
def room_real_estate_listing_schema(real_estate_listing_files_schema):
    class RoomRealEstateListing(real_estate_listing_files_schema):
        """Represents a room shown in the image of a real estate listing."""
        room = pz.StringField(
            desc="The room shown in an image. Room can be one of [\"living_room\", \"kitchen\", \"bedroom\", \"other\"]",
            required=True,
        )

    return RoomRealEstateListing

@pytest.fixture
def case_data_schema():
    class CaseData(pz.Schema):
        """An individual row extracted from a table containing medical study data."""
        case_submitter_id = pz.Field(desc="The ID of the case", required=True)
        age_at_diagnosis = pz.Field(
            desc="The age of the patient at the time of diagnosis", required=False
        )
        race = pz.Field(
            desc="An arbitrary classification of a taxonomic group that is a division of a species.",
            required=False,
        )
        ethnicity = pz.Field(
            desc="Whether an individual describes themselves as Hispanic or Latino or not.",
            required=False,
        )
        gender = pz.Field(desc="Text designations that identify gender.", required=False)
        vital_status = pz.Field(desc="The vital status of the patient", required=False)
        ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
        ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
        ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
        tumor_grade = pz.Field(desc="The tumor grade", required=False)
        tumor_focality = pz.Field(desc="The tumor focality", required=False)
        tumor_largest_dimension_diameter = pz.Field(
            desc="The tumor largest dimension diameter", required=False
        )
        primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
        morphology = pz.Field(desc="The morphology", required=False)
        tissue_or_organ_of_origin = pz.Field(
            desc="The tissue or organ of origin", required=False
        )
        # tumor_code = pz.Field(desc="The tumor code", required=False)
        filename = pz.Field(
            desc="The name of the file the record was extracted from", required=False
        )
        study = pz.Field(
            desc="The last name of the author of the study, from the table name",
            required=False,
        )

    return CaseData

@pytest.fixture
def foobar_schema():
    class FooBar(pz.Schema):
        foo = pz.Field("foo")
        bar = pz.Field("bar")

    return FooBar

@pytest.fixture
def baz_schema():
    class Baz(pz.Schema):
        baz = pz.Field("baz")

    return Baz
