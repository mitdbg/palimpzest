#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import palimpzest as pz
from palimpzest.execution import PipelinedSingleThreadSentinelExecution
from palimpzest.utils import getModels, udfs

# Addresses far from MIT; we use a simple lookup like this to make the
# experiments re-producible w/out needed a Google API key for geocoding lookups
FAR_AWAY_ADDRS = [
    "Melcher St",
    "Sleeper St",
    "437 D St",
    "Seaport Blvd",
    "50 Liberty Dr",
    "Telegraph St",
    "Columbia Rd",
    "E 6th St",
    "E 7th St",
    "E 5th St",
]


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)


class EmailSender(pz.TextFile):
    sender = pz.StringField(
        desc="The email address of the sender", required=True
    )


class EmailSubject(pz.TextFile):
    subject = pz.StringField(desc="The subject of the email", required=True)


class EmailCC(pz.TextFile):
    cc_list = pz.StringField(
        desc="The list of people cc'ed on the email, if any", required=True
    )


class EmailBCC(pz.TextFile):
    bcc_list = pz.StringField(
        desc="The list of people bcc'ed on the email, if any", required=True
    )


class EmailMeetings(pz.TextFile):
    meetings = pz.StringField(
        desc="The time and place of any meetings described in the email.",
        required=True,
    )


class EmailSummary(pz.TextFile):
    summary = pz.StringField(
        desc="A one sentence summary of the email", required=True
    )


class EmailSentiment(pz.TextFile):
    sentiment = pz.StringField(
        desc='The sentiment of the email, one of ["positive", "negative", "neutral"]'
    )


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
    gender = pz.Field(
        desc="Text designations that identify gender.", required=False
    )
    vital_status = pz.Field(
        desc="The vital status of the patient", required=False
    )
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(
        desc="The AJCC pathologic stage", required=False
    )
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
        desc="The name of the file the record was extracted from",
        required=False,
    )
    study = pz.Field(
        desc="The last name of the author of the study, from the table name",
        required=False,
    )


# TODO: it might not be obvious to a new user how to write/split up a schema for multimodal file data;
#       under our current setup, we have one schema which represents a file (e.g. pz.File), so the equivalent
#       here is to have a schema which represents the different (sets of) files, but I feel like users
#       will naturally just want to define the fields they wish to extract from the underlying (set of) files
#       and have PZ take care of the rest
class RealEstateListingFiles(pz.Schema):
    """The source text and image data for a real estate listing."""

    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.StringField(
        desc="The content of the listing's text description", required=True
    )
    image_contents = pz.ListField(
        element_type=pz.BytesField,
        desc="A list of the contents of each image of the listing",
        required=True,
    )


class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""

    address = pz.StringField(desc="The address of the property")
    price = pz.NumericField(desc="The listed price of the property")


class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""

    is_modern_and_attractive = pz.BooleanField(
        desc="True if the home interior design is modern and attractive and False otherwise"
    )
    has_natural_sunlight = pz.BooleanField(
        desc="True if the home interior has lots of natural sunlight and False otherwise"
    )


class RealEstateListingSource(pz.UserSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)

    def getSize(self):
        return sum(
            file.stat().st_size for file in Path(self.listings_dir).rglob("*")
        )

    def getItem(self, idx: int):
        # fetch listing
        listing = self.listings[idx]

        # create data record
        dr = pz.DataRecord(self.schema, scan_idx=idx)
        dr.listing = listing
        dr.image_contents = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            bytes_data = None
            with open(os.path.join(listing_dir, file), "rb") as f:
                bytes_data = f.read()
            if file.endswith(".txt"):
                dr.text_content = bytes_data.decode("utf-8")
            elif file.endswith(".png"):
                dr.image_contents.append(bytes_data)

        return dr


def get_workload_for_eval_dataset(dataset):
    """
    This assumes you have preregistered the enron and biofabric datasets:

    $ pz reg --path testdata/enron-eval --name enron
    $ pz reg --path testdata/biofabric-medium --name biofabric
    """
    if dataset == "enron":
        emails = pz.Dataset(dataset, schema=Email)
        emails = emails.filter(
            "The email is not quoting from a news article or an article written by someone outside of Enron"
        )
        emails = emails.filter(
            'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
        )
        return emails

    if dataset == "enron-deep":
        # emails = pz.Dataset(dataset, schema=EmailSender)
        # emails = emails.convert(EmailSubject, depends_on="text_content")
        # emails = emails.convert(EmailCC, depends_on="text_content")
        # emails = emails.convert(EmailBCC, depends_on="text_content")
        # emails = emails.convert(EmailMeetings, depends_on="text_content")
        # emails = emails.convert(EmailSummary, depends_on="text_content")
        # emails = emails.convert(EmailSentiment, depends_on="text_content")
        emails = pz.Dataset(dataset, schema=Email)
        emails = emails.filter(
            "The email is about business at Enron", depends_on="text_content"
        )
        emails = emails.filter(
            "The email has a negative sentiment", depends_on="text_content"
        )
        emails = emails.filter(
            "The email has no attachments", depends_on="text_content"
        )
        emails = emails.filter(
            "The email is not about scheduling a meeting",
            depends_on="text_content",
        )
        emails = emails.filter(
            "The email is replying to another email", depends_on="text_content"
        )
        emails = emails.filter(
            "The email is written in clear and concise language",
            depends_on="text_content",
        )
        return emails

    if dataset == "real-estate":

        def within_two_miles_of_mit(record):
            # NOTE: I'm using this hard-coded function so that folks w/out a
            #       Geocoding API key from google can still run this example
            try:
                if any(
                    [
                        street.lower() in record.address.lower()
                        for street in FAR_AWAY_ADDRS
                    ]
                ):
                    return False
                return True
            except:
                return False

        def in_price_range(record):
            try:
                price = record.price
                if type(price) == str:
                    price = price.strip()
                    price = int(price.replace("$", "").replace(",", ""))
                return 6e5 < price and price <= 2e6
            except:
                return False

        listings = pz.Dataset(dataset, schema=RealEstateListingFiles)
        listings = listings.convert(
            TextRealEstateListing, depends_on="text_content"
        )
        listings = listings.convert(
            ImageRealEstateListing,
            image_conversion=True,
            depends_on="image_contents",
        )
        listings = listings.filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        listings = listings.filter(
            within_two_miles_of_mit, depends_on="address"
        )
        listings = listings.filter(in_price_range, depends_on="price")
        return listings

    if dataset == "biofabric":
        xls = pz.Dataset(dataset, schema=pz.XLSFile)
        patient_tables = xls.convert(
            pz.Table,
            udf=udfs.xls_to_tables,
            cardinality=pz.Cardinality.ONE_TO_MANY,
        )
        patient_tables = patient_tables.filter(
            "The rows of the table contain the patient age"
        )
        case_data = patient_tables.convert(
            CaseData,
            desc="The patient data in the table",
            cardinality="oneToMany",
        )

        return case_data


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(
        description="Run the evaluation(s) for the paper"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help='The dataset: one of ["biofabric", "enron", "real-estate"]',
    )
    parser.add_argument(
        "--listings-dir",
        default="testdata/real-estate-eval-100",
        type=str,
        help="The directory with real-estate listings",
    )
    parser.add_argument(
        "--reoptimize",
        default=False,
        action="store_true",
        help="Run reoptimization",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Just print plans w/out actually running any",
    )
    parser.add_argument(
        "--execution",
        default=PipelinedSingleThreadSentinelExecution,
        action="store_true",
        help="The execution engine to use",
    )

    args = parser.parse_args()

    # register real-estate dataset if necessary
    if args.dataset == "real-estate":
        print("Registering Datasource")
        pz.DataDirectory().registerUserSource(
            RealEstateListingSource(args.dataset, args.listings_dir),
            args.dataset,
        )

    # # re-optimization is unique enough to warrant its own code path
    # if args.reoptimize:
    #     os.makedirs(f"final-eval-results/reoptimization/{args.dataset}", exist_ok=True)
    #     run_reoptimize_eval(args.dataset, args.policy, args.parallel)
    #     exit(1)

    # create directory for final results
    os.makedirs(f"final-eval-results/{args.dataset}", exist_ok=True)

    # The user has to indicate the evaluation to be run
    if args.dataset is None:
        print("Please provide a dataset (--dataset)")
        exit(1)

    # get PZ plan metrics
    print("Running PZ Plans")
    print("----------------")
    dataset_to_size = {
        "enron": 1000,
        "real-estate": 100,
        "biofabric": 11,
        "enron-deep": 100,
    }
    dataset_size = dataset_to_size[args.dataset]
    num_samples = int(0.05 * dataset_size) if args.dataset != "biofabric" else 1

    workload = get_workload_for_eval_dataset(args.dataset)

    available_models = getModels(include_vision=True)
    num_sentinels = len(available_models) - 1
    records, plan, stats = pz.Execute(
        workload,
        policy=pz.MinCost(),
        available_models=available_models,
        num_samples=num_samples,
        max_workers=num_sentinels,
        nocache=True,
        verbose=True,
        allow_bonded_query=True,
        allow_code_synth=True,
        allow_token_reduction=True,
        execution_engine=args.execution,
    )
