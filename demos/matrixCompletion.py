import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.elements import DataRecord
from palimpzest.utils import udfs, getModels, getVisionModels
from io import BytesIO
from openai import OpenAI
from pathlib import Path

from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd

import argparse
import datasets
import json
import os
import random

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

class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)


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

class RealEstateValidationSource(pz.ValidationDataSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir), key=lambda listing: int(listing.split('listing')[-1]))

        self.val_listings = self.listings
        self.listings = []
        
        self.label_fields_to_values = {}
        for listing_dir in os.listdir(listings_dir):
            for file in os.listdir(os.path.join(listings_dir, listing_dir)):
                filepath = os.path.join(listings_dir, listing_dir, file)
                if not filepath.endswith('.txt'):
                    continue
                with open(filepath,'r') as f:
                    listing_text = f.read()
                    listing_lines = listing_text.split('\n')
                    address = listing_lines[0].split("Address:")[-1].strip()
                    price = listing_lines[1].split("Price:")[-1].strip()
                    price = int(price.replace("$", "").replace(",", ""))
                    self.label_fields_to_values[listing_dir] = {"address": address, "price": price}

    def copy(self):
        return RealEstateValidationSource(self.dataset_id, self.listings_dir)

    def __len__(self):
        return 1 # len(self.listings)

    def getValLength(self):
        return len(self.val_listings)

    def getSize(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob('*'))

    def getFieldToMetricFn(self):
        # define quality eval function for price field
        def price_eval(price: str, expected_price: int):
            if type(price) == str:
                try:
                    price = price.strip()
                    price = int(price.replace("$", "").replace(",", ""))
                except:
                    return False
            return price == expected_price

        fields_to_metric_fn = {
            "address": "exact",
            "price": price_eval,
        }

        return fields_to_metric_fn

    def getItem(self, idx: int, val: bool=False, include_label: bool=False):
        # fetch listing
        listing = self.listings[idx] if not val else self.val_listings[idx]

        # create data record
        dr = pz.DataRecord(self.schema, source_id=listing)
        dr.listing = listing
        dr.image_filepaths = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    dr.text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                dr.image_filepaths.append(os.path.join(listing_dir, file))

        # if requested, also return the label information
        if include_label:
            # augment data record with label info
            labels_dict = self.label_fields_to_values[listing]

            for field, value in labels_dict.items():
                setattr(dr, field, value)

        return dr


class BiodexEntry(pz.Schema):
    """A single entry in the Biodex ICSR Dataset."""

    pmid = pz.StringField(desc="The PubMed ID of the medical paper", required=True)
    title = pz.StringField(desc="The title of the medical paper", required=True)
    abstract = pz.StringField(desc="The abstract of the medical paper", required=True)
    fulltext = pz.StringField(desc="The full text of the medical paper, which contains information relevant for creating a drug safety report.", required=True)

class BiodexDrugs(BiodexEntry):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. In this task,
    you will be asked to extract a list of every drug mentioned in the article.
    """
    drugs = pz.ListField(desc="The **list** of all active substance names of the drugs discussed in the report.\n - For example: [\"azathioprine\", \"infliximab\", \"mesalamine\", \"prednisolone\"]", element_type=pz.StringField, required=True)

class BiodexValidationSource(pz.ValidationDataSource):
    def __init__(self, datasetId, seed: int=42):
        super().__init__(BiodexEntry, datasetId)
        self.dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")
        self.seed = seed

        # shuffle and sample from full test dataset
        self.val_dataset = [self.dataset['test'][idx] for idx in range(len(self.dataset['test']))]
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(self.val_dataset)
        # TODO: also store mapping from optimizations (phys_op_id,op_details,op_str) to columns in metrics
        # NOTE: used first 100 for MC evaluation; 250 for plan execution
        self.val_dataset = self.val_dataset[:100] # 250

        self.test_dataset = []

        # construct mapping from listing --> label (field, value) pairs
        def compute_target_record(entry):
            target_lst = entry['target'].split('\n')
            label_dict = {"drugs": [drug.strip().lower() for drug in target_lst[2].split(':')[-1].split(",")]}
            return label_dict

        self.label_fields_to_values = {entry['pmid']: compute_target_record(entry) for entry in self.val_dataset}

    def copy(self):
        return BiodexValidationSource(self.dataset_id, self.seed)

    def __len__(self):
        return 1 # len(self.test_dataset)

    def getValLength(self):
        return len(self.val_dataset)

    def getSize(self):
        return 0

    def getFieldToMetricFn(self):
        # define quality eval function for drugs and reactions fields
        def f1_eval(preds: list, targets: list):
            # TODO? convert preds to a list of strings
            if preds is None:
                return 0.0

            # compute precision and recall
            s_preds = set(preds)
            s_targets = set(targets)

            intersect = s_preds.intersection(s_targets)

            precision = len(intersect) / len(s_preds) if len(s_preds) > 0 else 0.0
            recall = len(intersect) / len(s_targets)

            # compute f1 score and return
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return f1

        fields_to_metric_fn = {"drugs": f1_eval}

        return fields_to_metric_fn

    def getItem(self, idx: int, val: bool=False, include_label: bool=False):
        # fetch entry
        entry = self.test_dataset[idx] if not val else self.val_dataset[idx]

        # create data record
        dr = pz.DataRecord(self.schema, source_id=entry['pmid'])
        dr.pmid = entry['pmid']
        dr.title = entry['title']
        dr.abstract = entry['abstract']
        dr.fulltext = entry['fulltext']
  
        # if requested, also return the label information
        if include_label:
            # augment data record with label info
            labels_dict = self.label_fields_to_values[entry['pmid']]

            for field, value in labels_dict.items():
                setattr(dr, field, value)

        return dr


if __name__ == "__main__":
    # NOTE: set DSP_CACHEBOOL=False before executing this script (if running an experimental/benchmarking evaluation)
    if "DSP_CACHEBOOL" not in os.environ or os.environ["DSP_CACHEBOOL"].lower() != "false":
        raise Exception("TURN OFF DSPy CACHE BY SETTING `export DSP_CACHEBOOL=False")

    if "LOG_MATRICES" not in os.environ or os.environ["LOG_MATRICES"].lower() != "true":
        raise Exception("SET LOG_MATRICES=TRUE")

    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Print verbose output"
    )
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument("--workload", type=str, help="The workload to run. One of enron, real-estate, medical-schema-matching.")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of validation samples",
        default=5,
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="Rank for low-rank MC",
        default=4,
    )
    args = parser.parse_args()

    # The user has to indicate the dataset id and the workload
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.workload is None:
        print("Please provide a workload")
        exit(1)

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)

    datasetid = args.datasetid
    workload = args.workload
    verbose = args.verbose
    rank = args.rank
    num_samples = args.num_samples
    execution_engine = pz.SequentialParallelSentinelExecution
    # execution_engine = pz.SequentialSingleThreadSentinelExecution

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    plan = None
    if workload == "real-estate":
        # datasetid="real-estate-eval-100" for paper evaluation
        data_filepath = f"testdata/{datasetid}"
        user_dataset_id = f"{datasetid}-user"

        # create and register validation data source
        datasource = RealEstateValidationSource(
            datasetId=f"{user_dataset_id}",
            listings_dir=data_filepath,
        )
        pz.DataDirectory().registerUserSource(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )

        plan = pz.Dataset(user_dataset_id, schema=RealEstateListingFiles)
        plan = plan.convert(TextRealEstateListing, depends_on="text_content")

    elif workload == "biodex":
        user_dataset_id = f"biodex-user"

        # create and register validation data source
        datasource = BiodexValidationSource(
            datasetId=f"{user_dataset_id}",
            seed=42,
        )
        pz.DataDirectory().registerUserSource(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )
        plan = pz.Dataset(user_dataset_id, schema=BiodexEntry)
        plan = plan.convert(BiodexDrugs)

    # select optimization strategy and available models based on engine
    optimization_strategy = pz.OptimizationStrategy.PARETO
    available_models = getModels(include_vision=True)

    # execute pz plan
    _, _ = pz.Execute(
        plan,
        pz.MaxQuality(),
        nocache=True,
        available_models=available_models,
        optimization_strategy=optimization_strategy,
        execution_engine=execution_engine,
        rank=rank,
        verbose=verbose,
        allow_code_synth=(workload != "biodex"),
    )
