import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.elements import DataRecord
from palimpzest.utils import udfs, getModels, getVisionModels
from io import BytesIO
from pathlib import Path

import pandas as pd

import argparse
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
    def __init__(self, datasetId, listings_dir, split_idx: int=25, num_samples: int=5, shuffle: bool=False, seed: int=42):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.split_idx = split_idx
        self.listings = sorted(os.listdir(self.listings_dir), key=lambda listing: int(listing.split('listing')[-1]))

        self.val_listings = self.listings[:split_idx]
        self.listings = self.listings[split_idx:]

        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        if split_idx != 25:
            raise Exception("Currently must split on split_idx=25 for correctness")

        if num_samples > 25:
            raise Exception("We have not labelled more than the first 25 listings!")

        # construct mapping from listing --> label (field, value) pairs
        self.label_fields_to_values = {
            "listing1": {"address": "161 Auburn St Unit 161, Cambridge, MA 02139", "price": 1550000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": True},
            "listing2": {"address": "14 Concord Unit 712, Cambridge, MA, 02138", "price": 610000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing3": {"address": "10 Dana St Unit 7, Cambridge, MA, 02138", "price": 524900, "is_modern_and_attractive": False, "has_natural_sunlight": False, "_passed_filter": False},
            "listing4": {"address": "27 Winter St, Cambridge, MA, 02141", "price": 739000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing5": {"address": "59 Kelly Rd Unit 59, Cambridge, MA, 02139", "price": 1775000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": True},
            "listing6": {"address": "24 Greenough Ave, Cambridge, MA, 02139", "price": 4999999, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing7": {"address": "362-366 Commonwealth Ave Unit 4C, Boston, MA, 02115", "price": 609900, "is_modern_and_attractive": False, "has_natural_sunlight": False, "_passed_filter": False},
            "listing8": {"address": "188 Brookline Ave Unit 21H, Boston, MA, 02215", "price": 1485000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": True},
            "listing9": {"address": "11 Aberdeen St Unit 4, Boston, MA, 02215", "price": 699000, "is_modern_and_attractive": False, "has_natural_sunlight": False, "_passed_filter": False},
            "listing10": {"address": "188 Brookline Ave Unit 19A, Boston, MA, 02215", "price": 3200000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing11": {"address": "49 Melcher St Unit 205, Boston, MA, 02210", "price": 860000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing12": {"address": "15 Sleeper St Unit 406, Boston, MA, 02210", "price": 1450000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing13": {"address": "437 D St Unit 6C, Boston, MA, 02210", "price": 1025000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing14": {"address": "133 Seaport Blvd Unit 1715, Boston, MA, 02210", "price": 1299999, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing15": {"address": "50 Liberty Dr Unit 5E, Boston, MA, 02210", "price": 2995000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing16": {"address": "133 Seaport Blvd Unit 802, Boston, MA, 02210", "price": 1679000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing17": {"address": "14 Ware St Unit 44, Cambridge, MA, 02138", "price": 660000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing18": {"address": "20 Mcternan Unit 203, Cambridge, MA, 02139", "price": 825000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing19": {"address": "150 Hampshire St Unit 5, Cambridge, MA, 02139", "price": 895000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing20": {"address": "144 Spring St, Cambridge, MA, 02141", "price": 2350000, "is_modern_and_attractive": False, "has_natural_sunlight": False, "_passed_filter": False},
            "listing21": {"address": "41-41A Pleasant St, Cambridge, MA, 02139", "price": 4450000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": False},
            "listing22": {"address": "1 Pine St, Cambridge, MA, 02139", "price": 1875000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing23": {"address": "1055 Cambridge Unit 200, Cambridge, MA, 02139", "price": 1390000, "is_modern_and_attractive": True, "has_natural_sunlight": True, "_passed_filter": True},
            "listing24": {"address": "570 Franklin St Unit 1, Cambridge, MA, 02139", "price": 589000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
            "listing25": {"address": "12 Kinnaird, Cambridge, MA, 02139", "price": 1200000, "is_modern_and_attractive": False, "has_natural_sunlight": True, "_passed_filter": False},
        }

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.val_listings)

        # trim to number of samples
        self.val_listings = self.val_listings[:num_samples]

    def copy(self):
        return RealEstateValidationSource(self.dataset_id, self.listings_dir, self.split_idx, self.num_samples, self.shuffle, self.seed)

    def __len__(self):
        return len(self.listings)

    def getValLength(self):
        return len(self.val_listings)

    def getSize(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob('*'))

    def getFieldToMetricFn(self):
        # define quality eval function for price field
        def price_eval(price: str, expected_price: int):
            if type(price) == str:
                price = price.strip()
                price = int(price.replace("$", "").replace(",", ""))
            return price == expected_price

        fields_to_metric_fn = {
            "address": "exact",
            "price": price_eval,
            "is_modern_and_attractive": "exact",
            "has_natural_sunlight": "exact",
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

    # def getValItem(self, idx):
    #     # fetch data record for this idx
    #     dr = self.getItem(idx)

    #     # look up the labels for this listing
    #     listing = self.val_listings[idx]

    #     # augment data record with label info
    #     labels_dict = self.label_fields_to_values[listing]

    #     for field, value in labels_dict.items():
    #         setattr(dr, field, value)

    #     return [dr]


if __name__ == "__main__":
    # NOTE: set DSP_CACHEBOOL=False before executing this script (if running an experimental/benchmarking evaluation)
    if "DSP_CACHEBOOL" not in os.environ or os.environ["DSP_CACHEBOOL"].lower() != "false":
        raise Exception("TURN OFF DSPy CACHE BY SETTING `export DSP_CACHEBOOL=False")

    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Print verbose output"
    )
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument("--workload", type=str, help="The workload to run. One of enron, real-estate, medical-schema-matching.")
    parser.add_argument(
        "--engine",
        type=str,
        help='The engine to use. One of sentinel, nosentinel',
        default='nosentinel',
    )
    parser.add_argument(
        "--executor",
        type=str,
        help='The plan executor to use. One of sequential, pipelined, parallel',
        default='parallel',
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
        default='mincost',
    )
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
    parser.add_argument(
        "--model",
        type=str,
        help="One of 'gpt-4o', 'gpt-4o-mini', 'llama', 'mixtral'",
        default='gpt-4o',
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
    policy = pz.MaxQuality()
    if args.policy == "mincost":
        policy = pz.MinCost()
    elif args.policy == "mintime":
        policy = pz.MinTime()
    elif args.policy == "maxquality":
        policy = pz.MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    execution_engine = None
    engine, executor = args.engine, args.executor
    if engine == "sentinel":
        if executor == "sequential":
            execution_engine = pz.SequentialSingleThreadSentinelExecution
        elif executor == "parallel":
            execution_engine = pz.SequentialParallelSentinelExecution
        else:
            print("Unknown executor")
            exit(1)
    elif engine == "nosentinel":
        if executor == "sequential":
            execution_engine = pz.SequentialSingleThreadNoSentinelExecution
        elif executor == "pipelined":
            execution_engine = pz.PipelinedSingleThreadNoSentinelExecution
        elif executor == "parallel":
            execution_engine = pz.PipelinedParallelNoSentinelExecution
        else:
            print("Unknown executor")
            exit(1)
    else:
        print("Unknown engine")
        exit(1)
    
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    plan = None
    if workload == "enron":
        # datasetid="enron-eval" for paper evaluation
        plan = pz.Dataset(datasetid, schema=Email)
        plan = plan.filter(
            "The email is not quoting from a news article or an article written by someone outside of Enron"
        )
        plan = plan.filter(
            'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
        )

    elif workload == "real-estate":
        # datasetid="real-estate-eval-100" for paper evaluation
        data_filepath = f"testdata/{datasetid}"
        user_dataset_id = f"{datasetid}-user"

        # create and register validation data source
        datasource = RealEstateValidationSource(
            datasetId=f"{user_dataset_id}",
            listings_dir=data_filepath,
            num_samples=args.num_samples,
            shuffle=False,
            seed=42,
        )
        pz.DataDirectory().registerUserSource(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )

        plan = pz.Dataset(user_dataset_id, schema=RealEstateListingFiles)
        plan = plan.convert(TextRealEstateListing, depends_on="text_content")
        plan = plan.convert(
            ImageRealEstateListing, image_conversion=True, depends_on="image_filepaths"
        )
        plan = plan.filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    # select optimization strategy and available models based on engine
    optimization_strategy, available_models = None, None
    if engine == "sentinel":
        optimization_strategy = pz.OptimizationStrategy.OPTIMAL
        available_models = getModels(include_vision=True)
    else:
        model_str_to_model = {
            "gpt-4o": Model.GPT_4o,
            "gpt-4o-mini": Model.GPT_4o_MINI,
            "mixtral": Model.MIXTRAL,
            "llama": Model.LLAMA3,
        }
        model_str_to_vision_model = {
            "gpt-4o": Model.GPT_4o_V,
            "gpt-4o-mini": Model.GPT_4o_MINI_V,
            "mixtral": Model.LLAMA3_V,
            "llama": Model.LLAMA3_V,
        }
        optimization_strategy = pz.OptimizationStrategy.NONE
        available_models = [model_str_to_model[args.model]] + [model_str_to_vision_model[args.model]]

    # execute pz plan
    records, execution_stats = pz.Execute(
        plan,
        policy,
        nocache=True,
        available_models=available_models,
        optimization_strategy=optimization_strategy,
        execution_engine=execution_engine,
        rank=args.rank,
        verbose=verbose,
    )

    # create filepaths for records and stats
    records_path = (
        f"opt-profiling-data/{workload}-records.json"
        if engine == "sentinel"
        else f"opt-profiling-data/{workload}-baseline-{args.model}-records.json"
    )
    stats_path = (
        f"opt-profiling-data/{workload}-profiling.json"
        if engine == "sentinel"
        else f"opt-profiling-data/{workload}-baseline-{args.model}-profiling.json"
    )

    # save record outputs
    record_jsons = []
    for record in records:
        record_dict = record._asDict()
        record_jsons.append(record_dict)

    with open(records_path, 'w') as f:
        json.dump(record_jsons, f)

    # save statistics
    execution_stats_dict = execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)