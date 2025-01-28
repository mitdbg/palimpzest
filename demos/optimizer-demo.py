import argparse
import json
import os
import random
import time
from pathlib import Path

import datasets
from ragatouille import RAGPretrainedModel

from palimpzest.constants import Model
from palimpzest.core.data.datasources import ValidationDataSource
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import BooleanField, ImageFilepathField, ListField, NumericField, StringField
from palimpzest.core.lib.schemas import Schema, TextFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost, MinTime
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils.model_helpers import get_models

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


def within_two_miles_of_mit(record: dict):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    try:
        return not any([street.lower() in record["address"].lower() for street in FAR_AWAY_ADDRS])
    except Exception:
        return False


def in_price_range(record: dict):
    try:
        price = record["price"]
        if isinstance(price, str):
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price <= 2e6
    except Exception:
        return False


class Email(TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = StringField(desc="The email address of the sender")
    subject = StringField(desc="The subject of the email")


class EnronValidationSource(ValidationDataSource):
    def __init__(
        self,
        file_dir,
        dataset_id,
        num_samples: int = 3,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(TextFile, dataset_id)
        self.file_dir = file_dir
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        # get list of filepaths
        self.test_filepaths = [
            os.path.join(file_dir, filename)
            for filename in sorted(os.listdir(file_dir))
            if os.path.isfile(os.path.join(file_dir, filename))
        ]

        # NOTE: this assumes the script is run from the root of the repository
        self.val_filepaths = [
            os.path.join("testdata/enron-eval-tiny", filename)
            for filename in sorted(os.listdir("testdata/enron-eval-tiny"))
            if os.path.isfile(os.path.join("testdata/enron-eval-tiny", filename))
        ]

        # use six labelled examples
        self.label_fields_to_values = {
            "buy-r-inbox-628.txt": {
                "sender": "sherron.watkins@enron.com",
                "subject": "RE: portrac",
                "passed_operator": True,
            },
            "buy-r-inbox-749.txt": {
                "sender": "david.port@enron.com",
                "subject": "RE: NewPower",
                "passed_operator": True,
            },
            "kaminski-v-deleted-items-1902.txt": {
                "sender": "vkaminski@aol.com",
                "subject": "Fwd: FYI",
                "passed_operator": False,
            },
            "martin-t-inbox-96-short.txt": {
                "sender": "sarah.palmer@enron.com",
                "subject": "Enron Mentions -- 01/18/02",
                "passed_operator": False,
            },
            "skilling-j-inbox-1109.txt": {
                "sender": "gary@cioclub.com",
                "subject": "Information Security Executive",
                "passed_operator": False,
            },
            "zipper-a-espeed-28.txt": {
                "sender": "travis.mccullough@enron.com",
                "subject": "Redraft of the Exclusivity Agreement",
                "passed_operator": True,
            },
        }

        if shuffle:
            random.Random(seed).shuffle(self.val_filepaths)

        # num samples cannot exceed the number of records
        assert self.num_samples <= len(self.label_fields_to_values), (
            "cannot have more samples than labelled data records"
        )

        # trim to number of samples
        self.val_filepaths = self.val_filepaths[:num_samples]

    def copy(self):
        return EnronValidationSource(self.file_dir, self.dataset_id, self.num_samples, self.shuffle, self.seed)

    def __len__(self):
        return len(self.test_filepaths)

    def get_val_length(self):
        return len(self.val_filepaths)

    def get_size(self):
        return 0

    def get_field_to_metric_fn(self):
        return {"sender": "exact", "subject": "exact"}

    def get_item(self, idx: int, val: bool = False, include_label: bool = False):
        # get filepath and filename
        filepath = self.val_filepaths[idx] if val else self.test_filepaths[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=filepath)
        dr.filename = os.path.basename(filepath)
        with open(filepath) as f:
            dr.contents = f.read()

        # if requested, also return the label information
        if include_label:
            # augment data record with label info
            labels_dict = self.label_fields_to_values[dr["filename"]]

            for field, value in labels_dict.items():
                setattr(dr, field, value)

        return dr


class RealEstateListingFiles(Schema):
    """The source text and image data for a real estate listing."""

    listing = StringField(desc="The name of the listing")
    text_content = StringField(desc="The content of the listing's text description")
    image_filepaths = ListField(
        element_type=ImageFilepathField,
        desc="A list of the filepaths for each image of the listing",
    )


class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""

    address = StringField(desc="The address of the property")
    price = NumericField(desc="The listed price of the property")


class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""

    is_modern_and_attractive = BooleanField(
        desc="True if the home interior design is modern and attractive and False otherwise"
    )
    has_natural_sunlight = BooleanField(
        desc="True if the home interior has lots of natural sunlight and False otherwise"
    )


class RealEstateValidationSource(ValidationDataSource):
    def __init__(
        self, dataset_id, listings_dir, split_idx: int = 25, num_samples: int = 5, shuffle: bool = False, seed: int = 42
    ):
        super().__init__(RealEstateListingFiles, dataset_id)
        self.listings_dir = listings_dir
        self.split_idx = split_idx
        self.listings = sorted(os.listdir(self.listings_dir), key=lambda listing: int(listing.split("listing")[-1]))
        assert len(self.listings) > split_idx, "split_idx is greater than the number of listings"

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
            "listing1": {
                "address": "161 Auburn St Unit 161, Cambridge, MA 02139",
                "price": 1550000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": True,
            },
            "listing2": {
                "address": "14 Concord Unit 712, Cambridge, MA, 02138",
                "price": 610000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing3": {
                "address": "10 Dana St Unit 7, Cambridge, MA, 02138",
                "price": 524900,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": False,
                "passed_operator": False,
            },
            "listing4": {
                "address": "27 Winter St, Cambridge, MA, 02141",
                "price": 739000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing5": {
                "address": "59 Kelly Rd Unit 59, Cambridge, MA, 02139",
                "price": 1775000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": True,
            },
            "listing6": {
                "address": "24 Greenough Ave, Cambridge, MA, 02139",
                "price": 4999999,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing7": {
                "address": "362-366 Commonwealth Ave Unit 4C, Boston, MA, 02115",
                "price": 609900,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": False,
                "passed_operator": False,
            },
            "listing8": {
                "address": "188 Brookline Ave Unit 21H, Boston, MA, 02215",
                "price": 1485000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": True,
            },
            "listing9": {
                "address": "11 Aberdeen St Unit 4, Boston, MA, 02215",
                "price": 699000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": False,
                "passed_operator": False,
            },
            "listing10": {
                "address": "188 Brookline Ave Unit 19A, Boston, MA, 02215",
                "price": 3200000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing11": {
                "address": "49 Melcher St Unit 205, Boston, MA, 02210",
                "price": 860000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing12": {
                "address": "15 Sleeper St Unit 406, Boston, MA, 02210",
                "price": 1450000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing13": {
                "address": "437 D St Unit 6C, Boston, MA, 02210",
                "price": 1025000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing14": {
                "address": "133 Seaport Blvd Unit 1715, Boston, MA, 02210",
                "price": 1299999,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing15": {
                "address": "50 Liberty Dr Unit 5E, Boston, MA, 02210",
                "price": 2995000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing16": {
                "address": "133 Seaport Blvd Unit 802, Boston, MA, 02210",
                "price": 1679000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing17": {
                "address": "14 Ware St Unit 44, Cambridge, MA, 02138",
                "price": 660000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing18": {
                "address": "20 Mcternan Unit 203, Cambridge, MA, 02139",
                "price": 825000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing19": {
                "address": "150 Hampshire St Unit 5, Cambridge, MA, 02139",
                "price": 895000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing20": {
                "address": "144 Spring St, Cambridge, MA, 02141",
                "price": 2350000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": False,
                "passed_operator": False,
            },
            "listing21": {
                "address": "41-41A Pleasant St, Cambridge, MA, 02139",
                "price": 4450000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing22": {
                "address": "1 Pine St, Cambridge, MA, 02139",
                "price": 1875000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing23": {
                "address": "1055 Cambridge Unit 200, Cambridge, MA, 02139",
                "price": 1390000,
                "is_modern_and_attractive": True,
                "has_natural_sunlight": True,
                "passed_operator": True,
            },
            "listing24": {
                "address": "570 Franklin St Unit 1, Cambridge, MA, 02139",
                "price": 589000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
            "listing25": {
                "address": "12 Kinnaird, Cambridge, MA, 02139",
                "price": 1200000,
                "is_modern_and_attractive": False,
                "has_natural_sunlight": True,
                "passed_operator": False,
            },
        }

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.val_listings)

        # trim to number of samples
        self.val_listings = self.val_listings[:num_samples]

    def copy(self):
        return RealEstateValidationSource(
            self.dataset_id, self.listings_dir, self.split_idx, self.num_samples, self.shuffle, self.seed
        )

    def __len__(self):
        return len(self.listings)

    def get_val_length(self):
        return len(self.val_listings)

    def get_size(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob("*"))

    def get_field_to_metric_fn(self):
        # define quality eval function for price field
        def price_eval(price: str, expected_price: int):
            if isinstance(price, str):
                try:
                    price = price.strip()
                    price = int(price.replace("$", "").replace(",", ""))
                except Exception:
                    return False
            return price == expected_price

        fields_to_metric_fn = {
            "address": "exact",
            "price": price_eval,
            "is_modern_and_attractive": "exact",
            "has_natural_sunlight": "exact",
        }

        return fields_to_metric_fn

    def get_item(self, idx: int, val: bool = False, include_label: bool = False):
        # fetch listing
        listing = self.listings[idx] if not val else self.val_listings[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=listing)
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


class BiodexEntry(Schema):
    """A single entry in the Biodex ICSR Dataset."""

    pmid = StringField(desc="The PubMed ID of the medical paper")
    title = StringField(desc="The title of the medical paper")
    abstract = StringField(desc="The abstract of the medical paper")
    fulltext = StringField(
        desc="The full text of the medical paper, which contains information relevant for creating a drug safety report.",
    )


class BiodexSerious(BiodexEntry):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. In this task,
    you will be asked to extract a rating of how serious the event was, with a definition of `serious`
    provided below.
    """

    serious = NumericField(
        desc="The seriousness of the adverse event.\n - Equal to 1 if the adverse event resulted in death, a life threatening condition, hospitalization, disability, congenital anomaly, or any other serious condition.\n - If none of the above occurred, equal to 2.",
    )


class BiodexPatientSex(BiodexEntry):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. In this task,
    you will be asked to extract the sex of the patient (if provided), with a definition of the
    expected output `patientsex` provided below.
    """

    patientsex = NumericField(
        desc="The reported biological sex of the patient.\n - Equal to 0 for unknown, 1 for male, 2 for female.",
    )


class BiodexDrugs(BiodexEntry):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. In this task,
    you will be asked to extract a list of every drug mentioned in the article.
    """

    drugs = ListField(
        desc='The **list** of all active substance names of the drugs discussed in the report.\n - For example: ["azathioprine", "infliximab", "mesalamine", "prednisolone"]',
        element_type=StringField,
    )


class BiodexReactions(BiodexEntry):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. In this task,
    you will be asked to extract a list of the primary adverse reactions which are experienced by the patient.
    """

    reactions = ListField(
        desc='The **list** of all reaction terms discussed in the report.\n - For example: ["Epstein-Barr virus", "infection reactivation", "Idiopathic interstitial pneumonia"]',
        element_type=StringField,
    )


class BiodexRankedReactions(BiodexReactions):
    """
    You will be presented with the text of a medical article which is partially or entirely about
    an adverse event experienced by a patient in response to taking one or more drugs. You will also
    be presented with a list of inferred reactions, and a set of retrieved labels which were matched
    to these inferred reactions. In this task, you are asked to output a ranked list of the labels
    which are most applicable based on the context of the article. Your output list must:
    - contain only elements from `reaction_labels`
    - place the most likely label first and the least likely label last
    - you may omit labels if you think they do not describe a reaction experienced by the patient
    """

    ranked_reaction_labels = ListField(
        desc="The ranked list of labels for adverse reactions experienced by the patient. The most likely label occurs first in the list.",
        element_type=StringField,
    )


class BiodexOutput(BiodexEntry):
    """The target output fields for an entry in the Biodex ICSR Dataset."""

    serious = NumericField(
        desc="The seriousness of the adverse event.\n - Equal to 1 if the adverse event resulted in death, a life threatening condition, hospitalization, disability, congenital anomaly, or any other serious condition.\n - If none of the above occurred, equal to 2.",
    )
    patientsex = NumericField(
        desc="The reported biological sex of the patient.\n - Equal to 0 for unknown, 1 for male, 2 for female.",
    )
    drugs = ListField(
        desc='The **list** of all active substance names of the drugs discussed in the report.\n - For example: ["azathioprine", "infliximab", "mesalamine", "prednisolone"]',
        element_type=StringField,
    )
    reactions = ListField(
        desc='The **list** of all reaction terms discussed in the report.\n - For example: ["Epstein-Barr virus", "infection reactivation", "Idiopathic interstitial pneumonia"]',
        element_type=StringField,
    )


class BiodexValidationSource(ValidationDataSource):
    def __init__(
        self,
        dataset_id,
        reactions_only: bool = True,
        rp_at_k: int = 5,
        num_samples: int = 5,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(BiodexEntry, dataset_id)
        self.dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")
        self.train_dataset = [self.dataset["train"][idx] for idx in range(250)]

        # sample from full test dataset
        self.test_dataset = [self.dataset["test"][idx] for idx in range(len(self.dataset["test"]))]
        self.test_dataset = self.test_dataset[:250]  # use first 250 to compare directly with biodex

        self.reactions_only = reactions_only
        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        # construct mapping from listing --> label (field, value) pairs
        def compute_target_record(entry, reactions_only: bool = False):
            target_lst = entry["target"].split("\n")
            label_dict = {
                "serious": int(target_lst[0].split(":")[-1]),
                "patientsex": int(target_lst[1].split(":")[-1]),
                "drugs": [drug.strip().lower() for drug in target_lst[2].split(":")[-1].split(",")],
                "reactions": [reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")],
                "reaction_labels": [reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")],
                "ranked_reaction_labels": [
                    reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")
                ],
            }
            if reactions_only:
                label_dict = {
                    k: v
                    for k, v in label_dict.items()
                    if k in ["reactions", "reaction_labels", "ranked_reaction_labels"]
                }
            return label_dict

        self.label_fields_to_values = {
            entry["pmid"]: compute_target_record(entry, reactions_only=reactions_only) for entry in self.train_dataset
        }

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.train_dataset)

        # trim to number of samples
        self.train_dataset = self.train_dataset[:num_samples]

    def copy(self):
        return BiodexValidationSource(self.dataset_id, self.num_samples, self.shuffle, self.seed)

    def __len__(self):
        return len(self.test_dataset)

    def get_val_length(self):
        return len(self.train_dataset)

    def get_size(self):
        return 0

    def get_field_to_metric_fn(self):
        # define f1 function
        def f1_eval(preds: list, targets: list):
            if preds is None:
                return 0.0

            try:
                # compute precision and recall
                s_preds = set([pred.lower() for pred in preds])
                s_targets = set([target.lower() for target in targets])

                intersect = s_preds.intersection(s_targets)

                precision = len(intersect) / len(s_preds) if len(s_preds) > 0 else 0.0
                recall = len(intersect) / len(s_targets)

                # compute f1 score and return
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                return f1

            except Exception:
                os.makedirs("f1-errors", exist_ok=True)
                ts = time.time()
                with open(f"f1-errors/error-{ts}.txt", "w") as f:
                    f.write(str(preds))
                return 0.0

        # define rank precision at k
        def rank_precision_at_k(preds: list, targets: list):
            if preds is None:
                return 0.0

            try:
                # lower-case each list
                preds = [pred.lower() for pred in preds]
                targets = set([target.lower() for target in targets])

                # compute rank-precision at k
                rn = len(targets)
                denom = min(self.rp_at_k, rn)
                total = 0.0
                for i in range(self.rp_at_k):
                    total += preds[i] in targets if i < len(preds) else 0.0

                return total / denom

            except Exception:
                os.makedirs("rp@k-errors", exist_ok=True)
                ts = time.time()
                with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                    f.write(str(preds))
                return 0.0

        # define quality eval function for drugs and reactions fields
        fields_to_metric_fn = {}
        if self.reactions_only:
            fields_to_metric_fn = {
                "reactions": f1_eval,
                "reaction_labels": f1_eval,
                "ranked_reaction_labels": rank_precision_at_k,
            }

        else:
            fields_to_metric_fn = {
                "serious": "exact",
                "patientsex": "exact",
                "drugs": f1_eval,
                "reactions": f1_eval,
            }

        return fields_to_metric_fn

    def get_item(self, idx: int, val: bool = False, include_label: bool = False):
        # fetch entry
        entry = self.test_dataset[idx] if not val else self.train_dataset[idx]

        # create data record
        dr = DataRecord(self.schema, source_id=entry["pmid"])
        dr.pmid = entry["pmid"]
        dr.title = entry["title"]
        dr.abstract = entry["abstract"]
        dr.fulltext = entry["fulltext"]

        # if requested, also return the label information
        if include_label:
            # augment data record with label info
            labels_dict = self.label_fields_to_values[entry["pmid"]]

            for field, value in labels_dict.items():
                setattr(dr, field, value)

        return dr


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print verbose output")
    parser.add_argument("--datasetid", type=str, help="The dataset id")
    parser.add_argument(
        "--workload", type=str, help="The workload to run. One of enron, real-estate, biodex, biodex-reactions."
    )
    parser.add_argument(
        "--processing_strategy",
        default="mab_sentinel",
        type=str,
        help="The engine to use. One of mab_sentinel, no_sentinel, random_sampling",
    )
    parser.add_argument(
        "--execution_strategy",
        default="pipelined_parallel",
        type=str,
        help="The plan executor to use. One of sequential, pipelined_single_thread, pipelined_parallel",
    )
    parser.add_argument(
        "--policy",
        default="mincost",
        type=str,
        help="One of 'mincost', 'mintime', 'maxquality'",
    )
    parser.add_argument(
        "--val-examples",
        default=5,
        type=int,
        help="Number of validation examples to sample from",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        help="One of 'gpt-4o', 'gpt-4o-mini', 'llama', 'mixtral'",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize RNG for MAB sampling algorithm",
    )
    parser.add_argument(
        "--k",
        default=10,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--j",
        default=3,
        type=int,
        help="Number of columns to sample in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument(
        "--sample-budget",
        default=100,
        type=int,
        help="Total sample budget in Random Sampling or MAB sentinel execution",
    )
    parser.add_argument("--sample-all-ops", default=False, action="store_true", help="Sample all operators")
    parser.add_argument("--sample-all-records", default=False, action="store_true", help="Sample all records")
    parser.add_argument(
        "--sample-start-idx",
        default=None,
        type=int,
        help="",
    )
    parser.add_argument(
        "--sample-end-idx",
        default=None,
        type=int,
        help="",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="Name of experiment which is used in output filename",
    )

    args = parser.parse_args()

    # The user has to indicate the dataset id and the workload
    if args.datasetid is None:
        print("Please provide a dataset id using --datasetid")
        exit(1)
    if args.workload is None:
        print("Please provide a workload using --workload")
        exit(1)
    if args.exp_name is None:
        print("Please provide an experiment name using --exp-name")
        exit(1)

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)

    datasetid = args.datasetid
    workload = args.workload
    verbose = args.verbose
    seed = args.seed
    val_examples = args.val_examples
    k = args.k
    j = args.j
    sample_budget = args.sample_budget
    sample_all_ops = args.sample_all_ops
    sample_all_records = args.sample_all_records
    sample_start_idx = args.sample_start_idx
    sample_end_idx = args.sample_end_idx
    exp_name = args.exp_name

    policy = MaxQuality()
    if args.policy == "mincost":
        policy = MinCost()
    elif args.policy == "mintime":
        policy = MinTime()
    elif args.policy == "maxquality":
        policy = MaxQuality()
    else:
        print("Policy not supported for this demo")
        exit(1)

    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    # create pz plan
    plan, use_final_op_quality = None, False
    if workload == "enron":
        # datasetid="real-estate-eval-100" for paper evaluation
        data_filepath = f"testdata/{datasetid}"
        user_dataset_id = f"{datasetid}-user"

        # create and register validation data source
        datasource = EnronValidationSource(file_dir=data_filepath, dataset_id=user_dataset_id)
        DataDirectory().register_user_source(src=datasource, dataset_id=user_dataset_id)

        plan = Dataset(user_dataset_id, schema=Email)
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
            dataset_id=f"{user_dataset_id}",
            listings_dir=data_filepath,
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )

        plan = Dataset(user_dataset_id, schema=RealEstateListingFiles)
        plan = plan.convert(TextRealEstateListing, depends_on="text_content")
        plan = plan.convert(ImageRealEstateListing, depends_on="image_filepaths")
        plan = plan.filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    elif workload == "biodex-reactions":
        user_dataset_id = "biodex-user"

        # load index
        index_path = ".ragatouille/colbert/indexes/reaction-terms"
        index = RAGPretrainedModel.from_index(index_path)

        # create and register validation data source
        datasource = BiodexValidationSource(
            dataset_id=f"{user_dataset_id}",
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )
        plan = Dataset(user_dataset_id, schema=BiodexEntry)
        plan = plan.convert(BiodexReactions)  # infer

        def search_func(index, query, k):
            results = index.search(query, k=1)
            results = [result[0] if isinstance(result, list) else result for result in results]
            sorted_results = sorted(results, key=lambda result: result["score"], reverse=True)
            return [result["content"] for result in sorted_results[:k]]

        plan = plan.retrieve(
            index=index,
            search_func=search_func,
            search_attr="reactions",
            output_attr="reaction_labels",
            output_attr_desc="Most relevant official terms for adverse reactions for the provided `reactions`",
            # k=10, # if we set k, then it will be fixed; if we leave it unspecified then the optimizer will choose
        )  # TODO: retrieve (top-1 retrieve per prediction? or top-k retrieve for all predictions?)
        plan = plan.convert(BiodexRankedReactions)

        # only use final op quality
        use_final_op_quality = True

    elif workload == "biodex":
        user_dataset_id = "biodex-user"

        # load index
        index_path = ".ragatouille/colbert/indexes/reaction-terms"
        index = RAGPretrainedModel.from_index(index_path)

        # create and register validation data source
        datasource = BiodexValidationSource(
            dataset_id=f"{user_dataset_id}",
            reactions_only=False,
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(
            src=datasource,
            dataset_id=f"{user_dataset_id}",
        )
        plan = Dataset(user_dataset_id, schema=BiodexEntry)
        plan = plan.convert(BiodexSerious, depends_on=["title", "abstract", "fulltext"])
        plan = plan.convert(BiodexPatientSex, depends_on=["title", "abstract", "fulltext"])
        plan = plan.convert(BiodexDrugs, depends_on=["title", "abstract", "fulltext"])
        plan = plan.convert(BiodexReactions, depends_on=["title", "abstract", "fulltext"])

        def search_func(index, query, k):
            results = index.search(query, k=1)
            results = [result[0] if isinstance(result, list) else result for result in results]
            sorted_results = sorted(results, key=lambda result: result["score"], reverse=True)
            return [result["content"] for result in sorted_results[:k]]

        plan = plan.retrieve(
            index=index,
            search_func=search_func,
            search_attr="reactions",
            output_attr="reaction_labels",
            output_attr_desc="Most relevant official terms for adverse reactions for the provided `reactions`",
            # k=10, # if we set k, then it will be fixed; if we leave it unspecified then the optimizer will choose
        )  # TODO: retrieve (top-1 retrieve per prediction? or top-k retrieve for all predictions?)
        plan = plan.convert(BiodexRankedReactions)

        # only use final op quality
        use_final_op_quality = True

    # select optimization strategy and available models based on engine
    optimizer_strategy, available_models = None, None
    if args.processing_strategy in ["mab_sentinel", "random_sampling"]:
        optimizer_strategy = "pareto"
        available_models = get_models(include_vision=True)
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
        optimizer_strategy = "none"
        available_models = [model_str_to_model[args.model]] + [model_str_to_vision_model[args.model]]

    # execute pz plan
    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        available_models=available_models,
        processing_strategy=args.processing_strategy,
        optimizer_strategy=optimizer_strategy,
        execution_strategy=args.execution_strategy,
        allow_code_synth=False,  # (workload != "biodex"),
        use_final_op_quality=use_final_op_quality,
        max_workers=10,
        verbose=verbose,
    )

    data_record_collection = plan.run(
        config=config,
        k=k,
        j=j,
        sample_budget=sample_budget,
        sample_all_ops=sample_all_ops,
        sample_all_records=sample_all_records,
        sample_start_idx=sample_start_idx,
        sample_end_idx=sample_end_idx,
        seed=seed,
        exp_name=exp_name,
    )

    print(data_record_collection.to_df())

    # create filepaths for records and stats
    records_path = (
        f"opt-profiling-data/{workload}-{exp_name}-records.json"
        if args.processing_strategy in ["mab_sentinel", "random_sampling"]
        else f"opt-profiling-data/{workload}-baseline-{exp_name}-records.json"
    )
    stats_path = (
        f"opt-profiling-data/{workload}-{exp_name}-profiling.json"
        if args.processing_strategy in ["mab_sentinel", "random_sampling"]
        else f"opt-profiling-data/{workload}-baseline-{exp_name}-profiling.json"
    )

    # save record outputs
    record_jsons = []
    for record in data_record_collection:
        record_dict = record.to_dict()
        if workload == "biodex":
            record_dict = {
                k: v for k, v in record_dict.items() if k in ["pmid", "serious", "patientsex", "drugs", "reactions"]
            }
        elif workload == "biodex-reactions":
            record_dict = {
                k: v
                for k, v in record_dict.items()
                if k in ["pmid", "reactions", "reaction_labels", "ranked_reaction_labels"]
            }
        record_jsons.append(record_dict)

    with open(records_path, "w") as f:
        json.dump(record_jsons, f)

    # save statistics
    execution_stats_dict = data_record_collection.execution_stats.to_json()
    with open(stats_path, "w") as f:
        json.dump(execution_stats_dict, f)
