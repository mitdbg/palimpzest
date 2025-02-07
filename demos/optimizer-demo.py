import argparse
import json
import os
import random
import time
from functools import partial

import datasets
from ragatouille import RAGPretrainedModel

from palimpzest.constants import Model
from palimpzest.core.data.datasources import DataSource
from palimpzest.core.lib.fields import ImageFilepathField, ListField
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

file_cols = [
    {"name": "filename", "type": str, "desc": "The UNIX-style name of the file"},
    {"name": "contents", "type": bytes, "desc": "The contents of the file"},
]

email_cols = [
    {"name": "sender", "type": str, "desc": "The email address of the sender"},
    {"name": "subject", "type": str, "desc": "The subject of the email"},
]

class EnronSource(DataSource):
    def __init__(
        self,
        file_dir,
        dataset_id,
        num_samples: int = 3,
        shuffle: bool = False,
        seed: int = 42,
    ):
        assert "enron-eval" in file_dir, "The dataset must be one of the 'enron-eval*' directories"
        super().__init__(file_cols, dataset_id)
        self.file_dir = file_dir
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        # get list of filepaths
        self.filepaths = [
            os.path.join(file_dir, filename)
            for filename in sorted(os.listdir(file_dir))
            if os.path.isfile(os.path.join(file_dir, filename))
        ]

        # use six labelled examples
        self.filename_to_labels = {
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
            random.Random(seed).shuffle(self.filepaths)

        # num samples cannot exceed the number of records
        assert self.num_samples <= len(self.filename_to_labels), (
            "cannot have more samples than labelled data records"
        )

        # trim to number of samples
        self.filepaths = self.filepaths[:num_samples]

    def __len__(self):
        return len(self.filepaths)

    def get_item(self, idx: int):
        # get filepath
        filepath = self.filepaths[idx]

        # get input fields
        filename = os.path.basename(filepath)
        with open(filepath) as f:
            contents = f.read()

        # create item with fields
        item = {"fields": {}, "labels": {}}
        item["fields"]["filename"] = filename
        item["fields"]["contents"] = contents

        # add label info
        item["labels"] = self.filename_to_labels[filename]

        return item


real_estate_listing_cols = [
    {"name": "listing", "type": str, "desc": "The name of the listing"},
    {"name": "text_content", "type": str, "desc": "The content of the listing's text description"},
    {"name": "image_filepaths", "type": ListField(ImageFilepathField), "desc": "A list of the filepaths for each image of the listing"},
]

real_estate_text_cols = [
    {"name": "address", "type": str, "desc": "The address of the property"},
    {"name": "price", "type": int | float, "desc": "The listed price of the property"},
]

real_estate_image_cols = [
    {"name": "is_modern_and_attractive", "type": bool, "desc": "True if the home interior design is modern and attractive and False otherwise"},
    {"name": "has_natural_sunlight", "type": bool, "desc": "True if the home interior has lots of natural sunlight and False otherwise"},
]

# class RealEstateListingFiles(Schema):
#     """The source text and image data for a real estate listing."""

#     listing = StringField(desc="The name of the listing")
#     text_content = StringField(desc="The content of the listing's text description")
#     image_filepaths = ListField(
#         element_type=ImageFilepathField,
#         desc="A list of the filepaths for each image of the listing",
#     )

class RealEstateSource(DataSource):
    def __init__(
        self,
        dataset_id,
        listings_dir,
        num_samples: int = 5,
        shuffle: bool = False,
        seed: int = 42,
    ):
        # NOTE: this source will throw an exception for real-estate-eval directories w/more than 25 examples
        #       due to the fact that shuffling (and lexographical ordering) may cause one of the listings
        #       to be different from the 25 which we've labelled
        assert "real-estate-eval" in listings_dir, "The dataset must be one of the 'real-estate-eval*' directories"
        super().__init__(real_estate_listing_cols, dataset_id)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir), key=lambda listing: int(listing.split("listing")[-1]))
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        if num_samples > 25:
            raise Exception("We have not labelled more than the first 25 listings!")

        # construct mapping from listing --> label (field, value) pairs
        self.listing_to_labels = {
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
            random.Random(seed).shuffle(self.listings)

        # trim to number of samples
        self.listings = self.listings[:num_samples]

    @staticmethod
    def price_eval(price: str | int, expected_price: int):
        if isinstance(price, str):
            try:
                price = price.strip()
                price = int(price.replace("$", "").replace(",", ""))
            except Exception:
                return 0.0
        return float(price == expected_price)

    def __len__(self):
        return len(self.listings)

    def get_item(self, idx: int):
        # get listing
        listing = self.listings[idx]

        # get input fields
        image_filepaths, text_content = [], None
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                image_filepaths.append(os.path.join(listing_dir, file))

        # create item with fields
        item = {"fields": {}, "labels": {}, "score_fn": {}}
        item["fields"]["listing"] = listing
        item["fields"]["text_content"] = text_content
        item["fields"]["image_filepaths"] = image_filepaths

        # add label info
        item["labels"] = self.listing_to_labels[listing]

        # add scoring function for price
        item["score_fn"]["price"] = RealEstateSource.price_eval

        return item

biodex_entry_cols = [
    {"name": "pmid", "type": str, "desc": "The PubMed ID of the medical paper"},
    {"name": "title", "type": str, "desc": "The title of the medical paper"},
    {"name": "abstract", "type": str, "desc": "The abstract of the medical paper"},
    {"name": "fulltext", "type": str, "desc": "The full text of the medical paper, which contains information relevant for creating a drug safety report."},
]

biodex_serious_cols = [
    {"name": "serious", "type": int, "desc": "The seriousness of the adverse event.\n - Equal to 1 if the adverse event resulted in death, a life threatening condition, hospitalization, disability, congenital anomaly, or any other serious condition.\n - If none of the above occurred, equal to 2."},
]

biodex_patient_sex_cols = [
    {"name": "patientsex", "type": int, "desc": "The reported biological sex of the patient.\n - Equal to 0 for unknown, 1 for male, 2 for female."},
]

biodex_drugs_cols = [
    {"name": "drugs", "type": list[str], "desc": "The list of all active substance names of the drugs discussed in the report."},
]

biodex_reactions_cols = [
    {"name": "reactions", "type": list[str], "desc": "The list of all reaction terms discussed in the report."},
]

biodex_reaction_labels_cols = [
    {"name": "reaction_labels", "type": list[str], "desc": "Most relevant official terms for adverse reactions for the provided `reactions`"},
]

biodex_ranked_reactions_labels_cols = [
    {"name": "ranked_reaction_labels", "type": list[str], "desc": "The ranked list of labels for adverse reactions experienced by the patient. The most likely label occurs first in the list."},
]


class BiodexSource(DataSource):
    def __init__(
        self,
        dataset_id,
        reactions_only: bool = True,
        rp_at_k: int = 5,
        num_samples: int = 5,
        split: str = "test",
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__(biodex_entry_cols, dataset_id)

        # for some weird reason we need to put the dataset through a generator to get items as dicts
        self.dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")
        self.dataset = [self.dataset[split][idx] for idx in range(len(self.dataset[split]))]

        # shuffle records if shuffle = True
        if shuffle:
            random.Random(seed).shuffle(self.dataset)

        # trim to number of samples
        self.dataset = self.dataset[:num_samples]
        self.reactions_only = reactions_only
        self.rp_at_k = rp_at_k
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

    def compute_label(self, entry: dict) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
        target_lst = entry["target"].split("\n")
        target_reactions = [reaction.strip().lower() for reaction in target_lst[3].split(":")[-1].split(",")]
        label_dict = {
            "reactions": target_reactions,
            "reaction_labels": target_reactions,
            "ranked_reaction_labels": target_reactions,
        }
        if not self.reactions_only:
            label_dict = {
                "serious": int(target_lst[0].split(":")[-1]),
                "patientsex": int(target_lst[1].split(":")[-1]),
                "drugs": [drug.strip().lower() for drug in target_lst[2].split(":")[-1].split(",")],
                **label_dict,
            }

        return label_dict

    @staticmethod
    def rank_precision_at_k(k: int, preds: list | None, targets: list):
        if preds is None:
            return 0.0

        try:
            # lower-case each list
            preds = [pred.lower() for pred in preds]
            targets = set([target.lower() for target in targets])

            # compute rank-precision at k
            rn = len(targets)
            denom = min(k, rn)
            total = 0.0
            for i in range(k):
                total += preds[i] in targets if i < len(preds) else 0.0

            return total / denom

        except Exception:
            os.makedirs("rp@k-errors", exist_ok=True)
            ts = time.time()
            with open(f"rp@k-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    @staticmethod
    def f1_eval(preds: list | None, targets: list):
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
            os.makedirs("f1-eval-errors", exist_ok=True)
            ts = time.time()
            with open(f"f1-eval-errors/error-{ts}.txt", "w") as f:
                f.write(str(preds))
            return 0.0

    def __len__(self):
        return len(self.dataset)

    def get_item(self, idx: int):
        # get entry
        entry = self.dataset[idx]

        # get input fields
        pmid = entry["pmid"]
        title = entry["title"]
        abstract = entry["abstract"]
        fulltext = entry["fulltext"]

        # create item with fields
        item = {"fields": {}, "labels": {}, "score_fn": {}}
        item["fields"]["pmid"] = pmid
        item["fields"]["title"] = title
        item["fields"]["abstract"] = abstract
        item["fields"]["fulltext"] = fulltext

        # add label info
        item["labels"] = self.compute_label(entry)

        # add scoring functions for list fields
        rank_precision_at_k = partial(BiodexSource.rank_precision_at_k, k=self.rp_at_k)
        item["score_fn"]["reactions"] = BiodexSource.f1_eval
        item["score_fn"]["reaction_labels"] = BiodexSource.f1_eval
        item["score_fn"]["ranked_reaction_labels"] = rank_precision_at_k,
        if not self.reactions_only:
            item["score_fn"]["drugs"] = BiodexSource.f1_eval

        return item


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
    plan, val_datasource, use_final_op_quality = None, None, False
    if workload == "enron":
        # datasetid="enron-eval" for paper evaluation
        data_filepath = f"testdata/{datasetid}"
        val_dataset_id = f"val-{datasetid}"

        # create and register validation data source
        val_datasource = EnronSource(file_dir=data_filepath, dataset_id=val_dataset_id)
        DataDirectory().register_user_source(src=val_datasource, dataset_id=val_dataset_id)

        plan = Dataset(datasetid).sem_add_columns(email_cols)
        plan = plan.sem_filter(
            "The email is not quoting from a news article or an article written by someone outside of Enron"
        )
        plan = plan.sem_filter(
            'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
        )

    elif workload == "real-estate":
        # datasetid="real-estate-eval-100" for paper evaluation
        data_filepath = f"testdata/{datasetid}"
        val_dataset_id = f"val-{datasetid}"

        # create and register validation data source
        val_datasource = RealEstateSource(
            dataset_id=val_dataset_id,
            listings_dir=data_filepath,
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(src=val_datasource, dataset_id=val_dataset_id)

        plan = Dataset(datasetid)
        plan = plan.sem_add_columns(real_estate_text_cols, depends_on="text_content")
        plan = plan.sem_add_columns(real_estate_image_cols, depends_on="image_filepaths")
        plan = plan.sem_filter(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
        )
        plan = plan.filter(within_two_miles_of_mit, depends_on="address")
        plan = plan.filter(in_price_range, depends_on="price")

    elif workload == "biodex-reactions":
        # create and register data source
        datasource = BiodexSource(
            dataset_id=datasetid,
            reactions_only=True,
            split="test",
            num_samples=250,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(src=datasource, dataset_id=datasetid)

        # create and register validation data source
        val_dataset_id = "val-biodex"
        val_datasource = BiodexSource(
            dataset_id=val_dataset_id,
            reactions_only=True,
            split="train",
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(src=val_datasource, dataset_id=val_dataset_id)

        # load index
        index_path = ".ragatouille/colbert/indexes/reaction-terms"
        index = RAGPretrainedModel.from_index(index_path)

        # construct plan
        plan = Dataset(datasetid)
        plan = plan.sem_add_columns(biodex_reactions_cols)

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
        plan = plan.sem_add_columns(biodex_ranked_reactions_labels_cols)

        # only use final op quality
        use_final_op_quality = True

    elif workload == "biodex":
        # create and register data source
        datasource = BiodexSource(
            dataset_id=datasetid,
            reactions_only=False,
            split="test",
            num_samples=250,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(src=datasource, dataset_id=datasetid)

        # create and register validation data source
        val_dataset_id = "val-biodex"
        val_datasource = BiodexSource(
            dataset_id=val_dataset_id,
            reactions_only=False,
            split="train",
            num_samples=val_examples,
            shuffle=False,
            seed=seed,
        )
        DataDirectory().register_user_source(src=val_datasource, dataset_id=val_dataset_id)

        # load index
        index_path = ".ragatouille/colbert/indexes/reaction-terms"
        index = RAGPretrainedModel.from_index(index_path)

        # construct plan
        plan = Dataset(datasetid)
        plan = plan.sem_add_columns(biodex_serious_cols, depends_on=["title", "abstract", "fulltext"])
        plan = plan.sem_add_columns(biodex_patient_sex_cols, depends_on=["title", "abstract", "fulltext"])
        plan = plan.sem_add_columns(biodex_drugs_cols, depends_on=["title", "abstract", "fulltext"])
        plan = plan.sem_add_columns(biodex_reactions_cols, depends_on=["title", "abstract", "fulltext"])

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
        plan = plan.sem_add_columns(biodex_ranked_reactions_labels_cols)

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
        val_datasource=val_datasource,
        available_models=available_models,
        processing_strategy=args.processing_strategy,
        optimizer_strategy=optimizer_strategy,
        execution_strategy=args.execution_strategy,
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
