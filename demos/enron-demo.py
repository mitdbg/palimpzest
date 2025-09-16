import json
import os

import palimpzest as pz
from palimpzest.core.lib.schemas import TextFile


class EnronValidator(pz.Validator):
    def __init__(self, labels_file: str):
        super().__init__()

        self.filename_to_labels = {}
        if labels_file:
            with open(labels_file) as f:
                self.filename_to_labels = json.load(f)

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        filename = input_record["filename"]
        labels = self.filename_to_labels[filename]
        if len(labels) == 0:
            return None

        labels = labels[0]
        return (float(labels["sender"] == output["sender"]) + float(labels["subject"] == output["subject"])) / 2.0


class EnronDataset(pz.IterDataset):
    def __init__(self, dir: str, labels_file: str | None = None, split: str = "test"):
        super().__init__(id="enron", schema=TextFile)
        self.filepaths = [os.path.join(dir, filename) for filename in os.listdir(dir)]
        self.filepaths = self.filepaths[:50] if split == "train" else self.filepaths[50:150]
        self.filename_to_labels = {}
        if labels_file:
            with open(labels_file) as f:
                self.filename_to_labels = json.load(f)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # get input fields
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)
        with open(filepath) as f:
            contents = f.read()

        # create item with fields
        item = {"filename": filename, "contents": contents}

        return item


if __name__ == "__main__":
    # create validator and train_dataset
    validator = EnronValidator(labels_file="testdata/enron-eval-medium-labels.json")
    train_dataset = EnronDataset(dir="testdata/enron-eval-medium", split="train")

    # construct plan
    plan = EnronDataset(dir="testdata/enron-eval-medium", split="test")
    plan = plan.sem_map([
        {"name": "subject", "type": str, "desc": "The subject of the email"},
        {"name": "sender", "type": str, "desc": "The email address of the email's sender"},
    ])
    plan = plan.sem_filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")',
        depends_on=["contents"],
    )
    plan = plan.sem_filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron",
        depends_on=["contents"],
    )

    # execute pz plan
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        k=5,
        j=6,
        sample_budget=100,
        max_workers=20,
        progress=True,
    )
    output = plan.optimize_and_run(train_dataset=train_dataset, validator=validator, config=config)

    # print output dataframe
    print(output.to_df())

    # print precision and recall
    with open("testdata/enron-eval-medium-labels.json") as f:
        filename_to_labels = json.load(f)
        test_filenames = os.listdir("testdata/enron-eval-medium")[50:150]
        filename_to_labels = {k: v for k, v in filename_to_labels.items() if k in test_filenames}

    target_filenames = set(filename for filename, labels in filename_to_labels.items() if labels != [])
    pred_filenames = set(output.to_df()["filename"])
    tp = sum(filename in target_filenames for filename in pred_filenames)
    fp = len(pred_filenames) - tp
    fn = len(target_filenames) - tp

    print(f"PRECISION: {tp/(tp + fp) if tp + fp > 0 else 0.0:.3f}")
    print(f"RECALL: {tp/(tp + fn) if tp + fn > 0 else 0.0:.3f}")
