import palimpzest as pz
from palimpzest.core.elements.records import DataRecord 
from palimpzest.core.lib.schemas import RawJSONObject, TextFile, Schema
from palimpzest.core.lib.fields import Field, StringField 
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.model_helpers import get_models
from palimpzest.constants import Model
import random
import json
import os 

class GithubIssue(Schema):
    """Represents a github issue containing a problem statement and the relevant issue code pertaining to the issue."""

    instance_id = StringField(
        desc="The instance_id",
    )
    problem_statement = StringField(
        desc="A text description of the github issue which can be found within the problem statement field of the provided json object. It may also include helpful exchanges between developers relating to the issue.",
    )
    base_commit = StringField(
        desc="This is the commit that the code is based on.",
    )
    relevant_code = StringField(
          desc="The relevant code pertaining to the issue. This is the code that should be modified in the patch."
		)

class GithubIssueDataReader(pz.DataReader):
    def __init__(
        self, 
        path, 
        num_samples: int = 5, 
        shuffle: bool = False, 
        split: str = "test",
        seed: int = 42,
    ):
        super().__init__(GithubIssue)
        self.split = split 
        self.path = path
        self.num_samples = num_samples

        files = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]

        if shuffle:
            files = random.Random(seed).shuffle(files)

        self.filepaths = files[:num_samples]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # get instance
        instance_filepath = self.filepaths[idx]
        with open(instance_filepath) as f:
            contents = f.read()

        data = json.loads(contents)

        # get input fields
        instance_id = data['instance_id']
        problem_statement = data['problem_statement']
        base_commit = data['base_commit']
        relevant_code = data['text']

        github_issue = {"fields": {}, "score_fn": {}, "labels": {}}
        github_issue["fields"]['instance_id'] = instance_id
        github_issue["fields"]['problem_statement'] = problem_statement
        github_issue["fields"]['base_commit'] = base_commit
        github_issue["fields"]['relevant_code'] = relevant_code 

        if self.split == "train":
            # Add the labels for the fix report for Debugger tuning
            github_issue["labels"]["bug_report"] = self.get_modified_files_from_patch(data['patch'])

            # Debugger Scoring function
            github_issue["score_fn"]["bug_report"] = GithubIssueDataReader.compute_file_set_similarity

            # Code Editor Scoring function
            github_issue["score_fn"]["model_patch"] = GithubIssueDataReader.compute_swe_bench_score

        return github_issue

def create_naive_plan1(path):
    # create instance of data reader
    datareader = GithubIssueDataReader(path, num_samples=300, split="test")

    # Create dataset
    github_issues = pz.Dataset(datareader)

    # Localization of bug 
    bug_locations = github_issues.sem_add_columns([
        {"name": "bug_location", "type": str, "desc": "Using the problem statement and the relevant code, reason about where the bug is located in the code."},
        {"name": "relevant_files", "type": str, "desc": "From the problem statement and relevant code, extract the file names that are relevant to the issue. Make sure to include the full file path."},
    ])

    # Generate a fix report 
    bug_reports = bug_locations.sem_add_columns([
        {"name": "bug_report", "type": str, "desc": "Reason about the changes required to fix the bug."},
    ])

    # Create patches
    patches = bug_reports.sem_add_columns([
        {"name": "model_patch", "type": str, "desc": "Generate a Github style patch to fix the issue."},
    ])

    # Remove unecessary columns
    patches = patches.project(["model_patch", "bug_report", "bug_location", "relevant_files", "instance_id", "base_commit"])

    data_record_collection = patches.run(max_quality=True)

    return data_record_collection

if __name__ == "__main__":
    # Execute plan
    path = "/Users/jasonli/Documents/GitHub/palimpzest/testdata/swe-bench-oracle-lite"
    records = create_naive_plan1(path)

    record_jsons = []
    for record in records: 
        record_dict = record.to_dict()
        record_dict = {
            k: v for k, v in record_dict.items()
        }
        record_jsons.append(record_dict)

    with open("naive_plan_records.json", "w") as f:
        json.dump(record_jsons, f)