import palimpzest as pz
from palimpzest.core.elements.records import DataRecord 
from palimpzest.core.lib.schemas import RawJSONObject, TextFile, Schema
from palimpzest.core.lib.fields import Field, StringField 
# from palimpzest.query.agents.debugger_agent import DebuggerAgent
# from palimpzest.query.agents.code_editor_agent import CodeEditorAgent
from palimpzest.query.processor.config import QueryProcessorConfig

import json
import os
import re

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


class GithubIssueDataReader(pz.DataReader):
    def __init__(self, path):
        super().__init__(GithubIssue)
        self.path = path
        self.filepaths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # get instance
        instance_filepath = self.filepaths[idx]
        with open(instance_filepath) as f:
            contents = f.read()

        data = json.loads(contents)

        github_issue = {}
        github_issue['instance_id'] = data['instance_id']
        github_issue['problem_statement'] = data['problem_statement']
        github_issue['base_commit'] = data['base_commit']

        return github_issue

def buildSweBenchPlan(path):
    # create instance of data reader
    datareader = GithubIssueDataReader(path)

    # Create dataset
    github_issues = pz.Dataset(datareader)

    # Process GitHub Issues
    code_plans = github_issues.add_agent(agent_name="debugger")
    code_patches = code_plans.add_agent(agent_name="code_editor") 

    return code_patches

def dump_records(filename, records, values, all_values=False):
    # Filter the dictionaries to include only specified columns (keys)
    dict_list = []

    for record in records:
        if all_values: 
            instance_patch = record.to_dict()
        else: 
            instance_patch = {key: record.to_dict().get(key) for key in values if key in record.to_dict()}
        instance_patch['model_name_or_path'] = 'palimpzest'
        dict_list.append(instance_patch)

    # Create the file (or clear it if it exists)
    open(filename, 'w').close()

    # Write the filtered list of dictionaries to the file
    with open(filename, 'w') as file:
        json.dump(dict_list, file, indent=4)

if __name__ == "__main__":
    plan = buildSweBenchPlan("/Users/jasonli/Documents/GitHub/palimpzest/testdata/swe-bench-oracle-lite")

    # execute pz plan
    config = QueryProcessorConfig(
        nocache=True,
        execution_strategy="pipelined",
    )
    data_record_collection = plan.run(config)

    print('COMPLETE')

    # Output Records into json file
    # filename = f'output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json' 
    #dump_records(filename, data_record_collection, values=['instance_id', 'model_patch']) 
