from palimpzest.agents.base_agent import BaseAgent
from palimpzest.core.lib.fields import Field 
from palimpzest.core.lib.schemas import RawJSONObject
from palimpzest.sets import Dataset
from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import Schema
from palimpzest.agents import utils
from palimpzest.agents.debugger_agent import LOGGER
import json
import dspy
from dspy import Tool

PARAMS = {
    "MAX_ITERS": 20, 
    "VERIFY_LOOPS": 2,
}

class GithubCodePatch(Schema):
    model_patch = Field(
        desc="You are a version control assistant tasked with generating a GitHub code patch (diff format) to represent changes between two versions of code: the relevant issue code and the code fix. Each file in the code is enclosed within boundaries marked by [start of <filename>] and [end of <filename>], with line numbers provided. A description of the change will also be provided for context. Analyze the differences between the two versions and produce a patch that correctly reflects the modifications. Ignore any changes in formatting and excessive new lines. For reference, here is an example of a GitHub patch format: diff --git a/example.py b/example.py --- a/example.py +++ b/example.py @@ -1,3 +1,3 @@ -print('Hello, world!') +print('Hello, Python!') print('This is line 2.') print('This is line 3.') Use this format to generate the required patch.",
    )
    instance_id = Field(
        desc="The instance id",
    )

class PatchGeneration(dspy.Signature):
    """
    Generates a GitHub code patch representing how the github repository of interest must be modified to implement the provided bug fix. 
    An example of a GitHub patch format is as follows: "diff --git a/astropy/io/ascii/html.py b/astropy/io/ascii/html.py \n--- a/astropy/io/ascii/html.py \n+++ b/astropy/io/ascii/html.py \n@@ -349,11 +349,13 @@ def write(self, table): \n    cols = list(table.columns.values()) \n\n    self.data.header.cols = cols \n+   self.data.cols = cols\n\n    if isinstance(self.data.fill_values, tuple): \n    self.data.fill_values = [self.data.fill_values] \n\n    self.data._set_fill_values(cols) \n+   self.data._set_col_formats() \n\n    lines = []"
    Only return the diff string without any extra text or explanation. 
    Make sure the patch has indentation that matches the codebase, no extra syntax, and can be immedietly applied to the git apply command.
    """

    instance_id: str = dspy.InputField(desc="An execution identifier used as an argument for tools")
    bug_report: str = dspy.InputField(desc="The code where the problem is located")
    problem_statement: str = dspy.InputField(desc="A description of the problem causing the bug")
    code_patch: str = dspy.OutputField(desc="A GitHub code patch representing how the github repository of interest must be modified to implement the provided bug fix.")


class CodeEditorAgent(BaseAgent):

    def __call__(self, data: Dataset) -> Dataset:
      """ Generates a solution plan for the Dataset """
      return Dataset(
          source=data,
          schema=GithubCodePatch,
          udf=self.generate_patch,
          cardinality=Cardinality.ONE_TO_ONE,
      )
  
    def generate_patch(self, candidate: DataRecord) -> dict: 
        # Let the agent navigate the code base with the same tools and provide the bug fix plan

        print(f'\n =============== CODE EDITOR AGENT START for {candidate["instance_id"]} ===============')

        patch = {
            'instance_id': candidate['instance_id'],
            'problem_statement': candidate['problem_statement'],
            'relevant_issue_code': candidate['relevant_issue_code'],
        }

        react = dspy.ReAct(
            PatchGeneration, 
            tools=[
                Tool(BaseAgent.get_classes_and_methods),
                Tool(BaseAgent.get_file_content),
                Tool(BaseAgent.extract_method), 
                Tool(BaseAgent.search_keyword),
                Tool(CodeEditorAgent.create_patch),
            ],
            max_iters=PARAMS['MAX_ITERS'],
        )

        result = react(
            instance_id=candidate['instance_id'],
            bug_report=candidate['bug_report'],
            problem_statement=candidate['problem_statement'], 
        )

        pretty_trajectory = json.dumps(result.toDict(), indent=4)

        if BaseAgent.LOGGING_ENABLED:
            LOGGER.info(f'Code Editor Trajectory {patch["instance_id"]}: {pretty_trajectory}')

        # TO DO: Implement patch/code verification
        # May want to clean patch (new lines, extra tokens, etc)
        patch['model_patch'] = result.code_patch

        cumulative_cost = utils.compute_cost_from_history(dspy.settings.lm.history)
        print(f'Code Agent Cumulative Cost: {cumulative_cost}')
        print(f'Number of prompts: {len(dspy.settings.lm.history)}')

        return patch

    def clean_patch(patch: str) -> str:


        return patch


    def create_patch(patch_data: dict, indent_size: str) -> str: 
        """
        Generate a GitHub patch string from a dictionary representing diff data.

        An example patch_data input is: 
        {
            "files": [
                {
                    "old_path": "old/file.txt",
                    "new_path": "new/file.txt",
                    "hunks": [
                    {
                        "old_start": 1,
                        "old_length": 3,
                        "new_start": 1,
                        "new_length": 3,
                        "lines": [
                            {"type": "context", "content": "unchanged line"},
                            {"type": "addition", "content": "added line"},
                            {"type": "deletion", "content": "removed line"}
                        ]
                    }
                }
            ]
        }

        Make sure that the lines array contains dictionaries with "type" and "content" keys.
        If a line's content already starts with a prefix (' ', '+', or '-'),
        it will be used as is; otherwise, the prefix is added based on the "type".
        """

        indent_size = int(indent_size)

        if BaseAgent.PRINTING_ENABLED:
            print(f'create_patch')

        patch_lines = []
        
        for file in patch_data.get("files", []):
                old_path = file["old_path"]
                new_path = file["new_path"]
                patch_lines.append(f"diff --git a/{old_path} b/{new_path}")
                patch_lines.append(f"--- a/{old_path}")
                patch_lines.append(f"+++ b/{new_path}")
                
                for hunk in file.get("hunks", []):
                    old_start = hunk["old_start"]
                    old_length = hunk["old_length"]
                    new_start = hunk["new_start"]
                    new_length = hunk["new_length"]
                    patch_lines.append(f"@@ -{old_start},{old_length} +{new_start},{new_length} @@")
                    
                    for line in hunk.get("lines", []):
                        content = line.get("content", "")
                        if line["type"] == "context":
                            patch_lines.append(" " * indent_size + content)
                        elif line["type"] == "addition":
                            patch_lines.append("+" + " " * (indent_size - 1)  + content)
                        elif line["type"] == "deletion":
                            patch_lines.append("-" + " " * (indent_size - 1)  + content)
                        else:
                            patch_lines.append(content)
                
                return "\n".join(patch_lines)


