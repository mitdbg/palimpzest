from palimpzest.agents.base_agent import BaseAgent
from palimpzest.core.lib.fields import Field 
from palimpzest.core.lib.schemas import RawJSONObject
from palimpzest.sets import Dataset
from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import Schema
from palimpzest.agents import utils
from palimpzest.agents.debugger_agent import LOGGER
import dspy
from dspy import Tool

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
    An example of a GitHub patch format is as follows: diff --git a/astropy/coordinates/angles.py b/astropy/coordinates/angles.py --- a/astropy/coordinates/angles.py +++ b/astropy/coordinates/angles.py @@ -587,7 +587,7 @@ def _validate_angles(self, angles=None): if angles.unit is u.deg: limit = 90 elif angles.unit is u.rad: - limit = 0.5 * np.pi + limit = self.dtype.type(0.5 * np.pi) else: limit = u.degree.to(angles.unit, 90.0)
    """
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
            max_iters=10    
        )

        result = react(
            bug_report=candidate['bug_report'],
            problem_statement=candidate['problem_statement'], 
        )
        LOGGER.info(f'Code Editor Trajectory {patch['instance_id']}: {result.trajectory}')
        patch['model_patch'] = result.code_patch

        import pdb; pdb.set_trace()
        return patch


    def create_patch(patch_data: dict) -> str: 
      """
      Generate a GitHub patch string from a dictionary representing diff data.
      
      The expected structure of patch_data is:
        {
            "files": [
                {
                    "old_path": "path/to/old/file",
                    "new_path": "path/to/new/file",
                    "hunks": [
                        {
                            "old_start": <int>,
                            "old_length": <int>,
                            "new_start": <int>,
                            "new_length": <int>,
                            "lines": [
                                {"type": "context" | "addition" | "deletion", "content": <str>},
                                ...
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
      
      If a line's content already starts with a prefix (' ', '+', or '-'),
      it will be used as is; otherwise, the prefix is added based on the "type".
      """
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
                    # If content already has a prefix, use it directly.
                    if content and content[0] in (" ", "+", "-"):
                        patch_lines.append(content)
                    else:
                        # Otherwise, add a prefix based on the line type.
                        if line["type"] == "context":
                            patch_lines.append(" " + content)
                        elif line["type"] == "addition":
                            patch_lines.append("+" + content)
                        elif line["type"] == "deletion":
                            patch_lines.append("-" + content)
                        else:
                            patch_lines.append(content)
        
            return "\n".join(patch_lines)


