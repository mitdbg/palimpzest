from palimpzest.agents.base_agent import BaseAgentOp
from palimpzest.core.elements.records import DataRecord
from palimpzest.agents.react import ReAct 
from palimpzest.agents import utils
from palimpzest.agents.debugger_agent import LOGGER
from palimpzest.core.data.dataclasses import GenerationStats
import json
import dspy
import time 
from datetime import datetime
from dspy import Tool


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


class CodeEditorAgentOp(BaseAgentOp):

    def __init__(self, max_iters: int , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iters = max_iters
        self.output_dir = f'output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json' 

    def run_agent(self, candidate: DataRecord) -> dict: 
        # Let the agent navigate the code base with the same tools and provide the bug fix plan

        print(f'\n =============== CODE EDITOR AGENT START for {candidate["instance_id"]} ===============')

        patch = {
            'instance_id': candidate['instance_id'],
            'model_name_or_path': 'palimpzest',
        }

        react = ReAct(
            PatchGeneration, 
            tools=[
                Tool(BaseAgentOp.get_classes_and_methods),
                Tool(BaseAgentOp.get_file_content),
                Tool(BaseAgentOp.extract_method), 
                Tool(BaseAgentOp.search_keyword),
                Tool(CodeEditorAgentOp.create_patch),
            ],
            max_iters=self.max_iters,
        )

        start_time = time.time()

        result = react(
            instance_id=candidate['instance_id'],
            bug_report=candidate['bug_report'],
            problem_statement=candidate['problem_statement'], 
        )

        # TO DO: Implement patch/code verification
        # May want to clean patch (new lines, extra tokens, etc)
        patch['model_patch'] = result.code_patch

        cumulative_cost = utils.compute_cost_from_history(dspy.settings.lm.history)

        # Construct generation stats
        # TODO: Compute number of input and output tokens for a single react run
        generation_stats = GenerationStats(
            model_name=str(dspy.settings.lm.model),
            llm_call_duration_secs=time.time() - start_time, 
            # total_input_tokens=input_tokens,
            # total_output_tokens=output_tokens,
            # total_input_cost=input_tokens * usd_per_input_token,
            # total_output_cost=output_tokens * usd_per_output_token,
            # cost_per_record=input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
        )

        # Save patch result 
        utils.add_patch_to_output_dir(self.output_dir, patch)

        if BaseAgentOp.LOGGING_ENABLED:
            pretty_trajectory = json.dumps(result.toDict(), indent=4)
            LOGGER.info(f'Code Editor Trajectory {patch["instance_id"]}: {pretty_trajectory}')

        if BaseAgentOp.PRINTING_ENABLED: 
            print(f'Completed Patch Generation for {candidate["instance_id"]} \n')
            print(f'Code Agent Cumulative Cost: {cumulative_cost} \n')
            print(f'Number of prompts: {len(dspy.settings.lm.history)} \n')

        return patch, generation_stats

    def clean_patch(patch: str) -> str:
        # TO DO: Implement patch cleaning 
        pass


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

        if BaseAgentOp.PRINTING_ENABLED:
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
    
    def get_fields_to_generate(self, candidate: DataRecord) -> list[str]:
        candidate_field_names = candidate.get_field_names()
        return candidate_field_names + ['model_patch', 'model_name_or_path']


