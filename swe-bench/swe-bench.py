import palimpzest as pz
from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord 
from palimpzest.core.lib.schemas import RawJSONObject, TextFile
from palimpzest.core.lib.fields import Field 
from palimpzest.agents.debugger_agent import DebuggerAgent
from palimpzest.agents.code_editor_agent import CodeEditorAgent
from palimpzest.query.processor.config import QueryProcessorConfig

import json
import re

class GithubIssue(TextFile):
    """Represents a github issue containing a problem statement and the relevant issue code pertaining to the issue."""

    instance_id = Field(
        desc="The instance_id",
    )
    problem_statement = Field(
        desc="A text description of the github issue which can be found within the problem statement field of the provided json object. It may also include helpful exchanges between developers relating to the issue.",
    )
    relevant_issue_code = Field(
        desc="The relevant code pertaining to the issue. Code across multiple files may be included where the start and end of each file is indicated by 'start of' and 'end of' statemnts. ",
    )
    base_commit = Field(
        desc="This is the commit that the code is based on.",
    )
  
class CodeFix(GithubIssue):
    new_code = Field(
        desc="You are an expert software engineer. You are provided the relevant issue code and a problem statement describing an issue to fix in the code. Your task is to return the entire modified version of the provided code after addressing the problem, while keeping the structure and any necessary comments intact. This includes preserving the line numbers and the start and end file boundary markings ([start of <filename>]). Make sure to also return every file that was provided, even those that were not modified.", 
    )
    change_description = Field(
        desc="A description of the code change made. Be specific and use line numbers.", 
    )

class SimplifiedCodeFix(RawJSONObject):
    change_description = Field(
        desc="A description of the code change made. Be specific and use line numbers.", 
    )
    instance_id = Field(
        desc="The instance_id",
    )
    relevant_issue_code = Field(
        desc="The relevant code pertaining to the issue. Code across multiple files may be included where the start and end of each file is indicated by 'start of' and 'end of' statemnts. ",
    )
    code_fix = Field(
        desc="You are an expert software engineer. You are provided the relevant issue code and a problem statement describing an issue to fix in the code. Your task is to return the entire modified version of the provided code after addressing the problem, while keeping the structure and any necessary comments intact. This includes preserving the line numbers and the start and end file boundary markings ([start of <filename>]). Make sure to also return every file that was provided, even those that were not modified.", 
    )

class GithubCodePatch(RawJSONObject):
    model_patch = Field(
        desc="You are a version control assistant tasked with generating a GitHub code patch (diff format) to represent changes between two versions of code: the relevant issue code and the code fix. Each file in the code is enclosed within boundaries marked by [start of <filename>] and [end of <filename>], with line numbers provided. A description of the change will also be provided for context. Analyze the differences between the two versions and produce a patch that correctly reflects the modifications. Ignore any changes in formatting and excessive new lines. For reference, here is an example of a GitHub patch format: diff --git a/example.py b/example.py --- a/example.py +++ b/example.py @@ -1,3 +1,3 @@ -print('Hello, world!') +print('Hello, Python!') print('This is line 2.') print('This is line 3.') Use this format to generate the required patch.",
    )

    instance_id = Field(
        desc="The instance id",
    )

def parse_files(input_string):
    """
    Parses a single input string containing multiple files marked with
    `[start of ...]` and `[end of ...]`.

    Args:
        input_string (str): Input string with all the code.

    Returns:
        dict: A dictionary where keys are file names and values are file content.
    """
    file_pattern = re.compile(r'\[start of (.+?)\](.*?)\[end of \1\]', re.DOTALL)
    matches = file_pattern.findall(input_string)
    files = {match[0].strip(): match[1].strip() for match in matches}
    return files

def extract_relevant_fields(candidate: DataRecord):
    data = json.loads(candidate['contents'])

    github_issue = {}

    github_issue 
    github_issue['instance_id'] = data['instance_id']
    github_issue['problem_statement'] = data['problem_statement']
    github_issue['base_commit'] = data['base_commit']

    # Regular expression to extract content between <code> and <code/>
    pattern = r"<code>(.*?)</code>"

    # Search for the content
    match = re.search(pattern, candidate['contents'])

    if match:
        extracted_content = match.group(1)
    else: 
        extracted_content = ""
        print("Relevant code not found")
    
    github_issue['relevant_issue_code'] = extracted_content
    github_issue['contents'] = ''
    github_issue['filename'] = data['instance_id']

    return github_issue

def remove_irrelevant_fields(candidate: DataRecord):
    # code_fix = DataRecord(schema=SimplifiedCodeFix)
    code_fix = {}
    code_fix['instance_id'] = candidate['instance_id']
    code_fix['relevant_issue_code'] = candidate['relevant_issue_code']
    code_fix['code_fix'] = candidate['new_code'] 
    # code_fix.contents = ""
    # code_fix.problem_statement = ""
    code_fix['change_description'] = candidate['change_description']
    # code_fix.filename = data['instance_id']

    return code_fix 

def buildSweBenchPlan(dataset):
    debugger_agent = DebuggerAgent()
    code_agent = CodeEditorAgent()

    github_issues = pz.Dataset(dataset, schema=GithubIssue, udf=extract_relevant_fields, cardinality=Cardinality.ONE_TO_ONE)
    code_plans = debugger_agent(github_issues)
    code_patches = code_agent(code_plans)
    return code_patches
    # code_fixes = github_issues.convert(outputSchema=CodeFix)
    # code_fixes = code_plans.convert(output_schema=CodeFix)
    # patches = code_fixes.convert(outputSchema=GithubCodePatch, udf=generate_patch)
    # simplified_code_fixes = code_fixes.convert(output_schema=SimplifiedCodeFix, udf=remove_irrelevant_fields) patches = simplified_code_fixes.convert(output_schema=GithubCodePatch)
    # verified_patches = patches.verify(run_tests, retries=3)
    # return patches

def dump_records(filename, records, values, all_values=False):
    # Filter the dictionaries to include only specified columns (keys)
    dict_list = []

    for record in records:
        if all_values: 
            instance_patch = record._asDict()
        else: 
            instance_patch = {key: record._asDict().get(key) for key in values if key in record._asDict()}
        instance_patch['model_name_or_path'] = 'palimpzest'
        dict_list.append(instance_patch)

    # Create the file (or clear it if it exists)
    open(filename, 'w').close()

    # Write the filtered list of dictionaries to the file
    with open(filename, 'w') as file:
        json.dump(dict_list, file, indent=4)

if __name__ == "__main__":
    plan = buildSweBenchPlan("swe-bench-oracle-lite")

    # execute pz plan
    config = QueryProcessorConfig(
        nocache=True,
        execution_strategy="sequential",
    )
    data_record_collection = plan.run(config)

    # execution_engine = pz.PipelinedParallelNoSentinelExecution
    # records, plan_stats = pz.Execute(plan, 
    #                             policy=policy,
    #                             nocache=True,
    #                             allow_token_reduction=False,
    #                             allow_code_synth=False,
    #                             execution_engine=execution_engine,
    #                             verbose=True)

    import pdb; pdb.set_trace() 
    # TO DO: Save the output in JSON file 

    # print(f'Record Type: {records}')
    
    # print(f"Policy is: {str(policy)}")
    # print("Executed plan:")

    # plan_str = list(plan_stats.plan_strs.values())[0]
    # print(plan_str)

    # # Output Records into json file
    # filename = 'output.json'
    # dump_records(filename, records, values=['instance_id', 'model_patch']) 
