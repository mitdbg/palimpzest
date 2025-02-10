import palimpzest as pz

from palimpzest.constants import Cardinality
from palimpzest.elements import DataRecord, GroupBySig
import difflib

from bs4 import BeautifulSoup
from PIL import Image
from requests_html import HTMLSession # for downloading JavaScript content
from tabulate import tabulate
from utils import printTable

import gradio as gr
import numpy as np
import pandas as pd

import argparse
import csv
import datetime
import json
import os
import requests
import sys
import time
import re

class GithubIssue(pz.TextFile):
    """Represents a github issue containing a problem statement and the relevant issue code pertaining to the issue."""

    instance_id = pz.Field(
        desc="The instance_id",
        required=True,
    )
    problem_statement = pz.Field(
        desc="A text description of the github issue which can be found within the problem statement field of the provided json object. It may also include helpful exchanges between developers relating to the issue.",
        required=True,
    )
    relevant_issue_code = pz.Field(
        desc="The relevant code pertaining to the issue. Code across multiple files may be included where the start and end of each file is indicated by 'start of' and 'end of' statemnts. ",
        required=True,
    )
  
class CodeFix(GithubIssue):
    new_code = pz.Field(
        desc="You are an expert software engineer. You are provided the relevant issue code and a problem statement describing an issue to fix in the code. Your task is to return the entire modified version of the provided code after addressing the problem, while keeping the structure and any necessary comments intact. This includes preserving the line numbers and the start and end file boundary markings ([start of <filename>]). Make sure to also return every file that was provided, even those that were not modified.", 
        required=True,
    )
    change_description = pz.Field(
        desc="A description of the code change made. Be specific and use line numbers.", 
        required=True,
    )

class SimplifiedCodeFix(GithubIssue):
    code_fix = pz.Field(
        desc="You are an expert software engineer. You are provided the relevant issue code and a problem statement describing an issue to fix in the code. Your task is to return the entire modified version of the provided code after addressing the problem, while keeping the structure and any necessary comments intact. This includes preserving the line numbers and the start and end file boundary markings ([start of <filename>]). Make sure to also return every file that was provided, even those that were not modified.", 
        required=True,
    )

class GithubCodePatch(pz.RawJSONObject):
    model_patch = pz.Field(
        desc="You are a version control assistant tasked with generating a GitHub code patch (diff format) to represent changes between two versions of code: the relevant issue code and the code fix. Each file in the code is enclosed within boundaries marked by [start of <filename>] and [end of <filename>], with line numbers provided. A description of the change will also be provided for context. Analyze the differences between the two versions and produce a patch that correctly reflects the modifications. Ignore any changes in formatting and excessive new lines. For reference, here is an example of a GitHub patch format: diff --git a/example.py b/example.py --- a/example.py +++ b/example.py @@ -1,3 +1,3 @@ -print('Hello, world!') +print('Hello, Python!') print('This is line 2.') print('This is line 3.') Use this format to generate the required patch.",
        required=True,
    )

    instance_id = pz.Field(
        desc="The instance id",
        required=True
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
    
def generate_patch(candidate: DataRecord):
    """
    Generates a unified patch for changes between two multi-file code strings.

    Args:
        before_code (str): The "before" code string containing all files.
        after_code (str): The "after" code string containing all files.

    Returns:
        str: A unified patch for all changes.
    """

    before_code = candidate.relevant_issue_code 
    after_code = candidate.new_code

    # Parse the before and after code into dictionaries
    before_files = parse_files(before_code)
    after_files = parse_files(after_code)

    patch_lines = []

    # Iterate through all files in the "before" code
    for file_name, before_content in before_files.items():
        # Get the corresponding content from the "after" code, default to the original if unchanged
        after_content = after_files.get(file_name, before_content)

        # Split the contents into lines
        before_lines = before_content.replace('\\n', '\n').splitlines()
        after_lines = after_content.replace('\\n', '\n').splitlines()

        # Generate the diff -> becomes a generator object
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{file_name}",
            tofile=f"b/{file_name}",
            lineterm=""
        )

        # Append the diff to the patch
        patch_lines.extend(diff)
        patch_lines.append("")  # Add a blank line between files

    # Handle any new files in the "after" code that were not in the "before" code
    # for file_name, after_content in after_files.items():
    #     if file_name not in before_files:
    #         after_lines = after_content.splitlines()
    #         diff = difflib.unified_diff(
    #             [],
    #             after_lines,
    #             fromfile=f"a/{file_name}",
    #             tofile=f"b/{file_name}",
    #             lineterm=""
    #         )
    #         patch_lines.extend(diff)
    #         patch_lines.append("")

    patch_record = DataRecord(GithubCodePatch)
    patch_record.model_patch = "\n".join(patch_lines)
    patch_record.json = patch_record.model_patch
    return patch_record

def extract_relevant_fields(candidate: DataRecord):
    data = json.loads(candidate.contents)

    github_issue = DataRecord(schema=GithubIssue)
    github_issue.instance_id = data['instance_id']
    github_issue.problem_statement = data['problem_statement']

    # Regular expression to extract content between <code> and <code/>
    pattern = r"<code>(.*?)</code>"

    # Search for the content
    match = re.search(pattern, candidate.contents)

    if match:
        extracted_content = match.group(1)
    else: 
        extracted_content = ""
        print("Relevant code not found")
    
    github_issue.relevant_issue_code = extracted_content
    github_issue.contents = ""
    github_issue.filename = data['instance_id']

    return github_issue

def remove_irrelevant_fields(candidate: DataRecord):
    data = candidate._asDict()
    code_fix = DataRecord(schema=CodeFix)
    code_fix.instance_id = data['instance_id']
    code_fix.relevant_issue_code = data['relevant_issue_code']
    code_fix.code_fix = data['new_code'] 
    code_fix.contents = ""
    code_fix.problem_statement = ""
    code_fix.change_description = data['change_description']
    code_fix.filename = data['instance_id']

    return code_fix 

def buildSweBenchPlan(dataset):
    github_issues = pz.Dataset(dataset, schema=GithubIssue, udf=extract_relevant_fields, cardinality=Cardinality.ONE_TO_ONE)
    code_fixes = github_issues.convert(outputSchema=CodeFix)
    # patches = code_fixes.convert(outputSchema=GithubCodePatch, udf=generate_patch)
    simplified_code_fixes = code_fixes.convert(outputSchema=SimplifiedCodeFix, udf=remove_irrelevant_fields)
    patches = simplified_code_fixes.convert(outputSchema=GithubCodePatch)
    # verified_patches = patches.verify(run_tests, retries=3)
    return patches

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

    policy = pz.MaxQuality()
    execution_engine = pz.PipelinedParallelNoSentinelExecution
    # execution_engine = pz.SequentialSingleThreadNoSentinelExecution 
    records, plan_stats = pz.Execute(plan, 
                                policy=policy,
                                nocache=True,
                                allow_token_reduction=False,
                                allow_code_synth=False,
                                execution_engine=execution_engine,
                                verbose=True)

    print(f'Record Type: {records}')
    
    print(f"Policy is: {str(policy)}")
    print("Executed plan:")

    plan_str = list(plan_stats.plan_strs.values())[0]
    print(plan_str)

    # Output Records into json file
    filename = 'output.json'
    dump_records(filename, records, values=['instance_id', 'model_patch'])
