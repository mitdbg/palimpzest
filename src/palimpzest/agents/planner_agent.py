
from palimpzest.agents.base import BaseAgent
from palimpzest.sets import Dataset
from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.generators.generators import get_api_key
from palimpzest.core.lib.fields import Field, ListField, StringField
from palimpzest.core.lib.schemas import RawJSONObject
from palimpzest.agents.utils import fetch_github_code, extract_structure, search_files
import palimpzest as pz
import dspy
from dspy import Tool
import ast
import re

# Global variables for relevant code and model
openai_key = get_api_key("OPENAI_API_KEY")

GLOBAL_CONTEXT = {
    "relevant_issue_code": None,
    "model": dspy.LM('openai/gpt-4o', api_key=openai_key),
    "base_commit": None,
}

class FixPlan(RawJSONObject):
  """ Defines a plan for fixing a code issue """
  bug_report = Field(
    desc="A report on the cause of the bug and how it can be fixed, including the problem statement and relevant code",
  )
  instance_id = Field(
      desc="The instance_id",
  )
  problem_statement = Field(
      desc="A text description of the github issue which can be found within the problem statement field of the provided json object. It may also include helpful exchanges between developers relating to the issue.",
  )
  relevant_issue_code = Field(
      desc="The relevant code pertaining to the issue. Code across multiple files may be included where the start and end of each file is indicated by 'start of' and 'end of' statemnts. ",
  )

class DebugGeneration(dspy.Signature): 
  """ 
    Generates a report on the cause of the code issue and how it can be fixed, referencing function and file names 
  """

  relevant_code: str = dspy.InputField(desc="The code where the problem is located")
  problem_statement: str = dspy.InputField(desc="A description of the problem causing the bug")
  fix_report: str = dspy.OutputField(desc="A report detailing the cause of the bug and how it can be fixed, referencing to exact line numbers and files.")

class DebuggerAgent(BaseAgent): 

    def __call__(self, data: Dataset) -> Dataset:
        """ Generates a solution plan for the Dataset """
        return Dataset(
            source=data,
            schema=FixPlan,
            udf=self.generate_debug_plan,
            cardinality=Cardinality.ONE_TO_ONE,
        )
  
    def generate_debug_plan(self, candidate: DataRecord) -> dict:
        # Store relevant issue code and configure model 
        GLOBAL_CONTEXT["relevant_issue_code"] = candidate['relevant_issue_code']
        GLOBAL_CONTEXT["base_commit"] = candidate['base_commit']
        dspy.configure(lm=GLOBAL_CONTEXT['model'])

        # Clean problem statement
        problem_statement = re.sub(r'<!--.*?-->', '', candidate['problem_statement'], flags=re.DOTALL).strip()

        plan = {
            'instance_id': candidate['instance_id'],
            'problem_statement': problem_statement,
            'relevant_issue_code': candidate['relevant_issue_code'],
        }

        # Maybe we can try this a few times and generate a few theories 

        react = dspy.ReAct(
            DebugGeneration, 
            tools=[
                Tool(DebuggerAgent.get_classes_and_methods),
                Tool(DebuggerAgent.get_file_content),
                Tool(DebuggerAgent.extract_method), 
                Tool(DebuggerAgent.search_keyword),
                # Tool(PlannerAgent.get_range),
            ],
            max_iters=10    
        )

        plan['plan'] = react(relevant_code=candidate['relevant_issue_code'], problem_statement=problem_statement)
        print(plan)
        import pdb; pdb.set_trace()
        return plan 

    def get_file_content(file_name: str) -> str:
        """
        Returns the content of an entire file. Use this when the entire context of a file is imporant. 
        """
        content = fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
        return content

    
    @staticmethod
    def search_keyword(keywords: list[str]) -> str:
        """
        Searches the codebase for keywords and returns the files that contain them.
        """
        results = search_files(keywords)

        if results and "items" in results:
            print(f"Found {results['total_count']} matching files:")
            relevant_files = ', '.join([item['name'] for item in results["items"]])
            return relevant_files 
        else:
            print("No results found.")
            return "No Results found"


    @staticmethod
    def extract_method(file_name: str, function_name: str) -> str: 
        """
        Extracts the implementation of a function from a file. Use this when you only need a single function in the file.
        """

        try:
            # Parse the source code into an AST
            content = fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
            tree = ast.parse(content)
        except SyntaxError as e:
            return "extract_method(): Error parsing the code"

        # Walk through all nodes in the AST.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Ensure the node has an end_lineno attribute (Python 3.8+)
                if not hasattr(node, 'end_lineno'):
                    print("Your Python version does not support end_lineno on AST nodes.")
                    return None

                start_line = node.lineno
                end_line = node.end_lineno

                # Split the content into lines and extract the block.
                lines = content.splitlines()
                function_lines = lines[start_line - 1:end_line]
                return "\n".join(function_lines)

        return "function not found in the file"

    @staticmethod
    def get_classes_and_methods(file_name: str) -> str:
        """ 
            Summarizes all the classes and methods in a file, returning a dict where keys are classes and values are a list of method. 
            Useful for understanding the structure of a file for subsequent method extraction. 
        """

        relevant_issue_code = GLOBAL_CONTEXT["relevant_issue_code"]

        pattern = rf"\[start of ([^\]]*{re.escape(file_name)}[^\]]*)\](.*?)\[end of \1\]"
        match = re.search(pattern, relevant_issue_code, re.DOTALL)
        
        if match: 
            code = match.group(2).strip()
        else: 
            code = fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
            if not code: 
                return "That file is not found in the relevant issue code, please try another file"

        code_structure = extract_structure(code)
        return code_structure 

    @staticmethod
    def get_range(file_name: str, start_line: str, end_line: str) -> str:
        """ Gets the code in a file between a range of lines """

        code = fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
        lines = code.splitlines()
        range = lines[int(start_line)-1:int(end_line)]  
        return range
