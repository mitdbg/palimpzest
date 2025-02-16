
from palimpzest.agents.base import BaseAgent
from palimpzest.sets import Dataset
from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.generators.generators import get_api_key
from palimpzest.core.lib.fields import Field, ListField, StringField
from palimpzest.core.lib.schemas import RawJSONObject
from palimpzest.agents.utils import fetch_github_code, extract_structure
import palimpzest as pz
import dspy
from dspy import Tool
import ast
import re

# Global variables for relevant code and model
openai_key = get_api_key("OPENAI_API_KEY")

GLOBAL_CONTEXT = {
    "relevant_issue_code": None,
    "model": dspy.LM('openai/gpt-4o-mini', api_key=openai_key),
    "base_commit": None,
}

class FixPlan(RawJSONObject):
  """ Defines a plan for fixing a code issue """
  plan = ListField(
    element_type=StringField,
    desc="The list of steps required to fix the code issue", 
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

class PlanGeneration(dspy.Signature): 
  """ Generates a report on the cause of the code issue and how it can be fixed, detailing exact line numbers """

  relevant_code: str = dspy.InputField(desc="The code where the problem is located")
  problem_statement: str = dspy.InputField(desc="A description of the problem")
  fix_report: str = dspy.OutputField(desc="A report detailing the cause of the code issue and how it can be fixed, referencing to exact line numbers and files.")

class PlannerAgent(BaseAgent): 

    def __call__(self, data: Dataset) -> Dataset:
        """ Generates a solution plan for the Dataset """
        return Dataset(
            source=data,
            schema=FixPlan,
            udf=self.generate_plan,
            cardinality=Cardinality.ONE_TO_ONE,
        )
  
    def generate_plan(self, candidate: DataRecord) -> dict:
        # Store relevant issue code and configure model 
        GLOBAL_CONTEXT["relevant_issue_code"] = candidate['relevant_issue_code']
        GLOBAL_CONTEXT["base_commit"] = candidate['base_commit']
        dspy.configure(lm=GLOBAL_CONTEXT['model'])

        plan = {
            'instance_id': candidate['instance_id'],
            'problem_statement': candidate['problem_statement'],
            'relevant_issue_code': candidate['relevant_issue_code'],
        }

        # TO DO: clean the problem statement 

        react = dspy.ReAct(
            PlanGeneration, 
            tools=[
                Tool(PlannerAgent.get_classes_and_methods),
                Tool(PlannerAgent.get_range),
                Tool(PlannerAgent.extract_method)
            ]
        )

        import pdb; pdb.set_trace()
        plan['plan'] = react(relevant_code=candidate['relevant_issue_code'], problem_statement=candidate['problem_statement'], max_iters=10)
        import pdb; pdb.set_trace()
        return plan 
    
    @staticmethod
    def extract_method(file_name: str, function_name: str) -> str: 
        """
        Given the Python source code as a string and a function name,
        returns the full source code of that function (including decorators,
        signature, and body) or None if the function isn't found.
        """

        import pdb; pdb.set_trace()

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
        """ Summarizes all the classes and methods in a file, returning a dict where keys are classes and values are a list of methods """
        # import pdb; pdb.set_trace()

        relevant_issue_code = GLOBAL_CONTEXT["relevant_issue_code"]
        model = GLOBAL_CONTEXT["model"]

        pattern = rf"\[start of ([^\]]*{re.escape(file_name)}[^\]]*)\](.*?)\[end of \1\]"
        match = re.search(pattern, relevant_issue_code, re.DOTALL)
        
        if match: 
            code = match.group(2).strip()
        else: 
            code = fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
            if not code: 
                return "That file is not found in the relevant issue code, please try another file"

        # import pdb; pdb.set_trace()
        code_structure = extract_structure(code)
        return code_structure 

    @staticmethod
    def get_range(file_name: str, start_line: str, end_line: str) -> str:
        """ Gets the code in a file between a range of lines """
        import pdb; pdb.set_trace()

        code = fetch_github_code(file_name, GLOBAL_CONTEXT["base_commit"])
        lines = code.splitlines()
        range = lines[int(start_line)-1:int(end_line)]  
        return range
