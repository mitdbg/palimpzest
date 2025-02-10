
from palimpzest.agents.base import BaseAgent
from palimpzest.sets import Dataset
from palimpzest.constants import Cardinality
from palimpzest.elements import DataRecord
from palimpzest.generators import get_api_key
import palimpzest as pz
import dspy
from dspy import Tool
import re

class FixPlan(pz.RawJSONObject):
  """ Defines a plan for fixing a code issue """
  plan = pz.ListField(
    element_type=pz.StringField,
    desc="The list of steps required to fix the code issue", 
    required=True
  )
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

class PlanGeneration(dspy.Signature): 
  """ Generates a report on the cause of the code issue and how it can be fixed """
  code: str = dspy.InputField()
  problem_statement: str = dspy.InputField()
  fix_report: str = dspy.OutputField()

class PlannerAgent(BaseAgent): 
  def __init__(self):
    openai_key = get_api_key("OPENAI_API_KEY")
    max_tokens = 4096
    self.model = dspy.OpenAI(
      model='gpt-4o-mini',
      api_key=openai_key,
      temperature=0.0,
      max_tokens=max_tokens,
      logprobs=True,
    )
    self.relevant_issue_code = None

  def __call__(self, data: Dataset) -> Dataset:
    """ Generates a solution plan for the Datasets """
    # needs to return a Dataset object
    # Dataset object implements solution doc generation 
    # Can use UDF functionality here-> write a UDF that generates a solution doc 

    return Dataset(
        source=data,
        schema=FixPlan,
        udf=self.generate_plan,
        cardinality=Cardinality.ONE_TO_ONE,
    )
  
  def generate_plan(self, candidate: DataRecord) -> DataRecord:
    dspy.configure(lm=self.model)

    plan = DataRecord(schema = FixPlan)

    # Propogate values from candidate
    plan.instance_id = candidate.instance_id
    plan.problem_statement = candidate.problem_statement
    plan.relevant_issue_code = candidate.relevant_issue_code
    self.relevant_issue_code = candidate.relevant_issue_code

    # Maybe have a step to clean up and summarize the problem statement 

    # Maybe have a step to clean and format the code 

    # Maybe have a debug step first to understand and identify the source of the problem

    # Uses DSPy to generate a plan for fixing the code issue
    # TO DO: Implement this functionality in generators.py instead
    import pdb; pdb.set_trace()
    react = dspy.ReAct(PlanGeneration, tools=[Tool(self.get_classes_and_methods), Tool(self.get_range)])
    plan.plan = react(code = candidate.relevant_issue_code, problem_statement = candidate.problem_statement)
    return plan 
  
  
  def get_classes_and_methods(self, file_name: str) -> str:
    """ Gets all the classes and methods in a file, returning a dict where keys are classes and values are a list of methods """

    # TO DO: extract the code from within a single file
    pattern = rf"\[start of ([^\]]*{re.escape(file_name)}[^\]]*)\](.*?)\[end of \1\]"
    
    match = re.search(pattern, self.relevant_issue_code, re.DOTALL)
    if match: 
      file_code = match.group(2).strip()

      res = self.model(f"Given the below code file, return only a dictionary (no explanation) where the keys are classes and the values are a list of methods in the class: {file_code}")
      return res
    else: 
      return "File not found"
    
  def get_range(self, file_name: str, start_line: str, end_line: str) -> str:
    """ Gets the code in a file between a range of lines """
    
    # Extract the code from within a single file
    pattern = rf"\[start of ([^\]]*{re.escape(file_name)}[^\]]*)\](.*?)\[end of \1\]"
    
    match = re.search(pattern, self.relevant_issue_code, re.DOTALL)
    if match: 
      file_code = match.group(2).strip()

      # Get the code between the start and end lines
      pattern = fr"{start_line}(.*?){end_line}"
      match = re.search(pattern, file_code, re.DOTALL)

      if match:
          res = match.group(1)
          return res
      else: 
        return "Line range not found"
    else: 
      return "File not found" 