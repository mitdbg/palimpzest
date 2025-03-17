
from palimpzest.agents.base_agent import BaseAgent, GLOBAL_CONTEXT
from palimpzest.sets import Dataset
from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import Field 
from palimpzest.core.lib.schemas import RawJSONObject
from palimpzest.core.lib.schemas import Schema
import palimpzest.agents.utils as utils
import json
import dspy
from dspy import Tool
import re

PARAMS = {
    "MAX_ITERS": 20, 
    "VERIFY_LOOPS": 2,
}

LOGGER = utils.setup_logger()

class FixPlan(Schema):
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

class DebugGeneration(dspy.Signature): 
    """ 
    Generates a detailed report of the root cause of the code issue and how it can be fixed, refering to function, class, and file names. 
    Include how the bug was discovered, detailing the files, functions, and classes that were examined in its discovery. 
    """

    problem_statement: str = dspy.InputField(desc="A description of the problem causing the bug")
    instance_id: str = dspy.InputField(desc="An execution identifier used as an argument for tools")
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
        print(f'=============== DEBUGGER AGENT START for {candidate["instance_id"]} ===============')

        dspy.configure(lm=GLOBAL_CONTEXT['model'])
        print(f'Model: {dspy.settings.lm.model}')

        # Clean problem statement
        problem_statement = re.sub(r'<!--.*?-->', '', candidate['problem_statement'], flags=re.DOTALL).strip()

        plan = {
            'instance_id': candidate['instance_id'],
            'problem_statement': problem_statement,
        }

        # Set instance values
        pattern = r'^(?P<owner>[^_]+)__(?P<repo>.+)-\d+$'
        match = re.match(pattern, candidate['instance_id'])
        if match:
            owner = match.group('owner')
            repo = match.group('repo')

        BaseAgent.instance_params[candidate['instance_id']] = {
            "base_commit": candidate['base_commit'],
            "owner": owner, 
            "repo": repo,
        }

        print(f'Instance Params: {BaseAgent.instance_params[candidate["instance_id"]]}')

        # TO DO: Maybe we can try this a few times and generate a few theories to combine at the end
        # Provide the next iteration, the tools used and the output in order to guide further investigation

        # Error handling and incremental write

        react = dspy.ReAct(
            DebugGeneration, 
            tools=[
                Tool(BaseAgent.get_classes_and_methods),
                Tool(BaseAgent.get_file_content),
                Tool(BaseAgent.extract_method), 
                Tool(BaseAgent.search_keyword),
                # TO DO: implement get_class() tool (?)
            ],
            max_iters=PARAMS['MAX_ITERS'],
        )

        result = react(instance_id=candidate['instance_id'], problem_statement=problem_statement) 

        plan['bug_report'] = result.fix_report

        pretty_trajectory = json.dumps(result.toDict(), indent=4)

        if BaseAgent.LOGGING_ENABLED:
            LOGGER.info(f'Debugger Trajectory {plan["instance_id"]}: {pretty_trajectory}')

        cumulative_cost = utils.compute_cost_from_history(dspy.settings.lm.history)
        print(f'Debugger Agent Cumulative Cost: {cumulative_cost}')

        return plan 