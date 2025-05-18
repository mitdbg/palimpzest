
from palimpzest.agents.base_agent import BaseAgentOp
from palimpzest.agents.react import ReAct 
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.data.dataclasses import GenerationStats  
import palimpzest.constants as constants 
import palimpzest.agents.utils as utils
import time
import json
import dspy
from dspy import Tool
import re

LOGGER = utils.setup_logger()

class DebugGeneration(dspy.Signature): 
    """ 
    Generates a detailed json formatted report of the root cause of the code issue and how it can be fixed, refering to function, class, and file names. 
    Include how the bug was discovered, detailing the files, functions, and classes that were examined in its discovery. 

    Structure:  
        - The report should be formatted as a json object with two fields: "report" and "files". 
        - The "report" field should contain the report and the "files" field should contain a list of the files that should be modified or examined to make a fix. 
        - Only a json object should be returned

    An Example Output: 

    {
        "report": "The root cause of the bug is the `_return_list_of_arrays` function in the `astropy/wcs/wcs.py` file, which does not handle empty input arrays properly...",
        "files": ["file1.py", "file2.py", ...]
    }

    """

    problem_statement: str = dspy.InputField(desc="A description of the problem causing the bug")
    instance_id: str = dspy.InputField(desc="An execution identifier used as an argument for tools")
    bug_report: str = dspy.OutputField(desc="A report detailing the cause of the bug and how it can be fixed, referencing to exact line numbers and files.")

class DebuggerAgentOp(BaseAgentOp): 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
  
    def run_agent(self, candidate: DataRecord) -> dict:
        print(f'=============== DEBUGGER AGENT START for {candidate["instance_id"]} ===============')
        print(f'Max Iterations: {self.max_iters}, Model: {self.model}')

        self.set_model()
        print(f'Model: {dspy.settings.lm.model}')

        # Clean problem statement
        problem_statement = re.sub(r'<!--.*?-->', '', candidate['problem_statement'], flags=re.DOTALL).strip()

        plan = {
            'instance_id': candidate['instance_id'],
            'problem_statement': problem_statement,
        }

        # Set instance values in class instance_params
        pattern = r'^(?P<owner>[^_]+)__(?P<repo>.+)-\d+$'
        match = re.match(pattern, candidate['instance_id'])
        if match:
            owner = match.group('owner')
            repo = match.group('repo')

        BaseAgentOp.instance_params[candidate['instance_id']] = {
            "base_commit": candidate['base_commit'],
            "owner": owner, 
            "repo": repo,
        }

        # TO DO: Maybe we can try this a few times and generate a few theories to combine at the end
        # Provide the next iteration, the tools used and the output in order to guide further investigation
        # Error handling 

        react = ReAct(
            DebugGeneration, 
            tools=[
                Tool(BaseAgentOp.get_classes_and_methods),
                Tool(BaseAgentOp.get_file_content),
                Tool(BaseAgentOp.extract_method), 
                Tool(BaseAgentOp.search_keyword),
                Tool(BaseAgentOp.extract_class)
            ],
            max_iters=self.max_iters,
            context_size = self.context_size
        )

        start_time = time.time()
        result = react(instance_id=candidate['instance_id'], problem_statement=problem_statement) 

        plan['bug_report'] = result.bug_report

        # Construct generation stats
        input_tokens = react.get_total_input_tokens()
        output_tokens = react.get_total_output_tokens()
        usd_per_input_token, usd_per_output_token = self.get_token_costs()

        generation_stats = GenerationStats(
            model_name=str(dspy.settings.lm.model),
            llm_call_duration_secs=time.time() - start_time, 
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=input_tokens * usd_per_input_token,
            total_output_cost=output_tokens * usd_per_output_token,
            cost_per_record=input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
        )

        # Logging and printing 
        if BaseAgentOp.LOGGING_ENABLED:
            pretty_trajectory = json.dumps(result.toDict(), indent=4)
            LOGGER.info(f'Debugger Trajectory {plan["instance_id"]} (Max Iters: {self.max_iters}, Model: {self.model}): {pretty_trajectory}')
            
        # if BaseAgentOp.PRINTING_ENABLED:
            # cumulative_cost = utils.compute_cost_from_history(dspy.settings.lm.history)
            # print(f'Debugger Agent Cumulative Cost: {cumulative_cost}')

        return plan, generation_stats

    def get_fields_to_generate(self, candidate: DataRecord) -> list[str]:
        candidate_field_names = candidate.get_field_names()
        return candidate_field_names + ["bug_report"]
        
    
