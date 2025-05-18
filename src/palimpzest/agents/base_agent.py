
import palimpzest.agents.utils as utils
from typing import Any
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.data.dataclasses import GenerationStats, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.generators.generators import get_api_key
from palimpzest.core.data.dataclasses import GenerationStats, OperatorCostEstimates 
from palimpzest.agents.constants import DEBUGGER_NAIVE_EST_NUM_INPUT_TOKENS_PER_ITER, CODE_EDITOR_NAIVE_EST_NUM_OUTPUT_TOKENS_PER_ITER
from palimpzest.constants import MODEL_CARDS
import palimpzest.constants as constants
from abc import ABC, abstractmethod
import json
import time
import dspy
import ast

# TYPE DEFINITIONS
FieldName = str

LOGGER = utils.setup_logger()

openai_key = get_api_key("OPENAI_API_KEY")

# GLOBAL VARIABLES
# TO DO: Find a better way to create state accessible  

class BaseAgentOp(PhysicalOperator):
    # instance_id -> {"base_commit": <str>, "owner": <str>, "repo": <str>}
    instance_params = {}
    PRINTING_ENABLED = False 
    LOGGING_ENABLED = True 

    def __init__(
        self,
        max_iters: int,
        agent_name: str = None, 
        model: str = 'gpt-4o',
        context_size: int = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.agent_name = agent_name
        self.max_iters = max_iters
        self.model = model
        self.context_size = context_size
    
    def __str__(self):
        op = super().__str__()
        op += f"    Agent Name: {self.agent_name}\n"
        op += f"    Max Iterations: {self.max_iters}\n"
        op += f"    Model: {self.model}\n"
        op += f"    Context Size: {self.context_size}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {"agent_name": self.agent_name, 
                     "max_iters": self.max_iters, 
                     "model": self.model,
                     "context_size": self.context_size,
                     **id_params
                    }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {"agent_name": self.agent_name, 
                     "max_iters": self.max_iters, 
                     "model": self.model,
                     "context_size": self.context_size,
                     **op_params
                    }

        return op_params

    def set_model(self):
        if self.model == 'gpt-4o':
            model = dspy.LM('openai/gpt-4o', api_key=openai_key)
        elif self.model == 'gpt-4o-mini':
            model = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
        else: 
            raise ValueError(f"Model {self.model} is not supported")

        dspy.configure(lm=model)

    @abstractmethod
    def run_agent(self, candidate: DataRecord) -> dict:
        """
        This abstract method will be implemented by subclasses of BaseAgent to process the input DataRecord
        through an agentic workflow and return the results.
        """
        pass

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        # Get fields the agent will generate
        fields_to_generate = self.get_fields_to_generate(candidate)

        field_answers, generation_stats = self.run_agent(candidate)

        # transform the mapping from fields to answers into a (list of) DataRecord(s)
        dr = self._create_data_record_from_field_answers(field_answers, candidate)

        record_set = self._create_record_set(
            record=dr,
            fields=fields_to_generate,
            generation_stats=generation_stats,
            total_time=time.time() - start_time,
        )

        return record_set

    def _create_data_record_from_field_answers(
        self,
        field_answers: dict[FieldName, list[Any]],
        candidate: DataRecord,
    ) -> DataRecord:
        """
        Given a mapping from each field to its (list of) generated value(s), we construct the corresponding
        list of output DataRecords.
        """

        # initialize record with the correct output schema, and parent record
        dr = DataRecord.from_parent(self.output_schema, parent_record=candidate)

        # copy all fields from the input record
        # NOTE: this means that records processed by PZ agents will inherit all pre-computed fields
        #       in an incremental fashion; this is a design choice which may be revisited in the future
        for field in candidate.get_field_names():
            setattr(dr, field, getattr(candidate, field))

        # get input field names and output field names
        input_fields = self.input_schema.field_names()
        output_fields = self.output_schema.field_names()

        # parse newly generated fields from the field_answers dictionary for this field; if the list
        # of generated values is shorter than the number of records, we fill in with None
        for field in output_fields:
            if field not in input_fields:
                value = field_answers[field]
                setattr(dr, field, value)
            
        return dr

    
    @abstractmethod
    def get_fields_to_generate(self, candidate):
        """
        This abstract method will be implemented by subclasses of BaseAgent to return the fields that the agent will generate.
        """
        pass
    
    def _create_record_set(
        self,
        record: DataRecord,
        fields: list[str],
        generation_stats: GenerationStats,
        total_time: float,
    ) -> DataRecordSet:
        """
        Construct list of RecordOpStats objects (one for each DataRecord).
        """
        # amortize the generation stats across all generated records
        per_record_stats = generation_stats 
        time_per_record = total_time 

        # create the RecordOpStats objects for each output record
        record_op_stats = RecordOpStats(
            record_id=record.id,
            record_parent_id=record.parent_id,
            record_source_idx=record.source_idx,
            record_state=record.to_dict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time_per_record,
            cost_per_record=per_record_stats.cost_per_record,
            model_name=self.get_model_name(),
            answer={field_name: getattr(record, field_name) for field_name in fields},
            input_fields=self.input_schema.field_names(),
            generated_fields=fields,
            total_input_tokens=per_record_stats.total_input_tokens,
            total_output_tokens=per_record_stats.total_output_tokens,
            total_input_cost=per_record_stats.total_input_cost,
            total_output_cost=per_record_stats.total_output_cost,
            llm_call_duration_secs=per_record_stats.llm_call_duration_secs,
            fn_call_duration_secs=per_record_stats.fn_call_duration_secs,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        # create and return the DataRecordSet
        return DataRecordSet([record], [record_op_stats])

    
    @staticmethod
    def get_file_content(file_name: str, starting_line_number: int, instance_id: str) -> str:
        """
        Returns the content of an entire file including line numbers, up to a set max of 200 lines.
        The starting line number is one indexed so first line in the file is 1, do not use 0. 
        """

        MAX_NUM_LINES = 200

        if BaseAgentOp.PRINTING_ENABLED:
            print(f'get_file_content {file_name}')

        base_commit = BaseAgentOp.instance_params[instance_id]["base_commit"]
        owner = BaseAgentOp.instance_params[instance_id]["owner"]
        repo = BaseAgentOp.instance_params[instance_id]["repo"]

        # Fetch the file content from GitHub
        best_match, content = utils.fetch_github_code(file_name, owner, repo, base_commit)
        if best_match != file_name:
            return f"File {file_name} not found in the repo. Did you mean {best_match}?"
        content = utils.add_line_numbers(content)
        total_lines = len(content)

        # Limit number of lines returned
        end_line = min(len(content) + 1, starting_line_number + MAX_NUM_LINES)
        content = {i: content[str(i)] for i in range(starting_line_number, end_line)}

        # Format the content string 
        content = json.dumps(content, indent=4)
        content_string = f"There are {total_lines} lines in the file. The requested lines are as follows:\n {content}"
        return content_string

    @staticmethod
    def search_keyword(repo_name : str, keyword: str, instance_id: str) -> str:
        """
        Searches the entire codebase for the provided keyword and returns the files the keyword appears in. 
        Provide repo_name in "owner/repo" format. 

        Guidelines: 
        - If searching for a function or class definition, it may be useful to prefix the keyword with "def " or "class ".
        - Do not search for vague keywords that may return too many results, such as "test", "function", or "def".  Search for more specific words instead. 
        - Only the first 15 results will be returned so be specific.

        Usage Tips:
        - Use this to find the location of a function, class definition, or member variable.
        """
        FILE_LIMIT = 15

        if BaseAgentOp.PRINTING_ENABLED:
            print(f'search_keyword {keyword}')

        local_repo_path = utils.download_repo(repo_name)

        base_commit = BaseAgentOp.instance_params[instance_id]["base_commit"]
        matching_files = utils.search_keyword(local_repo_path, base_commit, keyword)

        if len(matching_files) > FILE_LIMIT: 
            matching_files = matching_files[:FILE_LIMIT]

        return matching_files

    @staticmethod
    def extract_method(file_name: str, function_name: str, instance_id: str, include_line_numbers: bool = False, class_name: str = None) -> str:
        """
        Extracts the implementation of a function from a file. 
        Provide class_name if the function is inside a class. 

        Guidelines: 
        - Only set include_line_numbers to True if being used to generate a patch.
        """

        if BaseAgentOp.PRINTING_ENABLED:
            print(f'extract_method {function_name} from {file_name}, class={class_name}')

        try:
            # Parse the source code into an AST
            base_commit = BaseAgentOp.instance_params[instance_id]["base_commit"]
            owner = BaseAgentOp.instance_params[instance_id]["owner"]
            repo = BaseAgentOp.instance_params[instance_id]["repo"]
            best_match, content = utils.fetch_github_code(file_name, owner, repo, base_commit)
            if best_match != file_name: 
                return f"File {file_name} not found in the repo. Did you mean {best_match}?"
            tree = ast.parse(content)
        except SyntaxError as e:
            return "extract_method(): Error parsing the code"
        
        # Walk through nodes in the AST
        target_function_node = None

        if class_name:
            # Search inside classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for class_node in node.body:
                        if isinstance(class_node, ast.FunctionDef) and class_node.name == function_name:
                            target_function_node = class_node
                            break
                    if target_function_node:
                        break
        else:
            # Search top-level functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    target_function_node = node
                    break

        if not target_function_node:
            if class_name:
                return f"Function '{function_name}' not found inside class '{class_name}'."
            else:
                return f"Function '{function_name}' not found at top level in file."
            
        # Extract the function's source code
        start_line = target_function_node.lineno
        end_line = target_function_node.end_lineno

        lines = content.splitlines()
        function_lines = lines[start_line - 1:end_line]
        method_str = "\n".join(function_lines)

        if include_line_numbers:
            return utils.add_line_numbers(method_str, start_line_no=start_line)
        else:
            return method_str
    
    @staticmethod
    def extract_class(file_name: str, class_name: str, instance_id: str, include_line_numbers: bool = False) -> str:
        """
        Extracts the implementation of a class from a file.

        Guildelines: 
        - Only set include_line_numbers to True if being used to generate a patch.
        """

        if BaseAgentOp.PRINTING_ENABLED:
            print(f'extract_class {class_name} from {file_name}')

        try:
            # Parse the source code into an AST
            base_commit = BaseAgentOp.instance_params[instance_id]["base_commit"]
            owner = BaseAgentOp.instance_params[instance_id]["owner"]
            repo = BaseAgentOp.instance_params[instance_id]["repo"]
            best_match, content = utils.fetch_github_code(file_name, owner, repo, base_commit)
            if best_match != file_name: 
                return f"File {file_name} not found in the repo. Did you mean {best_match}?"
            tree = ast.parse(content)
        except SyntaxError as e:
            return "extract_class(): Error parsing the code"

        target_class_node = None

        # Walk through all nodes in the AST to find the class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                target_class_node = node
                break

        if not target_class_node:
            return f"Class '{class_name}' not found in file."

        start_line = target_class_node.lineno
        end_line = target_class_node.end_lineno

        lines = content.splitlines()
        class_lines = lines[start_line - 1:end_line]
        class_str = "\n".join(class_lines)

        if include_line_numbers:
            return utils.add_line_numbers(class_str, start_line_no=start_line)
        else:
            return class_str

    @staticmethod
    def get_classes_and_methods(file_name: str, instance_id: str) -> str:
        """ 
        Summarizes all the classes and standalone functions in a file.

        The expected output format is: 
        {
            "classes": {
                "<ClassName>": [
                "<method_or_member_function_name>",
                "... more methods ..."
                ],
                "... more classes ..."
            },
            "functions": [
                "<function_name>",
                "... more functions ..."
            ]
        }

        Usage Tips:
        - Use to verify the existence of a function or class in the file.
        - Use to understand the structure and contents of the file.
        """

        if BaseAgentOp.PRINTING_ENABLED:
            print(f'get_class_and_methods for {file_name}')

        # relevant_issue_code = GLOBAL_CONTEXT["relevant_issue_code"]

        # pattern = rf"\[start of ([^\]]*{re.escape(file_name)}[^\]]*)\](.*?)\[end of \1\]"
        # match = re.search(pattern, relevant_issue_code, re.DOTALL)
        
        # if match: 
        #     code = match.group(2).strip()
        # else: 

        base_commit = BaseAgentOp.instance_params[instance_id]["base_commit"]
        owner = BaseAgentOp.instance_params[instance_id]["owner"]
        repo = BaseAgentOp.instance_params[instance_id]["repo"]
        best_match, code = utils.fetch_github_code(file_name, owner, repo, base_commit)
        if best_match != file_name:
            return f"File {file_name} not found in the repo. Did you mean {best_match}?"
        if not code: 
            return "That file is not found in the relevant issue code, please try another file"

        code_structure = utils.extract_structure(code)
        return code_structure 
    
    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        est_input_tokens_per_iter = DEBUGGER_NAIVE_EST_NUM_INPUT_TOKENS_PER_ITER
        est_num_output_tokens_per_iter = CODE_EDITOR_NAIVE_EST_NUM_OUTPUT_TOKENS_PER_ITER
        est_num_iters = self.max_iters if self.max_iters <= 10 else self.max_iters * 0.7

        # Compute cardinality
        cardinality = source_op_cost_estimates.cardinality

        # Get model name
        if self.model == 'gpt-4o':
            model_name = 'gpt-4o-2024-08-06' 
        elif self.model == 'gpt-4o-mini':
            model_name = 'gpt-4o-mini-2024-07-18'

        # Compute time per record
        model_conversion_time_per_record = est_num_iters * MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens_per_iter

        # Compute cost per record
        # TODO: Doesn't always go to max_iters - can make better
        model_conversion_usd_per_record = est_num_iters * (
            MODEL_CARDS[model_name]["usd_per_input_token"] * est_input_tokens_per_iter
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens_per_iter
        )

        # Compute quality based on max_iters and model
        quality = 0.8 if self.max_iters >= 10 else 0.8 * (self.max_iters / 10) 
        quality = quality * 0.5 if self.model == 'gpt-4o-mini' else quality

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def get_token_costs(self) -> tuple[float, float]: 
        if self.model == 'gpt-4o':
            usd_per_input_token = constants.GPT_4o_MODEL_CARD['usd_per_input_token']
            usd_per_output_token = constants.GPT_4o_MODEL_CARD['usd_per_output_token']
        elif self.model == 'gpt-4o-mini':
            usd_per_input_token = constants.GPT_4o_MINI_MODEL_CARD['usd_per_input_token']
            usd_per_output_token = constants.GPT_4o_MINI_MODEL_CARD['usd_per_output_token']
        
        return usd_per_input_token, usd_per_output_token