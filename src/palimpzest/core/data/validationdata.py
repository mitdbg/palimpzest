import json
import os
import logging
from palimpzest.core.data.dataclasses import ExecutionStats

logger = logging.getLogger(__name__)

class ValidationData:
    def __init__(self, file_path: str | None = None, annotations: dict | None = None):
        """
        Initialize ValidationData with a directory path containing execution_stats.updated.json.
        Args:
            file_path: Path to the execution stats file
        """
        self.file_path = file_path
        self.annotations = annotations
        self.expected_output = {}
        self._input_dataset = None
        
        # Read execution stats JSON
        if self.file_path is not None:
            with open(self.file_path) as f:
                execution_stats = json.load(f)
        elif self.annotations is not None:
            execution_stats = self.annotations
        else:
            execution_stats = ExecutionStats().to_json()

        if "plan_stats" in execution_stats:
            working_stats = execution_stats["plan_stats"]
            if len(working_stats) > 1:
                _, working_stats = working_stats.popitem()
        else:
            working_stats = execution_stats

        if "operator_stats" not in working_stats:
            return

        expected_outputs = {}
        for _, op_stats in working_stats["operator_stats"].items():
            logical_op_id = None
            # Process each record's stats
            for record_stats in op_stats["record_op_stats_lst"]:
                logical_op_id = record_stats["logical_op_id"]
                record_source_idx = record_stats["record_source_idx"]

                # If there is no annotations, then we use the record_state and passed_operator
                if "annotations" in record_stats:
                    labels = record_stats["annotations"]["labels"]
                    pass_filter = record_stats["annotations"]["pass_filter"]
                else:
                    labels = record_stats["record_state"]
                    pass_filter = record_stats["passed_operator"]

                # Initialize nested dictionaries if they don't exist
                if logical_op_id not in expected_outputs:
                    expected_outputs[logical_op_id] = {}

                score_fns = {}
                if labels is not None:
                    score_fns = {field: "exact" for field in labels}
                # Store the record state
                expected_outputs[logical_op_id][record_source_idx] = {
                    "labels": labels,
                    "pass_filter": pass_filter,
                    "score_fn": score_fns
                }
            
        self.expected_output = expected_outputs

        self._input_dataset, self._num_samples = self._get_input_dataset(working_stats)
        
    
    # TODO: This is hack. Find a better way to get the original dataset from execution_stats.
    def _get_input_dataset(self, annotated_plan) -> str:
        """
        Returns the input dataset as a DataReader.
        """
        input_dataset = set()
        cnt = 0
        for _, op_stats in annotated_plan["operator_stats"].items():
            if op_stats["op_name"] != "MarshalAndScanDataOp":
                continue
            for record_stats in op_stats["record_op_stats_lst"]:
                if "annotations" in record_stats:
                    input_dataset.add(os.path.dirname(record_stats["annotations"]["labels"]["filename"]))
                else:
                    input_dataset.add(os.path.dirname(record_stats["record_state"]["filename"]))
                cnt += 1
        assert len(input_dataset) == 1, "Only one input dataset (one file or one directory) is supported for now"
        return input_dataset.pop(), cnt
    
    def set_score_fn(self, field: str, score_fn: callable):
        """
        Set the score function for the validation data.
        Creates the field entry if it doesn't exist.
        """
        for _, annotations in self.expected_output.items():
            for _, record_stats in annotations.items():
                if "score_fn" not in record_stats:
                    record_stats["score_fn"] = {}
                record_stats["score_fn"][field] = score_fn

    def set_field_annotation(self, logical_op_num: int, record_source_idx: int, field: str, label: str, pass_filter: bool = True):
        """
        Set the annotation for a given field. If the logical_op_num or record_source_idx don't exist,
        initialize new entries in the expected_output dictionary.
        """
        logical_op_id = None
        for idx, (op_id, _) in enumerate(self.expected_output.items()):
            if idx == logical_op_num:
                logical_op_id = op_id
                break
            if op_id == logical_op_num:
                logical_op_id = op_id
                break
        if logical_op_id is None:
            self.expected_output[logical_op_num] = {}
            logical_op_id = logical_op_num

        # Initialize the record source entry if it doesn't exist
        if record_source_idx not in self.expected_output[logical_op_id]:
            self.expected_output[logical_op_id][record_source_idx] = {
                "labels": {},
                "pass_filter": pass_filter,
                "score_fn": {field: "exact"}
            }
        # Set the annotation
        self.expected_output[logical_op_id][record_source_idx]["labels"][field] = label


    def num_samples(self) -> int:
        return self._num_samples
    
    def input_dataset(self) -> str:
        return self._input_dataset
    
    def expected_outputs(self):
        """
        Returns the expected outputs for the given logical operator ID.
        Format:
        {
            logical_op_id: {
                record_source_idx: {
                    "labels": {field1: value1, ...},
                    "score_fn": {field1: "exact", ...}
                },
                ...
            },
            ...
        }
        """
        return self.expected_output
    
    def to_json(self):
        """
        Convert the ValidationData object to a JSON-serializable format.
        Converts callable score_fn to function names for serialization.
        """
        json_output = {}
        for logical_op_id, op_stats in self.expected_output.items():
            json_output[logical_op_id] = {}
            for record_idx, record_stats in op_stats.items():
                json_output[logical_op_id][record_idx] = {
                    "labels": record_stats["labels"],
                    "pass_filter": record_stats["pass_filter"],
                    "score_fn": {
                        field: fn.__name__ if callable(fn) else fn 
                        for field, fn in record_stats["score_fn"].items()
                    }
                }
        return json_output
