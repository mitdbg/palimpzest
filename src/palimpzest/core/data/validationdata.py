import json
import os
import logging

logger = logging.getLogger(__name__)

class ValidationData:
    def __init__(self, file_path: str):
        """
        Initialize ValidationData with a directory path containing execution_stats.updated.json.
        Args:
            file_path: Path to the execution stats file
        """
        self.file_path = file_path
        self.expected_output = {}
        self._input_dataset = None
        
        # Read execution stats JSON
        with open(self.file_path) as f:
            execution_stats = json.load(f)

        if "plan_stats" in execution_stats:
            _, working_stats = execution_stats["plan_stats"].popitem()
        else:
            working_stats = execution_stats

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
                    labels_filtered = record_stats["annotations"]["labels_filtered"]
                else:
                    labels = record_stats["record_state"]
                    labels_filtered = record_stats["passed_operator"]

                # Initialize nested dictionaries if they don't exist
                if logical_op_id not in expected_outputs:
                    expected_outputs[logical_op_id] = {}

                score_fns = {}
                if labels is not None:
                    score_fns = {field: "exact" for field in labels}
                # Store the record state
                expected_outputs[logical_op_id][record_source_idx] = {
                    "labels": labels,
                    "labels_filtered": labels_filtered,
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
        """
        set_score_fn = False
        for _, op_stats in self.expected_output.items():
            for _, record_stats in op_stats.items():
                if field in record_stats["score_fn"]:
                    record_stats["score_fn"][field] = score_fn
                    logger.info(f"Set score function for {field} to {score_fn}")
                    set_score_fn = True
        
        if not set_score_fn:
            logger.warning(f"No field {field} found in expected outputs")

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
