from __future__ import annotations

import json
import math
import os

# NOTE: the answer.mode() call(s) inside of _est_quality() throw a UserWarning when there are multiple
#       answers to a convert with the same mode. This is because pandas tries to sort the answers
#       before returning them, but since answer is a column of dicts the '<' operator fails on dicts.
#       For now, we can simply ignore the warning b/c we pick an answer at random anyways if there are
#       multiple w/the same count, but in the future we may want to cast the 'dict' --> 'str' or compute
#       the mode on a per-field basis.
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from tqdm import tqdm

import palimpzest as pz
from palimpzest.constants import MODEL_CARDS, GPT_4o_MODEL_CARD, Model
from palimpzest.dataclasses import OperatorCostEstimates, PlanCost, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements.records import DataRecordSet
from palimpzest.operators.convert import ConvertOp, LLMConvert
from palimpzest.operators.datasource import CacheScanDataOp, DataSourcePhysicalOp, MarshalAndScanDataOp
from palimpzest.operators.filter import FilterOp, LLMFilter
from palimpzest.operators.physical import PhysicalOperator
from palimpzest.operators.retrieve import RetrieveOp
from palimpzest.optimizer.plan import SentinelPlan
from palimpzest.utils.model_helpers import get_champion_model_name, get_models

warnings.simplefilter(action='ignore', category=UserWarning)

class BaseCostModel:
    """
    This base class contains the interface/abstraction that every CostModel must implement
    in order to work with the Optimizer. In brief, the Optimizer expects the CostModel to
    make a prediction about the runtime, cost, and quality of a physical operator.
    """
    def __init__(self):
        """
        CostModel constructor; the arguments for individual CostModels may vary depending
        on the assumptions they make about the prevalance of historical execution data
        and online vs. batch execution settings.
        """
        pass

    def get_costed_phys_op_ids(self) -> set[str]:
        """
        Return the set of physical op ids which the cost model has cost estimates for.
        """
        raise NotImplementedError("Calling get_costed_phys_op_ids from abstract method")

    def __call__(self, operator: PhysicalOperator) -> PlanCost:
        """
        The interface exposed by the CostModel to the Optimizer. Subclasses may require
        additional arguments in order to make their predictions.
        """
        raise NotImplementedError("Calling __call__ from abstract method")


class SampleBasedCostModel:
    """
    """
    def __init__(
        self,
        sentinel_plan: SentinelPlan,
        execution_data: dict[str, dict[str, list[DataRecordSet]]],
        verbose: bool = False,
        exp_name: str | None = None,
    ):
        # TODO: remove this
        self.sentinel_plan = sentinel_plan

        # store verbose argument
        self.verbose = verbose

        # store experiment name if one is provided
        self.exp_name = exp_name

        # construct cost, time, quality, and selectivity matrices for each operator set;
        self.operator_to_stats = self.compute_operator_stats(execution_data, sentinel_plan.operator_sets)

        # compute set of costed physical op ids from operator_to_stats
        self.costed_phys_op_ids = set([
            phys_op_id
            for _, phys_op_id_to_stats in self.operator_to_stats.items()
            for phys_op_id, _ in phys_op_id_to_stats.items()
        ])

        # reference to data directory
        self.datadir = DataDirectory()

        # import pdb; pdb.set_trace()

    def get_costed_phys_op_ids(self):
        return self.costed_phys_op_ids


    def compute_operator_stats(
            self,
            execution_data: dict[str, dict[str, list[DataRecordSet]]],
            operator_sets: list[list[PhysicalOperator]],
        ):
        # flatten the nested dictionary of execution data and pull out fields relevant to cost estimation
        execution_record_op_stats = []
        for idx, op_set in enumerate(operator_sets):
            # initialize variables
            logical_op_id = op_set[0].logical_op_id
            upstream_op_set_id = SentinelPlan.compute_op_set_id(operator_sets[idx - 1]) if idx > 0 else None
            op_set_id = SentinelPlan.compute_op_set_id(op_set)

            # filter for the execution data from this operator set
            op_set_execution_data = execution_data[op_set_id]

            # flatten the execution data into a list of RecordOpStats
            op_set_execution_data = [
                record_op_stats
                for _, record_sets in op_set_execution_data.items()
                for record_set in record_sets
                for record_op_stats in record_set.record_op_stats
            ]

            # add entries from execution data into matrices
            for record_op_stats in op_set_execution_data:
                record_op_stats_dict = {
                    "logical_op_id": logical_op_id,
                    "op_set_id": op_set_id,
                    "physical_op_id": record_op_stats.op_id,
                    "upstream_op_set_id": upstream_op_set_id,
                    "record_id": record_op_stats.record_id,
                    "record_parent_id": record_op_stats.record_parent_id,
                    "cost_per_record": record_op_stats.cost_per_record,
                    "time_per_record": record_op_stats.time_per_record,
                    "quality": record_op_stats.quality,
                    "passed_operator": record_op_stats.passed_operator,
                    "source_id": record_op_stats.record_source_id,  # TODO: remove
                    "op_details": record_op_stats.op_details,       # TODO: remove
                    "answer": record_op_stats.answer,               # TODO: remove
                }
                execution_record_op_stats.append(record_op_stats_dict)

        # convert flattened execution data into dataframe
        operator_stats_df = pd.DataFrame(execution_record_op_stats)

        # for each physical_op_id, compute its average cost_per_record, time_per_record, selectivity, and quality
        operator_to_stats = {}
        for logical_op_id, logical_op_df in operator_stats_df.groupby("logical_op_id"):
            operator_to_stats[logical_op_id] = {}

            # get the op_set_id of the upstream operator
            upstream_op_set_ids = logical_op_df.upstream_op_set_id.unique()
            assert len(upstream_op_set_ids) == 1, "More than one upstream op id"
            upstream_op_set_id = upstream_op_set_ids[0]

            for physical_op_id, physical_op_df in logical_op_df.groupby("physical_op_id"):
                # find set of parent records for this operator
                num_upstream_records = len(physical_op_df.record_parent_id.unique())

                # compute selectivity 
                selectivity = (
                    1.0 if upstream_op_set_id is None else physical_op_df.passed_operator.sum() / num_upstream_records
                )

                operator_to_stats[logical_op_id][physical_op_id] = {
                    "cost": physical_op_df.cost_per_record.mean(),
                    "time": physical_op_df.time_per_record.mean(),
                    "quality": physical_op_df.quality.mean(),
                    "selectivity": selectivity,
                }

        # if this is an experiment, log the dataframe and operator_to_stats dictionary
        if self.exp_name is not None:
            operator_stats_df.to_csv(f"opt-profiling-data/{self.exp_name}-operator-stats.csv", index=False)
            with open(f"opt-profiling-data/{self.exp_name}-operator-stats.json", "w") as f:
                json.dump(operator_to_stats, f)

        return operator_to_stats


    def __call__(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
        # NOTE: some physical operators may not have any sample execution data in this cost model;
        #       these physical operators are filtered out of the Optimizer, thus we can assume that
        #       we will have execution data for each operator passed into __call__; nevertheless, we
        #       still perform a sanity check
        # look up logical op id and matrix column associated with this physical operator
        phys_op_id = operator.get_op_id()
        logical_op_id = operator.logical_op_id
        assert self.operator_to_stats.get(logical_op_id).get(phys_op_id) is not None, f"No execution data for {str(operator)}"

        # look up stats for this operation
        est_cost_per_record = self.operator_to_stats[logical_op_id][phys_op_id]["cost"]
        est_time_per_record = self.operator_to_stats[logical_op_id][phys_op_id]["time"]
        est_quality = self.operator_to_stats[logical_op_id][phys_op_id]["quality"]
        est_selectivity = self.operator_to_stats[logical_op_id][phys_op_id]["selectivity"]

        # create source_op_estimates for datasources if they are not provided
        if isinstance(operator, DataSourcePhysicalOp):
            # get handle to DataSource and pre-compute its size (number of records)
            datasource = self.datadir.get_registered_dataset(operator.dataset_id)
            datasource_len = len(datasource)

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

        # generate new set of OperatorCostEstimates
        op_estimates = OperatorCostEstimates(
            cardinality=est_selectivity * source_op_estimates.cardinality,
            time_per_record=est_time_per_record,
            cost_per_record=est_cost_per_record,
            quality=est_quality,
        )

        # compute estimates for this operator
        op_time = op_estimates.time_per_record * source_op_estimates.cardinality
        op_cost = op_estimates.cost_per_record * source_op_estimates.cardinality
        op_quality = op_estimates.quality

        # construct and return op estimates
        return PlanCost(cost=op_cost, time=op_time, quality=op_quality, op_estimates=op_estimates)


class MatrixCompletionCostModel:
    """
    TODO: evaluate on some common benchmark datasets, e.g.:
    - Fever dataset
    - Wiki-QA might also be good choices in the future
    """
    def __init__(
            self,
            sentinel_plan: SentinelPlan,
            rank: int,
            execution_data: dict[str, dict[str, list[DataRecordSet]]],
            champion_outputs: dict[str, dict[str, DataRecordSet]],
            expected_outputs: dict[str, DataRecordSet] | None = None,
            field_to_metric_fn: dict[str, str | Callable] | None = None,
            num_epochs: int = 5000,
            verbose: bool = False,
        ):
        """
        TODO: update to match changes
        execution_data:
            List of RecordOpStats containing execution data from every execution of an operator
            on a record during sentinel execution

        champion_outputs:
            A mapping from op_set_id --> list of DataRecords for every operator set in our sentinel_plan.
            The list of DataRecords is the set of champion / ensemble model outputs for that operator.

        expected_outputs:
            An optional list of DataRecords which contains the user-provided validation data.
            Is None if no validation data is provided. If present, it will be a list of DataRecords
            representing the exact output of the (end-to-end) PZ program expected by the user.

        sentinel_plan:
            The sentinel plan which was used to generate the sample execution data.
        """
        # TODO: remove this
        self.sentinel_plan = sentinel_plan

        # set the rank for the low-rank completion
        self.rank = rank

        # set the number of iterations for matrix completion algorithm
        self.num_epochs = num_epochs

        # store verbose argument
        self.verbose = verbose

        # compute the quality of each record as judged by the validation exemplars (or champion model)
        execution_data = self.score_quality(sentinel_plan.operator_sets, execution_data, champion_outputs, expected_outputs, field_to_metric_fn)

        # compute mapping from op_set_id --> logical_op_id
        op_set_id_to_logical_id = {
            SentinelPlan.compute_op_set_id(op_set): op_set[0].logical_op_id
            for op_set in sentinel_plan.operator_sets
        }

        # compute mapping from logical_op_id --> sample mask
        self.logical_op_id_to_sample_masks = {
            op_set_id_to_logical_id[op_set_id]: (sample_matrix, record_to_row_map, phys_op_to_col_map)
            for op_set_id, (sample_matrix, record_to_row_map, phys_op_to_col_map) in sentinel_plan.sample_matrices.items()
        }

        # construct cost, time, quality, and selectivity matrices for each operator set;
        self.logical_op_id_to_raw_matrices = self.construct_matrices(
            execution_data,
            sentinel_plan.operator_sets,
            self.logical_op_id_to_sample_masks,
        )

        # TODO: remove after SIGMOD
        if os.environ['LOG_MATRICES'].lower() == "true":
            with open(f"opt-profiling-data/sentinel-plan-n-{self.rank + 1}-{sentinel_plan.plan_id}.json", "w") as f:
                json.dump({"plan_str": str(sentinel_plan)}, f)

            with open(f"opt-profiling-data/sample-masks-n-{self.rank + 1}-{sentinel_plan.plan_id}.json", "w") as f:
                logical_op_id_to_list_sample_masks = {}
                for logical_op_id, (sample_matrix, _, _) in self.logical_op_id_to_sample_masks.items():
                    logical_op_id_to_list_sample_masks[logical_op_id] = sample_matrix.tolist()
                json.dump(logical_op_id_to_list_sample_masks, f)

            with open(f"opt-profiling-data/raw-matrices-n-{self.rank + 1}-{sentinel_plan.plan_id}.json", "w") as f:
                logical_op_id_to_list_raw_matrices = {}
                for logical_op_id, raw_matrix_dict in self.logical_op_id_to_raw_matrices.items():
                    logical_op_id_to_list_raw_matrices[logical_op_id] = {
                        "cost": raw_matrix_dict["cost"].tolist(),
                        "time": raw_matrix_dict["time"].tolist(),
                        "selectivity": raw_matrix_dict["selectivity"].tolist(),
                        "quality": raw_matrix_dict["quality"].tolist(),
                    }
                json.dump(logical_op_id_to_list_raw_matrices, f)

        # complete the observation matrices
        self.logical_op_id_to_matrices = self.complete_matrices(
            self.logical_op_id_to_raw_matrices,
            self.logical_op_id_to_sample_masks,
        )

        # construct mapping from each logical operator id to its previous logical operator id
        self.logical_op_id_to_prev_logical_op_id = {}
        for idx, op_set in enumerate(sentinel_plan.operator_sets):
            logical_op_id = op_set[0].logical_op_id
            if idx == 0:
                self.logical_op_id_to_prev_logical_op_id[logical_op_id] = None
            else:
                prev_op_set = sentinel_plan.operator_sets[idx - 1]
                prev_logical_op_id = prev_op_set[0].logical_op_id
                self.logical_op_id_to_prev_logical_op_id[logical_op_id] = prev_logical_op_id

        # compute a dictionary mapping from physical operator id --> (logical_op_id, col)
        self.physical_op_id_to_matrix_col = {}
        for logical_op_id, (_, _, phys_op_to_col_map) in self.logical_op_id_to_sample_masks.items():
            for phys_op_id, col in phys_op_to_col_map.items():
                self.physical_op_id_to_matrix_col[phys_op_id] = (logical_op_id, col)

        # compute set of costed physical op ids from operator_to_stats
        self.costed_phys_op_ids = set(self.physical_op_id_to_matrix_col.keys())

        # reference to data directory
        self.datadir = DataDirectory()

        # import pdb; pdb.set_trace()

    def get_costed_phys_op_ids(self):
        return self.costed_phys_op_ids

    def compute_quality(
            self,
            record_set: DataRecordSet,
            expected_record_set: DataRecordSet | None = None,
            champion_record_set: DataRecordSet | None = None,
            is_filter_op: bool = False,
            is_convert_op: bool = False,
            field_to_metric_fn: dict[str, str | Callable] | None = None,
        ) -> DataRecordSet:
        """
        Compute the quality for the given `record_set` by comparing it to the `expected_record_set`.

        Update the record_set by assigning the quality to each entry in its record_op_stats and
        returning the updated record_set.
        """
        # compute whether we can only use the champion
        only_using_champion = expected_record_set is None

        # if this operation is a failed convert
        if is_convert_op and len(record_set) == 0:
            record_set.record_op_stats[0].quality = 0.0

        # if this operation is a filter:
        # - we assign a quality of 1.0 if the record is in the expected outputs and it passes this filter
        # - we assign a quality of 0.0 if the record is in the expected outputs and it does NOT pass this filter
        # - we assign a quality relative to the champion / ensemble output if the record is not in the expected outputs
        # we cannot know for certain what the correct behavior is a given filter on a record which is not in the output
        # (unless it is the only filter in the plan), thus we only evaluate the filter based on its performance on
        # records which are in the output
        elif is_filter_op:
            # NOTE:
            # - we know that record_set.record_op_stats will contain a single entry for a filter op
            # - if we are using the champion, then champion_record_set will also contain a single entry for a filter op
            record_op_stats = record_set.record_op_stats[0]
            if only_using_champion:
                champion_record = champion_record_set[0]
                record_op_stats.quality = int(record_op_stats.passed_operator == champion_record._passed_operator)

            # - if we are using validation data, we may have multiple expected records in the expected_record_set for this source_id,
            #   thus, if we can identify an exact match, we can use that to evaluate the filter's quality
            # - if we are using validation data but we *cannot* find an exact match, then we will once again use the champion record set
            else:
                # compute number of matches between this record's computed fields and this expected record's outputs
                found_match_in_output = False
                for expected_record in expected_record_set:
                    all_correct = True
                    for field, value in record_op_stats.record_state.items():
                        if value != getattr(expected_record, field):
                            all_correct = False
                            break

                    if all_correct:
                        found_match_in_output = True
                        break

                if found_match_in_output:
                    record_op_stats.quality = int(record_op_stats.passed_operator == expected_record._passed_operator)
                else:
                    champion_record = champion_record_set[0]
                    record_op_stats.quality = int(record_op_stats.passed_operator == champion_record._passed_operator)

        # if this is a succesful convert operation
        else:
            # NOTE: the following computation assumes we do not project out computed values
            #       (and that the validation examples provide all computed fields); even if
            #       a user program does add projection, we can ignore the projection on the
            #       validation dataset and use the champion model (as opposed to the validation
            #       output) for scoring fields which have their values projected out

            # set the expected_record_set to be the champion_record_set if we do not have validation data
            expected_record_set = champion_record_set if only_using_champion else expected_record_set

            # GREEDY ALGORITHM
            # for each record in the expected output, we look for the computed record which maximizes the quality metric;
            # once we've identified that computed record we remove it from consideration for the next expected output
            for expected_record in expected_record_set:
                best_quality, best_record_op_stats = 0.0, None
                for record_op_stats in record_set.record_op_stats:
                    # if we already assigned this record a quality, skip it
                    if record_op_stats.quality is not None:
                        continue

                    # compute number of matches between this record's computed fields and this expected record's outputs
                    total_quality = 0
                    for field in record_op_stats.generated_fields:
                        computed_value = record_op_stats.record_state.get(field, None)
                        expected_value = getattr(expected_record, field)

                        # get the metric function for this field
                        metric_fn = (
                            field_to_metric_fn[field]
                            if field_to_metric_fn is not None and field in field_to_metric_fn
                            else "exact"
                        )

                        # compute exact match
                        if metric_fn == "exact":
                            total_quality += int(computed_value == expected_value)

                        # compute UDF metric
                        elif callable(metric_fn):
                            total_quality += metric_fn(computed_value, expected_value)

                        # otherwise, throw an exception
                        else:
                            raise Exception(f"Unrecognized metric_fn: {metric_fn}")

                    # compute recall and update best seen so far
                    quality = total_quality / len(record_op_stats.generated_fields)
                    if quality > best_quality:
                        best_quality = quality
                        best_record_op_stats = record_op_stats

                # set best_quality as quality for the best_record_op_stats
                if best_record_op_stats is not None:
                    best_record_op_stats.quality = best_quality

        # for any records which did not receive a quality, set it to 0.0 as these are unexpected extras
        for record_op_stats in record_set.record_op_stats:
            if record_op_stats.quality is None:
                record_op_stats.quality = 0.0

        return record_set


    def score_quality(
            self,
            operator_sets: list[list[PhysicalOperator]],
            execution_data: dict[str, dict[str, list[DataRecordSet]]],
            champion_outputs: dict[str, dict[str, DataRecordSet]],
            expected_outputs: dict[str, DataRecordSet] = None,
            field_to_metric_fn: dict[str, str | Callable] = None,
        ) -> list[RecordOpStats]:
        """
        NOTE: This approach to cost modeling does not work directly for aggregation queries;
              for these queries, we would ask the user to provide validation data for the step immediately
              before a final aggregation

        NOTE: This function currently assumes that one-to-many converts do NOT create duplicate outputs.
        This assumption would break if, for example, we extracted the breed of every dog in an image.
        If there were two golden retrievers and a bernoodle in an image and we extracted:

            {"image": "file1.png", "breed": "Golden Retriever"}
            {"image": "file1.png", "breed": "Golden Retriever"}
            {"image": "file1.png", "breed": "Bernedoodle"}
        
        This function would currently give perfect accuracy to the following output:

            {"image": "file1.png", "breed": "Golden Retriever"}
            {"image": "file1.png", "breed": "Bernedoodle"}

        Even though it is missing one of the golden retrievers.
        """
        # extract information about the logical operation performed at this stage of the sentinel plan;
        # NOTE: we can infer these fields from context clues, but in the long-term we should have a more
        #       principled way of getting these directly from attributes either stored in the sentinel_plan
        #       or in the PhysicalOperator
        op_set = operator_sets[-1]
        op_set_id = SentinelPlan.compute_op_set_id(op_set)
        physical_op = op_set[0]
        is_source_op = isinstance(physical_op, (MarshalAndScanDataOp, CacheScanDataOp))
        is_filter_op = isinstance(physical_op, FilterOp)
        is_convert_op = isinstance(physical_op, ConvertOp)
        is_perfect_quality_op = (
            not isinstance(physical_op, LLMConvert)
            and not isinstance(physical_op, LLMFilter)
            and not isinstance(physical_op, RetrieveOp)
        )

        # pull out the execution data from this operator; place the upstream execution data in a new list
        this_op_execution_data = execution_data[op_set_id]

        # compute quality of each output computed by this operator
        for source_id, record_sets in this_op_execution_data.items():
            # NOTE
            # source_id is a particular input, for which we may have computed multiple output record_sets;
            # each of these record_sets may contain more than one record (b/c one-to-many) and we have one
            # record_set per operator in the op_set

            # if this operation does not involve an LLM, every record_op_stats object gets perfect quality
            if is_perfect_quality_op:
                for record_set in record_sets:
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = 1.0
                continue

            # get the expected output for this source_id if we have one
            expected_record_set = (
                expected_outputs[source_id]
                if expected_outputs is not None and source_id in expected_outputs
                else None
            )

            # extract champion output for this record set
            champion_record_set = champion_outputs[op_set_id][source_id]

            # for each record_set produced by an operation, compute its quality
            for record_set in record_sets:
                record_set = self.compute_quality(record_set, expected_record_set, champion_record_set, is_filter_op, is_convert_op, field_to_metric_fn)

        # if this operator is a source op (i.e. has no input logical operator), return the execution data
        if is_source_op:
            return execution_data

        # recursively call the function on the next logical operator until you reach a scan
        execution_data = self.score_quality(operator_sets[:-1], execution_data, champion_outputs, expected_outputs, field_to_metric_fn)

        # return the quality annotated record op stats
        return execution_data


    def construct_matrices(
            self,
            execution_data: dict[str, dict[str, list[DataRecordSet]]],
            operator_sets: list[list[PhysicalOperator]],
            logical_op_id_to_sample_masks: dict[str, tuple[np.array, dict[str, int], dict[str, int]]],
        ):
        # given a set of execution data, construct the mappings
        # from logical_op_id --> cost, runtime, selectivity, and quality matrices
        logical_op_id_to_matrices = {}
        for op_set in operator_sets:
            # filter for the execution data from this operator set
            logical_op_id = op_set[0].logical_op_id
            op_set_id = SentinelPlan.compute_op_set_id(op_set)
            op_set_execution_data = execution_data[op_set_id]

            # flatten the execution data into a list of RecordOpStats
            op_set_execution_data = [
                record_op_stats
                for _, record_sets in op_set_execution_data.items()
                for record_set in record_sets
                for record_op_stats in record_set.record_op_stats
            ]

            # fetch the mappings from records and physical operators to their location(s) in the sample mask
            sample_mask, record_to_row_map, phys_op_to_col_map = logical_op_id_to_sample_masks[logical_op_id]

            # initialize matrices
            cost_arr = np.zeros(sample_mask.shape)
            time_arr = np.zeros(sample_mask.shape)
            sel_arr = np.zeros(sample_mask.shape)
            quality_arr = np.zeros(sample_mask.shape)

            # add entries from execution data into matrices
            for record_op_stats in op_set_execution_data:
                # NOTE: record_op_stats contains info from the *output record(s)* that result from
                # applying this operator (set) to an input record; thus, the corresponding sample_matrix
                # row is determined by the record_op_stats' parent_record_id
                parent_record_id = (
                    record_op_stats.record_parent_id
                    if record_op_stats.record_parent_id is not None
                    else record_op_stats.record_source_id
                )

                # NOTE: we sum the cost_per_record and time_per_record, because for one-to-many outputs we
                # will have multiple record_op_stats objects for a given input record
                row = record_to_row_map[parent_record_id]
                col = phys_op_to_col_map[record_op_stats.op_id]
                cost_arr[row, col] += record_op_stats.cost_per_record
                time_arr[row, col] += record_op_stats.time_per_record

                # NOTE: for filter operations the selectivity of this operator on the input record is determined
                # by whether or not it passed the filter; for convert operations, we increment by 1 for every output
                # which had this input record as its parent
                if record_op_stats.passed_operator is not None:
                    sel_arr[row, col] = int(record_op_stats.passed_operator)
                else:
                    sel_arr[row, col] += 1

                # NOTE: for quality, we will sum the quality of each output record for the given input record;
                # we will then divide by sel_arr[row, col] outside of the for loop to compute the average output quality
                quality_arr[row, col] += record_op_stats.quality

            # if this operation was a convert operation, compute the average quality for each sample
            # by dividing by the number of outputs which were generated for this sample (i.e. sel_arr[row, col])
            quality_arr = np.where(sel_arr == 0, quality_arr, quality_arr / sel_arr)

            # set matrices
            logical_op_id_to_matrices[logical_op_id] = {
                "cost": cost_arr,
                "time": time_arr,
                "selectivity": sel_arr,
                "quality": quality_arr,
            }

        return logical_op_id_to_matrices

    def _learn_factor_matrices(self, true_mat, mat_mask):
        device = "cpu"
        num_rows, num_cols = true_mat.shape
        mat_mask = mat_mask.astype(bool)

        x = torch.empty((num_rows, self.rank), requires_grad=True)
        torch.nn.init.normal_(x)
        y = torch.empty((self.rank, num_cols), requires_grad=True)
        torch.nn.init.normal_(y)

        mse_loss = torch.nn.MSELoss()
        opt_x = torch.optim.Adam([x], lr=1e-3, weight_decay=1e-5)
        opt_y = torch.optim.Adam([y], lr=1e-3, weight_decay=1e-5)

        true_matrix = torch.as_tensor(true_mat, dtype=torch.float32, device=device)

        for _ in range(self.num_epochs):
            opt_x.zero_grad()
            opt_y.zero_grad()

            loss = mse_loss(torch.matmul(x, y)[mat_mask], true_matrix[mat_mask])

            loss.backward()

            opt_x.step()
            opt_y.step()

            with torch.no_grad():
                x[:] = x.clamp_(min=0)
                y[:] = y.clamp_(min=0)

        return [x, y]

    def _complete_matrix(self, matrix: np.array, sample_mask: np.array) -> np.array:
        x, y = self._learn_factor_matrices(matrix, sample_mask)
        return torch.matmul(x, y).detach().numpy()

    def complete_matrices(
            self,
            logical_op_id_to_matrices: dict[str, dict[str, np.array]],
            logical_op_id_to_sample_masks: dict[str, np.array],
        ):

        # for each logical operator (or op_set)...
        print("Completing Matrices for Logical Ops")
        logical_op_id_to_completed_matrices = {}
        for logical_op_id, matrices_dict in tqdm(logical_op_id_to_matrices.items()):
            # and for each metric of interest...
            logical_op_id_to_completed_matrices[logical_op_id] = {}
            sample_mask, _, _ = logical_op_id_to_sample_masks[logical_op_id]

            # if sample_mask is all 1's, no need to complete the matrix (it already is complete)
            if (sample_mask == 1.0).all():
                logical_op_id_to_completed_matrices[logical_op_id] = matrices_dict
                continue

            # otherwise, complete the matrix for each metric of interest
            for metric, matrix in tqdm(matrices_dict.items(), leave=False):
                # ensure that matrix has float dtype (not int)
                matrix = matrix.astype(float)

                # complete the matrix
                completed_matrix = self._complete_matrix(matrix, sample_mask)

                # only update matrix entries for which we did not have samples
                matrix[~sample_mask.astype(bool)] = completed_matrix[~sample_mask.astype(bool)]

                # clamp all matrices to be non-negative
                matrix = np.clip(matrix, 0.0, None)

                # clamp quality matrix to be less than 1.0
                if metric == "quality":
                    matrix = np.clip(matrix, 0.0, 1.0)

                # set the matrix to be the completed matrix
                logical_op_id_to_completed_matrices[logical_op_id][metric] = matrix

        return logical_op_id_to_completed_matrices

    def __call__(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
        # look up logical op id and matrix column associated with this physical operator
        phys_op_id = operator.get_op_id()
        logical_op_id, col = self.physical_op_id_to_matrix_col[phys_op_id]

        # look up column data for this operation
        cost_col_data = self.logical_op_id_to_matrices[logical_op_id]["cost"][:, col]
        time_col_data = self.logical_op_id_to_matrices[logical_op_id]["time"][:, col]
        selectivity_col_data = self.logical_op_id_to_matrices[logical_op_id]["selectivity"][:, col]
        quality_col_data = self.logical_op_id_to_matrices[logical_op_id]["quality"][:, col]

        # if this is a filter operation, estimate selectivity using the column mean;
        # otherwise, estimate fan-out using ratio between the number of rows between this matrix and its input
        est_selectivity = None
        if isinstance(operator, FilterOp):
            est_selectivity = np.mean(selectivity_col_data)
        else:
            prev_logical_op_id = self.logical_op_id_to_prev_logical_op_id[logical_op_id]
            if prev_logical_op_id is not None:
                prev_selectivity_col_data = self.logical_op_id_to_matrices[logical_op_id]["selectivity"]
                est_selectivity = len(selectivity_col_data)/len(prev_selectivity_col_data)
            else:
                est_selectivity = 1.0

        # create source_op_estimates for datasources if they are not provided
        if isinstance(operator, DataSourcePhysicalOp):
            # get handle to DataSource and pre-compute its size (number of records)
            datasource = self.datadir.get_registered_dataset(operator.dataset_id)
            datasource_len = len(datasource)

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

        # estimate cost, time, and quality using column means
        est_cost_per_record = np.mean(cost_col_data)
        est_time_per_record = np.mean(time_col_data)
        est_quality = np.mean(quality_col_data)
        est_cardinality = est_selectivity * source_op_estimates.cardinality

        # generate new set of OperatorCostEstimates
        op_estimates = OperatorCostEstimates(
            cardinality=est_cardinality,
            time_per_record=est_time_per_record,
            cost_per_record=est_cost_per_record,
            quality=est_quality,
        )

        # compute estimates for this operator
        op_time = op_estimates.time_per_record * source_op_estimates.cardinality
        op_cost = op_estimates.cost_per_record * source_op_estimates.cardinality
        op_quality = op_estimates.quality

        # construct and return op estimates
        return PlanCost(cost=op_cost, time=op_time, quality=op_quality, op_estimates=op_estimates)


class CostModel(BaseCostModel):
    """
    This class takes in a list of RecordOpStats and performs cost estimation on a given operator
    by taking the average of any sample execution that the CostModel has for that operator. If no
    such data exists, it returns a naive estimate.
    """
    def __init__(
        self,
        sample_execution_data: list[RecordOpStats] | None = None,
        available_models: list[Model] | None = None,
        confidence_level: float = 0.90,
    ) -> None:
        if sample_execution_data is None:
            sample_execution_data = []
        if available_models is None:
            available_models = []

        # construct full dataset of samples
        self.sample_execution_data_df = (
            pd.DataFrame(sample_execution_data)
            if len(sample_execution_data) > 0
            else None
        )
        # df contains a column called record_state, that sometimes contain a dict
        # we want to extract the keys from the dict and create a new column for each key

        # reference to data directory
        self.datadir = DataDirectory()

        # set available models
        self.available_models = available_models

        # set confidence level for CI estimates
        self.conf_level = confidence_level

        # compute per-operator estimates
        self.operator_estimates = self._compute_operator_estimates()

        # compute set of costed physical op ids from operator_to_stats
        self.costed_phys_op_ids = None if self.operator_estimates is None else set(self.operator_estimates.keys())

    def get_costed_phys_op_ids(self):
        return self.costed_phys_op_ids

    def _compute_ci(self, sample_mean: float, n_samples: int, std_dev: float) -> tuple[float, float]:
        """
        Compute confidence interval (for non-proportion quantities) given the sample mean, number of samples,
        and sample std. deviation at the CostModel's given confidence level. We use a t-distribution for
        computing the interval as many sample estimates in PZ may have few samples.
        """
        ci = stats.t.interval(
            confidence=self.conf_level,  # Confidence level
            df=n_samples - 1,            # Degrees of freedom
            loc=sample_mean,             # Sample mean
            scale=std_dev,               # Standard deviation estimate
        )
        return ci

    def _compute_proportion_ci(self, sample_prop: float, n_samples: int) -> tuple[float, float]:
        """
        Compute confidence interval for proportion quantities (i.e. selectivity) given the sample proportion
        and the number of samples. We use the normal distribution for computing the interval here, for reasons
        summarized by this post: https://stats.stackexchange.com/a/411727.
        """
        if sample_prop == 0.0 or sample_prop == 1.0:
            return (sample_prop, sample_prop)

        scaling_factor = math.sqrt((sample_prop * (1 - sample_prop)) / n_samples)
        lower_bound, upper_bound = stats.norm.interval(
            confidence=self.conf_level,  # Confidence level
            loc=sample_prop,             # Sample proportion
            scale=scaling_factor,        # Scaling factor
        )
        lower_bound = max(lower_bound, 0.0)
        upper_bound = max(upper_bound, 1.0)

        return (lower_bound, upper_bound)

    def _compute_mean_and_ci(self, df: pd.DataFrame, col: str, model_name: str | None = None, non_negative_lb: bool = False) -> tuple[float, float, float]:
        """
        Compute the mean and CI for the given column and dataframe. If the model_name is provided, filter
        for the subset of rows belonging to the model.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = df[df.model_name == model_name]
            if not model_df.empty:
                col_mean = model_df[col].mean()
                col_lb, col_ub = self._compute_ci(
                    sample_mean=col_mean,
                    n_samples=model_df[col].notna().sum(),
                    std_dev=model_df[col].std(),
                )
                if non_negative_lb:
                    col_lb = max(col_lb, 0.0)

                return col_mean, col_lb, col_ub

        # compute aggregate
        col_mean = df[col].mean()
        col_lb, col_ub = self._compute_ci(
            sample_mean=col_mean,
            n_samples=df[col].notna().sum(),
            std_dev=df[col].std(),
        )
        if non_negative_lb:
            col_lb = max(col_lb, 0.0)

        return col_mean, col_lb, col_ub

    def _est_time_per_record(self, op_df: pd.DataFrame, model_name: str | None = None) -> tuple[float, float, float]:
        """
        Given sample cost data observations for a specific operation, compute the mean and CI
        for the time per record.
        """
        return self._compute_mean_and_ci(df=op_df, col="time_per_record", model_name=model_name, non_negative_lb=True)

    def _est_cost_per_record(self, op_df: pd.DataFrame, model_name: str | None = None) -> tuple[float, float, float]:
        """
        Given sample cost data observations for a specific operation, compute the mean and CI
        for the cost per record.
        """
        return self._compute_mean_and_ci(df=op_df, col="cost_per_record", model_name=model_name, non_negative_lb=True)

    def _est_tokens_per_record(self, op_df: pd.DataFrame, model_name: str | None = None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Given sample cost data observations for a specific operation, compute the mean and CI
        for the total input tokens and total output tokens.
        """
        total_input_tokens_tuple = self._compute_mean_and_ci(df=op_df, col="total_input_tokens", model_name=model_name, non_negative_lb=True)
        total_output_tokens_tuple = self._compute_mean_and_ci(df=op_df, col="total_output_tokens", model_name=model_name, non_negative_lb=True)

        return total_input_tokens_tuple, total_output_tokens_tuple

    def _est_cardinality(self, op_df: pd.DataFrame, model_name: str | None = None) -> float:
        """
        Given sample cost data observations for a specific operation, compute the number of
        rows output by the operation.

        NOTE: right now, this should only be used by the ApplyGroupByOp as a way to gauge the
        number of output groups. Using this to estimate, e.g. the cardinality of a filter,
        convert, or base scan will lead to wildly inaccurate results because the absolute value
        of these cardinalities will simply be a reflection of the sample size.

        For those operations, we use the `_est_selectivity` function to estimate the operator's
        selectivity, which we can apply to an est. of the operator's input cardinality.
        """
        return op_df.shape[0] / len(op_df.plan_id.unique())

    def _est_selectivity(self, df: pd.DataFrame, op_df: pd.DataFrame, model_name: str | None = None) -> float:
        """
        Given sample cost data observations for the plan and a specific operation, compute
        the ratio of records between this operator and its source operator.
        """
        # compute whether or not this operation is a filter
        is_filter_op = "filter" in str(op_df.op_name.iloc[0]).lower()

        # use model-specific estimate if possible
        if model_name is not None:
            model_op_df = op_df[op_df.model_name == model_name]
            if not model_op_df.empty:
                num_input_records = model_op_df.shape[0]

                # get subset of records that were the source to this operator
                num_output_records = None
                if is_filter_op:
                    num_output_records = model_op_df.passed_operator.sum()
                else:
                    op_ids = model_op_df.op_id.unique().tolist()
                    plan_ids = model_op_df.plan_id.unique().tolist()
                    num_output_records = df[df.source_op_id.isin(op_ids) & df.plan_id.isin(plan_ids)].shape[0]

                # estimate the selectivity / fan-out and compute bounds
                est_selectivity = num_output_records / num_input_records
                if is_filter_op:
                    est_selectivity_lb, est_selectivity_ub = self._compute_proportion_ci(est_selectivity, n_samples=num_input_records)

                # for now, if we are doing a convert operation w/fan-out then the assumptions of _compute_proportion_ci
                # do not hold; until we have a better method for estimating bounds, just set them to the estimate
                else:
                    est_selectivity_lb = est_selectivity
                    est_selectivity_ub = est_selectivity

                return est_selectivity, est_selectivity_lb, est_selectivity_ub

        # otherwise average selectivity across all ops
        num_input_records = op_df.shape[0]

        # get subset of records that were the source to this operator
        num_output_records = None
        if is_filter_op:
            num_output_records = op_df.passed_operator.sum()
        else:
            op_ids = op_df.op_id.unique().tolist()
            num_output_records = df[df.source_op_id.isin(op_ids)].shape[0]

        # estimate the selectivity / fan-out and compute bounds
        est_selectivity = num_output_records / num_input_records
        if is_filter_op:
            est_selectivity_lb, est_selectivity_ub = self._compute_proportion_ci(est_selectivity, n_samples=num_input_records)

        # for now, if we are doing a convert operation w/fan-out then the assumptions of _compute_proportion_ci
        # do not hold; until we have a better method for estimating bounds, just set them to the estimate
        else:
            est_selectivity_lb = est_selectivity
            est_selectivity_ub = est_selectivity

        return est_selectivity, est_selectivity_lb, est_selectivity_ub

    def _compute_quality(self, row):
        # compute accuracy for filter
        if "filter" in row["op_name"].lower():
            row["correct"] = int(row["answer"] == row["accepted_answer"])
            row["num_answers"] = 1
            return row

        # otherwise, compute recall on a per-key basis
        try:
            # we'll measure recall on accepted_answer, as extraneous info is often not an issue
            answer = row["answer"]
            accepted_answer = row["accepted_answer"]
            correct = 0
            for key, value in accepted_answer.items():
                if key in answer and answer[key] == value:
                    correct += 1
            
            row["correct"] = correct
            row["num_answers"] = len(accepted_answer.keys())
            return row

        except Exception as e:
            print(f"WARNING: error decoding answer or accepted_answer: {str(e)}")
            row["correct"] = 0
            row["num_answers"] = 1
            return row

    def _est_quality(self, op_df: pd.DataFrame, model_name: str | None = None) -> float:
        """
        Given sample cost data observations for a specific operation, compute the an estimate
        of the quality of its outputs by using GPT-4 as a champion model.
        """
        # get unique set of records
        record_ids = op_df.record_id.unique()

        # get champion model name
        vision = ("image_operation" in op_df.columns and op_df.image_operation.any())
        champion_model_name = get_champion_model_name(self.available_models, vision)

        # compute champion's answer (per-record) across all models; fall-back to most common answer if champion is not present
        record_id_to_answer = {}
        for record_id in record_ids:
            record_df = op_df[op_df.record_id == record_id]
            champion_most_common_answer = record_df[
                record_df.model_name == champion_model_name
            ].answer.mode()
            all_models_most_common_answer = record_df.answer.mode()

            if not champion_most_common_answer.empty:
                record_id_to_answer[record_id] = champion_most_common_answer.iloc[0]
            elif not all_models_most_common_answer.empty:
                record_id_to_answer[record_id] = all_models_most_common_answer.iloc[0]
            else:
                record_id_to_answer[record_id] = None

        # compute accepted answers and clean all answers
        pd.options.mode.chained_assignment = None  # turn off copy warnings
        op_df.loc[:, "accepted_answer"] = op_df.record_id.apply(lambda id: record_id_to_answer[id])
        op_df = op_df.apply(lambda row: self._compute_quality(row), axis=1)

        # get subset of observations for model_name and estimate quality w/fraction of answers that match accepted answer
        model_df = (
            op_df[op_df.model_name == model_name]
            if model_name is not None
            else op_df[op_df.model_name.isna()]
        )

        # compute quality as the fraction of answers which are correct (recall on expected output)
        num_correct = model_df.correct.sum() if not model_df.empty else op_df.correct.sum()
        total_answers = model_df.num_answers.sum() if not model_df.empty else op_df.num_answers.sum()
        est_quality = num_correct / total_answers

        # compute CI on the proportion of correct answers
        est_quality_lb, est_quality_ub = self._compute_proportion_ci(est_quality, n_samples=total_answers)

        return est_quality, est_quality_lb, est_quality_ub

    def _compute_operator_estimates(self) -> dict[str, Any] | None:
        """
        Compute per-operator estimates of runtime, cost, and quality.
        """
        # if we don't have sample execution data, we cannot compute per-operator estimates
        if self.sample_execution_data_df is None:
            return None

        # get the set of operator ids for which we have sample data
        op_ids = self.sample_execution_data_df.op_id.unique()

        # compute estimates of runtime, cost, and quality (and intermediates like cardinality) for every operator
        operator_estimates = {}
        for op_id in op_ids:
            # filter for subset of sample execution data related to this operation
            op_df = self.sample_execution_data_df[
                self.sample_execution_data_df.op_id == op_id
            ]

            # skip computing an estimate if we didn't capture any sampling data for this operator
            # (this can happen if/when upstream filter operation(s) filter out all records) 
            if op_df.empty:
                continue

            # initialize estimates
            estimates = {}

            # get the op_name for this operation
            model_name = op_df.model_name.iloc[0] if op_df.model_name.iloc[0] is not None else None
            op_name = str(op_df.op_name.iloc[0])
            if model_name is not None:
                # compute estimates per-model, and add None which forces computation of avg. across all models
                model_names = [m.value for m in get_models(include_vision=True)] + [None]
                # model_names = op_df.model_name.unique().tolist()
                estimates = {model_name: None for model_name in model_names}
                for model_name in model_names:
                    time_per_record, time_per_record_lb, time_per_record_ub = self._est_time_per_record(op_df, model_name=model_name)
                    cost_per_record, cost_per_record_lb, cost_per_record_ub = self._est_cost_per_record(op_df, model_name=model_name)
                    input_tokens_tup, output_tokens_tup = self._est_tokens_per_record(op_df, model_name=model_name)
                    selectivity, selectivity_lb, selectivity_ub = self._est_selectivity(self.sample_execution_data_df, op_df, model_name=model_name)
                    quality, quality_lb, quality_ub = self._est_quality(op_df, model_name=model_name)
                    
                    model_estimates = {
                        "time_per_record": time_per_record,
                        "time_per_record_lower_bound": time_per_record_lb,
                        "time_per_record_upper_bound": time_per_record_ub,
                        "cost_per_record": cost_per_record,
                        "cost_per_record_lower_bound": cost_per_record_lb,
                        "cost_per_record_upper_bound": cost_per_record_ub,
                        "total_input_tokens": input_tokens_tup[0],
                        "total_input_tokens_lower_bound": input_tokens_tup[1],
                        "total_input_tokens_upper_bound": input_tokens_tup[2],
                        "total_output_tokens": output_tokens_tup[0],
                        "total_output_tokens_lower_bound": output_tokens_tup[1],
                        "total_output_tokens_upper_bound": output_tokens_tup[2],
                        "selectivity": selectivity,
                        "selectivity_lower_bound": selectivity_lb,
                        "selectivity_upper_bound": selectivity_ub,
                        "quality": quality,
                        "quality_lower_bound": quality_lb,
                        "quality_upper_bound": quality_ub,
                    }
                    estimates[model_name] = model_estimates

            # TODO pre-compute lists of op_names in groups
            elif op_name in ["NonLLMFilter"]:
                time_per_record, time_per_record_lb, time_per_record_ub = self._est_time_per_record(op_df)
                selectivity, selectivity_lb, selectivity_ub = self._est_selectivity(self.sample_execution_data_df, op_df)
                estimates = {
                    "time_per_record": time_per_record,
                    "time_per_record_lower_bound": time_per_record_lb,
                    "time_per_record_upper_bound": time_per_record_ub,
                    "selectivity": selectivity,
                    "selectivity_lower_bound": selectivity_lb,
                    "selectivity_upper_bound": selectivity_ub,
                }

            elif op_name in ["MarshalAndScanDataOp", "CacheScanDataOp", "LimitScanOp", "CountAggregateOp", "AverageAggregateOp"]:
                time_per_record, time_per_record_lb, time_per_record_ub = self._est_time_per_record(op_df)
                estimates = {
                    "time_per_record": time_per_record,
                    "time_per_record_lower_bound": time_per_record_lb,
                    "time_per_record_upper_bound": time_per_record_ub,
                }

            elif op_name in ["ApplyGroupByOp"]:
                time_per_record, time_per_record_lb, time_per_record_ub = self._est_time_per_record(op_df)
                cardinality = self._est_cardinality(op_df)
                estimates = {
                    "time_per_record": time_per_record,
                    "time_per_record_lower_bound": time_per_record_lb,
                    "time_per_record_upper_bound": time_per_record_ub,
                    "cardinality": cardinality,
                }

            operator_estimates[op_id] = estimates

        return operator_estimates

    def __call__(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
        # get identifier for operation which is unique within sentinel plan but consistent across sentinels
        op_id = operator.get_op_id()

        # initialize estimates of operator metrics based on naive (but sometimes precise) logic
        if isinstance(operator, pz.MarshalAndScanDataOp):
            # get handle to DataSource and pre-compute its size (number of records)
            datasource = self.datadir.get_registered_dataset(operator.dataset_id)
            dataset_type = self.datadir.get_registered_dataset_type(operator.dataset_id)
            datasource_len = len(datasource)
            datasource_memsize = datasource.get_size()

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

            op_estimates = operator.naiveCostEstimates(source_op_estimates,
                                                    input_record_size_in_bytes=datasource_memsize/datasource_len,
                                                    dataset_type=dataset_type)

        elif isinstance(operator, pz.CacheScanDataOp):
            datasource = self.datadir.get_cached_result(operator.dataset_id)
            datasource_len = len(datasource)
            datasource_memsize = datasource.get_size()

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

            op_estimates = operator.naiveCostEstimates(source_op_estimates, input_record_size_in_bytes=datasource_memsize/datasource_len)

        else:
            op_estimates = operator.naiveCostEstimates(source_op_estimates)

        # if we have sample execution data, update naive estimates with more informed ones
        sample_op_estimates = self.operator_estimates
        if sample_op_estimates is not None and op_id in sample_op_estimates:
            if isinstance(operator, (pz.MarshalAndScanDataOp, pz.CacheScanDataOp)):
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, pz.ApplyGroupByOp):
                # NOTE: in theory we should also treat this cardinality est. as a random variable, but in practice we will
                #       have K samples of the number of groups produced by the groupby operator, where K is the number of
                #       plans we generate sample data with. For now, we will simply use the estimate without bounds.
                #
                # NOTE: this cardinality is the only cardinality we estimate directly b/c we can observe how many groups are
                #       produced by the groupby in our sample and assume it may generalize to the full workload. To estimate
                #       actual cardinalities of operators we estimate their selectivities / fan-outs and multiply those by
                #       the input cardinality (where the initial input cardinality from the datasource is known).
                op_estimates.cardinality = sample_op_estimates[op_id]["cardinality"]
                op_estimates.cardinality_lower_bound = op_estimates.cardinality
                op_estimates.cardinality_upper_bound = op_estimates.cardinality
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, (pz.CountAggregateOp, pz.AverageAggregateOp)):  # noqa: SIM114
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, pz.LimitScanOp):
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, pz.NonLLMFilter):
                op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id]["selectivity"]
                op_estimates.cardinality_lower_bound = source_op_estimates.cardinality_lower_bound * sample_op_estimates[op_id]["selectivity_lower_bound"]
                op_estimates.cardinality_upper_bound = source_op_estimates.cardinality_upper_bound * sample_op_estimates[op_id]["selectivity_upper_bound"]

                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, pz.LLMFilter):
                model_name = operator.model.value
                op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                op_estimates.cardinality_lower_bound = source_op_estimates.cardinality_lower_bound * sample_op_estimates[op_id][model_name]["selectivity_lower_bound"]
                op_estimates.cardinality_upper_bound = source_op_estimates.cardinality_upper_bound * sample_op_estimates[op_id][model_name]["selectivity_upper_bound"]

                op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id][model_name]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id][model_name]["time_per_record_upper_bound"]

                op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                op_estimates.cost_per_record_lower_bound = sample_op_estimates[op_id][model_name]["cost_per_record_lower_bound"]
                op_estimates.cost_per_record_upper_bound = sample_op_estimates[op_id][model_name]["cost_per_record_upper_bound"]

                op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]
                op_estimates.quality_lower_bound = sample_op_estimates[op_id][model_name]["quality_lower_bound"]
                op_estimates.quality_upper_bound = sample_op_estimates[op_id][model_name]["quality_upper_bound"]

            elif isinstance(operator, pz.LLMConvert):
                # TODO: EVEN BETTER: do similarity match (e.g. largest param intersection, more exotic techniques);
                #       another heuristic: logical_op_id-->subclass_physical_op_id-->specific_physical_op_id-->most_param_match_physical_op_id
                # TODO: instead of [op_id][model_name] --> [logical_op_id][physical_op_id]
                # NOTE: code synthesis does not have a model attribute
                model_name = operator.model.value if hasattr(operator, "model") else None
                op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                op_estimates.cardinality_lower_bound = source_op_estimates.cardinality_lower_bound * sample_op_estimates[op_id][model_name]["selectivity_lower_bound"]
                op_estimates.cardinality_upper_bound = source_op_estimates.cardinality_upper_bound * sample_op_estimates[op_id][model_name]["selectivity_upper_bound"]

                op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id][model_name]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id][model_name]["time_per_record_upper_bound"]

                op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                op_estimates.cost_per_record_lower_bound = sample_op_estimates[op_id][model_name]["cost_per_record_lower_bound"]
                op_estimates.cost_per_record_upper_bound = sample_op_estimates[op_id][model_name]["cost_per_record_upper_bound"]

                op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]
                op_estimates.quality_lower_bound = sample_op_estimates[op_id][model_name]["quality_lower_bound"]
                op_estimates.quality_upper_bound = sample_op_estimates[op_id][model_name]["quality_upper_bound"]

                # NOTE: if code synth. fails, this will turn into ConventionalQuery calls to GPT-3.5,
                #       which would wildly mess up estimate of time and cost per-record
                # do code synthesis adjustment
                if isinstance(operator, pz.CodeSynthesisConvert):
                    op_estimates.time_per_record = 1e-5
                    op_estimates.time_per_record_lower_bound = op_estimates.time_per_record
                    op_estimates.time_per_record_upper_bound = op_estimates.time_per_record
                    op_estimates.cost_per_record = 1e-4
                    op_estimates.cost_per_record_lower_bound = op_estimates.cost_per_record
                    op_estimates.cost_per_record_upper_bound = op_estimates.cost_per_record
                    op_estimates.quality = op_estimates.quality * (GPT_4o_MODEL_CARD["code"] / 100.0)
                    op_estimates.quality_lower_bound = op_estimates.quality_lower_bound * (GPT_4o_MODEL_CARD["code"] / 100.0)
                    op_estimates.quality_upper_bound = op_estimates.quality_upper_bound * (GPT_4o_MODEL_CARD["code"] / 100.0)

                # token reduction adjustment
                if isinstance(operator, pz.TokenReducedConvert):
                    total_input_tokens = operator.token_budget * sample_op_estimates[op_id][model_name]["total_input_tokens"]
                    total_output_tokens = sample_op_estimates[op_id][model_name]["total_output_tokens"]
                    op_estimates.cost_per_record = (
                        MODEL_CARDS[model_name]["usd_per_input_token"] * total_input_tokens
                        + MODEL_CARDS[model_name]["usd_per_output_token"] * total_output_tokens
                    )
                    total_input_tokens_lb = operator.token_budget * sample_op_estimates[op_id][model_name]["total_input_tokens_lower_bound"]
                    total_output_tokens_lb = sample_op_estimates[op_id][model_name]["total_output_tokens_lower_bound"]
                    op_estimates.cost_per_record_lower_bound = (
                        MODEL_CARDS[model_name]["usd_per_input_token"] * total_input_tokens_lb
                        + MODEL_CARDS[model_name]["usd_per_output_token"] * total_output_tokens_lb
                    )
                    total_input_tokens_ub = operator.token_budget * sample_op_estimates[op_id][model_name]["total_input_tokens_upper_bound"]
                    total_output_tokens_ub = sample_op_estimates[op_id][model_name]["total_output_tokens_upper_bound"]
                    op_estimates.cost_per_record_upper_bound = (
                        MODEL_CARDS[model_name]["usd_per_input_token"] * total_input_tokens_ub
                        + MODEL_CARDS[model_name]["usd_per_output_token"] * total_output_tokens_ub
                    )

                    op_estimates.quality = op_estimates.quality * math.sqrt(math.sqrt(operator.token_budget))
                    op_estimates.quality_lower_bound = op_estimates.quality_lower_bound * math.sqrt(math.sqrt(operator.token_budget))
                    op_estimates.quality_upper_bound = op_estimates.quality_upper_bound * math.sqrt(math.sqrt(operator.token_budget))

            else:
                raise Exception("Unknown operator")

        # compute estimates for this operator
        op_time = op_estimates.time_per_record * source_op_estimates.cardinality
        op_cost = op_estimates.cost_per_record * source_op_estimates.cardinality
        op_quality = op_estimates.quality

        # compute bounds on total time and cost estimates for this operator
        op_cost_lower_bound = op_estimates.cost_per_record_lower_bound * source_op_estimates.cardinality_lower_bound
        op_cost_upper_bound = op_estimates.cost_per_record_upper_bound * source_op_estimates.cardinality_upper_bound
        op_time_lower_bound = op_estimates.time_per_record_lower_bound * source_op_estimates.cardinality_lower_bound
        op_time_upper_bound = op_estimates.time_per_record_upper_bound * source_op_estimates.cardinality_upper_bound
        op_quality_lower_bound = op_estimates.quality_lower_bound
        op_quality_upper_bound = op_estimates.quality_upper_bound

        # create and return PlanCost object for this op's statistics
        op_plan_cost = PlanCost(
            cost=op_cost,
            time=op_time,
            quality=op_quality,
            op_estimates=op_estimates,
            cost_lower_bound=op_cost_lower_bound,
            cost_upper_bound=op_cost_upper_bound,
            time_lower_bound=op_time_lower_bound,
            time_upper_bound=op_time_upper_bound,
            quality_lower_bound=op_quality_lower_bound,
            quality_upper_bound=op_quality_upper_bound,
        )

        return op_plan_cost
