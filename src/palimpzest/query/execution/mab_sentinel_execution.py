import time
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from typing import Callable

import numpy as np

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS, OptimizationStrategy
from palimpzest.core.data.dataclasses import ExecutionStats, OperatorStats, PlanStats, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import SourceRecord
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_engine import ExecutionEngine
from palimpzest.query.execution.plan_executors.parallel_plan_execution import PipelinedParallelPlanExecutor
from palimpzest.query.execution.plan_executors.single_threaded_plan_execution import SequentialSingleThreadPlanExecutor
from palimpzest.query.operators.convert import ConvertOp, LLMConvert
from palimpzest.query.operators.datasource import CacheScanDataOp, MarshalAndScanDataOp
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.optimizer.cost_model import CostModel, SampleBasedCostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.sets import Set


class MABSentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute_plan() method.
    """
    def __init__(
            self,
            k: int,
            j: int,
            sample_budget: int,
            early_stop_iters: int = 3,
            use_final_op_quality: bool = False,
            seed: int = 42,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # self.max_workers = self.get_parallel_max_workers()
        # TODO: undo
        # self.max_workers = 4
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.early_stop_iters = early_stop_iters
        self.use_final_op_quality = use_final_op_quality
        self.pick_output_fn = self.pick_ensemble_output
        self.rng = np.random.default_rng(seed=seed)

    def update_frontier_ops(
        self,
        frontier_ops,
        reservoir_ops,
        policy,
        all_outputs,
        logical_op_id_to_num_samples,
        phys_op_id_to_num_samples,
        is_filter_op_dict,
    ):
        """
        Update the set of frontier operators, pulling in new ones from the reservoir as needed.
        This function will (for each op_set):
        1. Compute the mean, LCB, and UCB for the cost, time, quality, and selectivity of each operator
        2. Compute the pareto optimal set of operators (using the mean values)
        3. Update the frontier and reservoir sets of operators based on their LCB/UCB overlap with the pareto frontier
        """
        # compute metrics for each operator in all_outputs
        logical_op_id_to_op_metrics = {}
        for logical_op_id, source_id_to_record_sets in all_outputs.items():
            # compute selectivity for each physical operator
            phys_op_to_num_inputs, phys_op_to_num_outputs = {}, {}
            for _, record_sets in source_id_to_record_sets.items():
                for record_set in record_sets:
                    op_id = record_set.record_op_stats[0].op_id
                    num_outputs = sum([record_op_stats.passed_operator for record_op_stats in record_set.record_op_stats])
                    if op_id not in phys_op_to_num_inputs:
                        phys_op_to_num_inputs[op_id] = 1
                        phys_op_to_num_outputs[op_id] = num_outputs
                    else:
                        phys_op_to_num_inputs[op_id] += 1
                        phys_op_to_num_outputs[op_id] += num_outputs

            phys_op_to_mean_selectivity = {
                op_id: phys_op_to_num_outputs[op_id] / phys_op_to_num_inputs[op_id]
                for op_id in phys_op_to_num_inputs
            }

            # compute average cost, time, and quality
            phys_op_to_costs, phys_op_to_times, phys_op_to_qualities = {}, {}, {}
            for _, record_sets in source_id_to_record_sets.items():
                for record_set in record_sets:
                    for record_op_stats in record_set.record_op_stats:
                        op_id = record_op_stats.op_id
                        cost = record_op_stats.cost_per_record
                        time = record_op_stats.time_per_record
                        quality = record_op_stats.quality
                        if op_id not in phys_op_to_costs:
                            phys_op_to_costs[op_id] = [cost]
                            phys_op_to_times[op_id] = [time]
                            phys_op_to_qualities[op_id] = [quality]
                        else:
                            phys_op_to_costs[op_id].append(cost)
                            phys_op_to_times[op_id].append(time)
                            phys_op_to_qualities[op_id].append(quality)

            phys_op_to_mean_cost = {op: np.mean(costs) for op, costs in phys_op_to_costs.items()}
            phys_op_to_mean_time = {op: np.mean(times) for op, times in phys_op_to_times.items()}
            phys_op_to_mean_quality = {op: np.mean(qualities) for op, qualities in phys_op_to_qualities.items()}

            # compute average, LCB, and UCB of each operator; the confidence bounds depend upon
            # the computation of the alpha parameter, which we scale to be 0.5 * the mean (of means)
            # of the metric across all operators in this operator set
            cost_alpha = 0.5 * np.mean([mean_cost for mean_cost in phys_op_to_mean_cost.values()])
            time_alpha = 0.5 * np.mean([mean_time for mean_time in phys_op_to_mean_time.values()])
            quality_alpha = 0.5 * np.mean([mean_quality for mean_quality in phys_op_to_mean_quality.values()])
            selectivity_alpha = 0.5 * np.mean([mean_selectivity for mean_selectivity in phys_op_to_mean_selectivity.values()])
 
            op_metrics = {}
            for op_id in phys_op_to_costs:
                sample_ratio = np.sqrt(np.log(logical_op_id_to_num_samples[logical_op_id]) / phys_op_id_to_num_samples[op_id])
                exploration_terms = np.array([cost_alpha * sample_ratio, time_alpha * sample_ratio, quality_alpha * sample_ratio, selectivity_alpha * sample_ratio])
                mean_terms = (phys_op_to_mean_cost[op_id], phys_op_to_mean_time[op_id], phys_op_to_mean_quality[op_id], phys_op_to_mean_selectivity[op_id])

                # NOTE: we could clip these; however I will not do so for now to allow for arbitrary quality metric(s)
                lcb_terms = mean_terms - exploration_terms
                ucb_terms = mean_terms + exploration_terms
                op_metrics[op_id] = {"mean": mean_terms, "lcb": lcb_terms, "ucb": ucb_terms}

            # store average metrics for each operator in the op_set
            logical_op_id_to_op_metrics[logical_op_id] = op_metrics

        # get the tuple representation of this policy
        policy_dict = policy.get_dict()

        # compute the pareto optimal set of operators for each logical_op_id
        pareto_op_sets = {}
        for logical_op_id, op_metrics in logical_op_id_to_op_metrics.items():
            pareto_op_sets[logical_op_id] = set()
            for op_id, metrics in op_metrics.items():
                cost, time, quality, selectivity = metrics["mean"]
                pareto_frontier = True

                # check if any other operator dominates op_id
                for other_op_id, other_metrics in op_metrics.items():
                    other_cost, other_time, other_quality, other_selectivity = other_metrics["mean"]
                    if op_id == other_op_id:
                        continue

                    # if op_id is dominated by other_op_id, set pareto_frontier = False and break
                    # NOTE: here we use a strict inequality (instead of the usual <= or >=) because
                    #       all ops which have equal cost / time / quality / sel. should not be
                    #       filtered out from sampling by our logic in this function
                    cost_dominated = True if policy_dict["cost"] == 0.0 else other_cost < cost
                    time_dominated = True if policy_dict["time"] == 0.0 else other_time < time
                    quality_dominated = True if policy_dict["quality"] == 0.0 else other_quality > quality
                    selectivity_dominated = True if not is_filter_op_dict[logical_op_id] else other_selectivity < selectivity
                    if cost_dominated and time_dominated and quality_dominated and selectivity_dominated:
                        pareto_frontier = False
                        break

                # add op_id to pareto frontier if it's not dominated
                if pareto_frontier:
                    pareto_op_sets[logical_op_id].add(op_id)

        # iterate over frontier ops and replace any which do not overlap with pareto frontier
        new_frontier_ops = {logical_op_id: [] for logical_op_id in frontier_ops}
        new_reservoir_ops = {logical_op_id: [] for logical_op_id in reservoir_ops}
        for logical_op_id, pareto_op_set in pareto_op_sets.items():
            num_dropped_from_frontier = 0
            for op, next_shuffled_sample_idx, new_operator, fully_sampled in frontier_ops[logical_op_id]:
                op_id = op.get_op_id()

                # if this op is fully sampled, remove it from the frontier
                if fully_sampled:
                    num_dropped_from_frontier += 1
                    continue

                # if this op is pareto optimal keep it in our frontier ops
                if op_id in pareto_op_set:
                    new_frontier_ops[logical_op_id].append((op, next_shuffled_sample_idx, new_operator, fully_sampled))
                    continue

                # otherwise, if this op overlaps with an op on the pareto frontier, keep it in our frontier ops
                # NOTE: for now, we perform an optimistic comparison with the ucb/lcb
                pareto_frontier = True
                op_cost = logical_op_id_to_op_metrics[logical_op_id][op_id]["lcb"][0]
                op_time = logical_op_id_to_op_metrics[logical_op_id][op_id]["lcb"][1]
                op_quality = logical_op_id_to_op_metrics[logical_op_id][op_id]["ucb"][2]
                op_selectivity = logical_op_id_to_op_metrics[logical_op_id][op_id]["lcb"][3]
                for pareto_op_id in pareto_op_set:
                    pareto_cost = logical_op_id_to_op_metrics[logical_op_id][pareto_op_id]["ucb"][0]
                    pareto_time = logical_op_id_to_op_metrics[logical_op_id][pareto_op_id]["ucb"][1]
                    pareto_quality = logical_op_id_to_op_metrics[logical_op_id][pareto_op_id]["lcb"][2]
                    pareto_selectivity = logical_op_id_to_op_metrics[logical_op_id][pareto_op_id]["ucb"][3]

                    # if op_id is dominated by pareto_op_id, set pareto_frontier = False and break
                    cost_dominated = True if policy_dict["cost"] == 0.0 else pareto_cost <= op_cost
                    time_dominated = True if policy_dict["time"] == 0.0 else pareto_time <= op_time
                    quality_dominated = True if policy_dict["quality"] == 0.0 else pareto_quality >= op_quality
                    selectivity_dominated = True if not is_filter_op_dict[logical_op_id] else pareto_selectivity <= op_selectivity
                    if cost_dominated and time_dominated and quality_dominated and selectivity_dominated:
                        pareto_frontier = False
                        break
                
                # add op_id to pareto frontier if it's not dominated
                if pareto_frontier:
                    new_frontier_ops[logical_op_id].append((op, next_shuffled_sample_idx, new_operator, fully_sampled))
                else:
                    num_dropped_from_frontier += 1

            # replace the ops dropped from the frontier with new ops from the reservoir
            num_dropped_from_frontier = min(num_dropped_from_frontier, len(reservoir_ops[logical_op_id]))
            for idx in range(num_dropped_from_frontier):
                new_frontier_ops[logical_op_id].append((reservoir_ops[logical_op_id][idx], 0, True, False))

            # update reservoir ops for this logical_op_id
            new_reservoir_ops[logical_op_id] = reservoir_ops[logical_op_id][num_dropped_from_frontier:]

        return new_frontier_ops, new_reservoir_ops


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
                record_op_stats.quality = int(record_op_stats.passed_operator == champion_record.passed_operator)

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
                    record_op_stats.quality = int(record_op_stats.passed_operator == expected_record.passed_operator)
                else:
                    champion_record = champion_record_set[0]
                    record_op_stats.quality = int(record_op_stats.passed_operator == champion_record.passed_operator)

        # if this is a successful convert operation
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
            op_set: list[PhysicalOperator],
            logical_op_id: str,
            execution_data: dict[str, dict[str, list[DataRecordSet]]],
            champion_outputs: dict[str, dict[str, DataRecordSet]],
            expected_outputs: dict[str, DataRecordSet] | None = None,
            field_to_metric_fn: dict[str, str | Callable] | None = None,
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
        physical_op = op_set[0]
        is_filter_op = isinstance(physical_op, FilterOp)
        is_convert_op = isinstance(physical_op, ConvertOp)
        is_perfect_quality_op = (
            not isinstance(physical_op, LLMConvert)
            and not isinstance(physical_op, LLMFilter)
            and not isinstance(physical_op, RetrieveOp)
        )

        # pull out the execution data from this operator; place the upstream execution data in a new list
        this_op_execution_data = execution_data[logical_op_id]

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
            champion_record_set = champion_outputs[logical_op_id][source_id]

            # for each record_set produced by an operation, compute its quality
            for record_set in record_sets:
                record_set = self.compute_quality(record_set, expected_record_set, champion_record_set, is_filter_op, is_convert_op, field_to_metric_fn)

        # return the quality annotated record op stats
        return execution_data


    def pick_ensemble_output(self, op_set_record_sets: list[tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # NOTE: I don't like that this assumes the models are consistent in
        #       how they order their record outputs for one-to-many converts;
        #       eventually we can try out more robust schemes to account for
        #       differences in ordering
        # aggregate records at each index in the response
        idx_to_records = {}
        for record_set, _ in op_set_record_sets:
            for idx, record in enumerate(record_set):
                if idx not in idx_to_records:
                    idx_to_records[idx] = [record]
                else:
                    idx_to_records[idx].append(record)

        # compute most common answer at each index
        out_records = []
        for idx in range(len(idx_to_records)):
            records = idx_to_records[idx]
            most_common_record = max(set(records), key=records.count)
            out_records.append(most_common_record)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, [])


    def pick_highest_quality_output(self, op_set_record_sets: list[tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # aggregate records at each index in the response
        idx_to_records = {}
        for record_set, _ in op_set_record_sets:
            for idx in range(len(record_set)):
                record, record_op_stats = record_set[idx], record_set.record_op_stats[idx]
                if idx not in idx_to_records:
                    idx_to_records[idx] = [(record, record_op_stats)]
                else:
                    idx_to_records[idx].append((record, record_op_stats))

        # compute highest quality answer at each index
        out_records = []
        for idx in range(len(idx_to_records)):
            records_lst, record_op_stats_lst = zip(*idx_to_records[idx])
            max_quality_record, max_quality = records_lst[0], record_op_stats_lst[0].quality
            for record, record_op_stats in zip(records_lst[1:], record_op_stats_lst[1:]):
                record_quality = record_op_stats.quality
                if record_quality > max_quality:
                    max_quality_record = record
                    max_quality = record_quality
            out_records.append(max_quality_record)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, [])


    @staticmethod
    def execute_op_wrapper(operator: PhysicalOperator, op_input: DataRecord | list[DataRecord]) -> tuple[DataRecordSet, PhysicalOperator]:
        """
        Wrapper function around operator execution which also and returns the operator.
        This is useful in the parallel setting(s) where operators are executed by a worker pool,
        and it is convenient to return the op_id along with the computation result.
        """
        record_set = operator(op_input)

        return record_set, operator, op_input


    def execute_op_set(self, op_candidate_pairs):
        # TODO: post-submission we will need to modify this to:
        # - submit all candidates for aggregate operators
        # - handle limits
        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = []
            for operator, candidate in op_candidate_pairs:
                future = executor.submit(MABSentinelExecutionEngine.execute_op_wrapper, operator, candidate)
                futures.append(future)

            # compute output record_set for each (operator, candidate) pair
            output_record_sets = []
            while len(futures) > 0:
                # get the set of futures that have (and have not) finished in the last PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

                # cast not_done_futures from a set to a list so we can append to it
                not_done_futures = list(not_done_futures)

                # process finished futures
                for future in done_futures:
                    # get the result and add it to the output records set
                    record_set, operator, candidate = future.result()
                    output_record_sets.append((record_set, operator, candidate))

                # update list of futures
                futures = not_done_futures

            # compute mapping from source_id to record sets for all operators and for champion operator
            all_record_sets, champion_record_sets = {}, {}
            for op, candidate in op_candidate_pairs:
                candidate_output_record_sets, source_id = [], None
                for record_set, operator, candidate_ in output_record_sets:
                    if candidate == candidate_:
                        candidate_output_record_sets.append((record_set, operator))

                        # NOTE: we should resolve this issue in a more thoughtful way, but currently the source_id for
                        #       scan candidate records is the sample_idx, when we want it to be the source_id which is
                        #       set by the scan operator when it returns the record
                        # get the source_id associated with this input record
                        source_id = (
                            candidate.source_id
                            if not (isinstance(op, (MarshalAndScanDataOp, CacheScanDataOp)))
                            else record_set[0].source_id
                        )

                # select the champion (i.e. best) record_set from all the record sets computed for this candidate
                champion_record_set = self.pick_output_fn(candidate_output_record_sets)

                # add champion record_set to mapping from source_id --> champion record_set
                champion_record_sets[source_id] = champion_record_set

                # add all record_sets computed for this source_id to mapping from source_id --> record_sets
                all_record_sets[source_id] = [tup[0] for tup in candidate_output_record_sets]

        return all_record_sets, champion_record_sets


    def execute_sentinel_plan(self, plan: SentinelPlan, expected_outputs: dict[str, DataRecordSet], policy: Policy):
        """
        """
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (sentinel):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # initialize plan stats and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for logical_op_id, logical_op_name, op_set in plan:
            op_set_details = {
                op.op_name(): {k: str(v) for k, v in op.get_id_params().items()}
                for op in op_set
            }
            plan_stats.operator_stats[logical_op_id] = OperatorStats(
                op_id=logical_op_id,
                op_name=logical_op_name,
                op_details=op_set_details,
            )

        # shuffle the indices of records to sample
        total_num_samples = self.datasource.get_val_length()
        shuffled_sample_indices = [int(idx) for idx in np.arange(total_num_samples)]
        self.rng.shuffle(shuffled_sample_indices)

        # sample k initial operators for each operator set; for each operator maintain a tuple of:
        # (operator, next_shuffled_sample_idx, new_operator); new_operator is True when an operator
        # is added to the frontier
        frontier_ops, reservoir_ops = {}, {}
        for logical_op_id, _, op_set in plan:
            op_set_copy = [op for op in op_set]
            self.rng.shuffle(op_set_copy)
            k = min(self.k, len(op_set_copy))
            frontier_ops[logical_op_id] = [(op, 0, True, False) for op in op_set_copy[:k]]
            reservoir_ops[logical_op_id] = [op for op in op_set_copy[k:]]

        # create mapping from logical and physical op ids to the number of samples drawn
        logical_op_id_to_num_samples = {logical_op_id: 0 for logical_op_id, _, _ in plan}
        phys_op_id_to_num_samples = {op.get_op_id(): 0 for _, _, op_set in plan for op in op_set}
        is_filter_op_dict = {
            logical_op_id: isinstance(op_set[0], FilterOp)
            for logical_op_id, _, op_set in plan
        }

        # TODO: long-term, we should do something which does not rely on scanning validation source to build this mapping
        sample_idx_to_source_id = {
            sample_idx: self.datasource.get_item(sample_idx, val=True).source_id
            for sample_idx in range(total_num_samples)
        }

        # NOTE: to maintain parity with our count of samples drawn in the random sampling execution,
        # for each logical_op_id, we count the number of (record, op) executions as the number of samples within that op_set;
        # the samples drawn is equal to the max of that number across all operator sets
        samples_drawn = 0
        all_outputs, champion_outputs = {}, {}
        while samples_drawn < self.sample_budget:
            # execute operator sets in sequence
            for op_idx, (logical_op_id, _, op_set) in enumerate(plan):
                prev_logical_op_id = plan.logical_op_ids[op_idx - 1] if op_idx > 0 else None
                prev_logical_op_is_filter =  prev_logical_op_id is not None and is_filter_op_dict[prev_logical_op_id]

                # create list of tuples for (op, candidate) which we should execute
                op_candidate_pairs = []
                updated_frontier_ops_lst = []
                for op, next_shuffled_sample_idx, new_operator, fully_sampled in frontier_ops[logical_op_id]:
                    # execute new operators on first j candidates, and previously sampled operators on one additional candidate
                    j = min(self.j, len(shuffled_sample_indices)) if new_operator else 1
                    for j_idx in range(j):
                        candidates = []
                        if isinstance(op, (MarshalAndScanDataOp, CacheScanDataOp)):
                            sample_idx = shuffled_sample_indices[(next_shuffled_sample_idx + j_idx) % len(shuffled_sample_indices)]
                            candidate = DataRecord(schema=SourceRecord, source_id=sample_idx)
                            candidate.idx = sample_idx
                            candidate.get_item_fn = partial(self.datasource.get_item, val=True)
                            candidates = [candidate]
                            logical_op_id_to_num_samples[logical_op_id] += 1
                            phys_op_id_to_num_samples[op.get_op_id()] += 1
                        else:
                            if next_shuffled_sample_idx + j_idx == len(shuffled_sample_indices):
                                fully_sampled = True
                                break

                            # pick best output from all_outputs from previous logical operator
                            sample_idx = shuffled_sample_indices[next_shuffled_sample_idx + j_idx]
                            source_id = sample_idx_to_source_id[sample_idx]
                            record_sets = all_outputs[prev_logical_op_id][source_id]
                            all_source_record_sets = [(record_set, None) for record_set in record_sets]
                            max_quality_record_set = self.pick_highest_quality_output(all_source_record_sets)
                            if (
                                not prev_logical_op_is_filter
                                or (
                                    prev_logical_op_is_filter
                                    and max_quality_record_set.record_op_stats[0].passed_operator
                                )
                            ):
                                candidates = [record for record in max_quality_record_set]

                            # increment number of samples drawn for this logical and physical op id; even if we get multiple
                            # candidates from the previous stage in the pipeline, we only count this as one sample
                            logical_op_id_to_num_samples[logical_op_id] += 1
                            phys_op_id_to_num_samples[op.get_op_id()] += 1

                        if len(candidates) > 0:
                            op_candidate_pairs.extend([(op, candidate) for candidate in candidates])

                    # set new_operator = False and update next_shuffled_sample_idx
                    updated_frontier_ops_lst.append((op, next_shuffled_sample_idx + j, False, fully_sampled))

                frontier_ops[logical_op_id] = updated_frontier_ops_lst

                # continue if op_candidate_pairs is an empty list, as this means all records have been filtered out
                if len(op_candidate_pairs) == 0:
                    continue

                # run sampled operators on sampled candidates
                source_id_to_record_sets, source_id_to_champion_record_set = self.execute_op_set(op_candidate_pairs)

                # update all_outputs and champion_outputs dictionary
                if logical_op_id not in all_outputs:
                    all_outputs[logical_op_id] = source_id_to_record_sets
                    champion_outputs[logical_op_id] = source_id_to_champion_record_set
                else:
                    for source_id, record_sets in source_id_to_record_sets.items():
                        if source_id not in all_outputs[logical_op_id]:
                            all_outputs[logical_op_id][source_id] = record_sets
                            champion_outputs[logical_op_id][source_id] = source_id_to_champion_record_set[source_id]
                        else:
                            all_outputs[logical_op_id][source_id].extend(record_sets)
                            # NOTE: short-term solution; in practice we can get multiple champion records from different
                            # sets of operators, so we should try to find a way to only take one
                            champion_outputs[logical_op_id][source_id] = source_id_to_champion_record_set[source_id]

                # flatten lists of records and record_op_stats
                all_records, all_record_op_stats = [], []
                for _, record_sets in source_id_to_record_sets.items():
                    for record_set in record_sets:
                        all_records.extend(record_set.data_records)
                        all_record_op_stats.extend(record_set.record_op_stats)

                # update plan stats
                plan_stats.operator_stats[logical_op_id].add_record_op_stats(
                    all_record_op_stats,
                    source_op_id=prev_logical_op_id,
                    plan_id=plan.plan_id,
                )

                # add records (which are not filtered) to the cache, if allowed
                if not self.nocache:
                    for record in all_records:
                        if getattr(record, "passed_operator", True):
                            self.datadir.append_cache(logical_op_id, record)

                # compute quality for each operator
                field_to_metric_fn = self.datasource.get_field_to_metric_fn()
                all_outputs = self.score_quality(
                    op_set,
                    logical_op_id,
                    all_outputs,
                    champion_outputs,
                    expected_outputs,
                    field_to_metric_fn,
                )

            # update the (pareto) frontier for each set of operators
            frontier_ops, reservoir_ops = self.update_frontier_ops(
                frontier_ops,
                reservoir_ops,
                policy,
                all_outputs,
                logical_op_id_to_num_samples,
                phys_op_id_to_num_samples,
                is_filter_op_dict,
            )

            # update the number of samples drawn to be the max across all logical operators
            samples_drawn = max(logical_op_id_to_num_samples.values())

        # if caching was allowed, close the cache
        if not self.nocache:
            for logical_op_id, _, _ in plan:
                self.datadir.close_cache(logical_op_id)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return all_outputs, plan_stats


    def generate_sample_observations(self, sentinel_plan: SentinelPlan, policy: Policy):
        """
        This function is responsible for generating sample observation data which can be
        consumed by the CostModel.

        To accomplish this, we construct a special sentinel plan using the Optimizer which is
        capable of executing any valid physical implementation of a Filter or Convert operator
        on each record.
        """
        # if we're using validation data, get the set of expected output records
        expected_outputs = {}
        for idx in range(self.datasource.get_val_length()):
            data_records = self.datasource.get_item(idx, val=True, include_label=True)
            if not isinstance(data_records, list):
                data_records = [data_records]
            record_set = DataRecordSet(data_records, None)
            expected_outputs[record_set.source_id] = record_set

        # run sentinel plan
        execution_data, plan_stats = self.execute_sentinel_plan(sentinel_plan, expected_outputs, policy)

        return execution_data, plan_stats


    def create_sentinel_plan(self, dataset: Set, policy: Policy) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
        """
        # TODO: explicitly pull up filters; for SIGMOD we can explicitly write plans w/filters pulled up
        # initialize the optimizer
        optimizer = Optimizer(
            policy=policy,
            cost_model=CostModel(),
            no_cache=True,
            verbose=self.verbose,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_conventional_query=self.allow_conventional_query,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            allow_rag_reduction=self.allow_rag_reduction,
            allow_mixtures=self.allow_mixtures,
            optimization_strategy=OptimizationStrategy.SENTINEL,
        )

        # use optimizer to generate sentinel plans
        sentinel_plans = optimizer.optimize(dataset, policy)
        sentinel_plan = sentinel_plans[0]

        return sentinel_plan


    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # for now, enforce that we are using validation data; we can relax this after paper submission
        if not self.using_validation_data:
            raise Exception("Make sure you are using ValidationDataSource with MABSentinelExecutionEngine")

        # if nocache is True, make sure we do not re-use codegen examples
        if self.nocache:
            self.clear_cached_examples()

        # create sentinel plan
        sentinel_plan = self.create_sentinel_plan(dataset, policy)

        # generate sample execution data
        all_execution_data, plan_stats = self.generate_sample_observations(sentinel_plan, policy)

        # put sentinel plan execution stats into list and prepare list of output records
        all_plan_stats = [plan_stats]
        all_records = []

        # construct the CostModel with any sample execution data we've gathered
        cost_model = SampleBasedCostModel(sentinel_plan, all_execution_data, self.verbose)

        # (re-)initialize the optimizer
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=self.nocache,
            verbose=self.verbose,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_conventional_query=self.allow_conventional_query,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            allow_rag_reduction=self.allow_rag_reduction,
            allow_mixtures=self.allow_mixtures,
            optimization_strategy=self.optimization_strategy,
            use_final_op_quality=self.use_final_op_quality,
        )
        total_optimization_time = time.time() - execution_start_time

        # execute plan(s) according to the optimization strategy
        if self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            records, plan_stats = self.execute_confidence_interval_strategy(dataset, policy, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        else:
            records, plan_stats = self.execute_strategy(dataset, policy, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        # aggregate plan stats
        aggregate_plan_stats = self.aggregate_plan_stats(all_plan_stats)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats=aggregate_plan_stats,
            total_optimization_time=total_optimization_time,
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))),
            plan_strs={plan_id: plan_stats.plan_str for plan_id, plan_stats in aggregate_plan_stats.items()},
        )

        return all_records, execution_stats


class MABSequentialSingleThreadSentinelExecution(MABSentinelExecutionEngine, SequentialSingleThreadPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        MABSentinelExecutionEngine.__init__(self, *args, **kwargs)
        SequentialSingleThreadPlanExecutor.__init__(self, *args, **kwargs)


class MABSequentialParallelSentinelExecution(MABSentinelExecutionEngine, SequentialSingleThreadPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a pipelined, parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        MABSentinelExecutionEngine.__init__(self, *args, **kwargs)
        PipelinedParallelPlanExecutor.__init__(self, *args, **kwargs)
