from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import SentinelPlanStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import CacheScanDataOp, MarshalAndScanDataOp, ScanPhysicalOp
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.tools.logger import setup_logger
from palimpzest.utils.progress import create_progress_manager

logger = setup_logger(__name__)

class MABExecutionStrategy(SentinelExecutionStrategy):
    """
    This class implements the Multi-Armed Bandit (MAB) execution strategy for SentinelQueryProcessors.
    """

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
        for logical_op_id, source_idx_to_record_sets in all_outputs.items():
            # compute selectivity for each physical operator
            phys_op_to_num_inputs, phys_op_to_num_outputs = {}, {}
            for _, record_sets in source_idx_to_record_sets.items():
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
            for _, record_sets in source_idx_to_record_sets.items():
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


    def execute_op_set(self, op_candidate_pairs: list[PhysicalOperator, DataRecord | int]):
        # TODO: post-submission we will need to modify this to:
        # - submit all candidates for aggregate operators
        # - handle limits
        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = []
            for operator, candidate in op_candidate_pairs:
                future = executor.submit(PhysicalOperator.execute_op_wrapper, operator, candidate)
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

            # compute mapping from source_idx to record sets for all operators and for champion operator
            all_record_sets, champion_record_sets = {}, {}
            for _, candidate in op_candidate_pairs:
                candidate_output_record_sets, source_idx = [], None
                for record_set, operator, candidate_ in output_record_sets:
                    if candidate == candidate_:
                        candidate_output_record_sets.append((record_set, operator))

                        # get the source_idx associated with this input record;
                        # for scan operators, `candidate` will be the source_idx
                        source_idx = candidate.source_idx if isinstance(candidate, DataRecord) else candidate

                # select the champion (i.e. best) record_set from all the record sets computed for this candidate
                champion_record_set = self.pick_output_fn(candidate_output_record_sets)

                # add champion record_set to mapping from source_idx --> champion record_set
                champion_record_sets[source_idx] = champion_record_set

                # add all record_sets computed for this source_idx to mapping from source_idx --> record_sets
                all_record_sets[source_idx] = [tup[0] for tup in candidate_output_record_sets]

        return all_record_sets, champion_record_sets


    def execute_sentinel_plan(self, plan: SentinelPlan, expected_outputs: dict[str, dict], policy: Policy):
        """
        """
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert all(isinstance(op, ScanPhysicalOp) for op in plan.operator_sets[0]), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize progress manager
        self.progress_manager = create_progress_manager(plan, self.num_samples)

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # shuffle the indices of records to sample
        total_num_samples = len(self.val_datasource)
        shuffled_source_indices = [int(idx) for idx in np.arange(total_num_samples)]
        self.rng.shuffle(shuffled_source_indices)

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
                    j = min(self.j, len(shuffled_source_indices)) if new_operator else 1
                    for j_idx in range(j):
                        candidates = []
                        if isinstance(op, (MarshalAndScanDataOp, CacheScanDataOp)):
                            source_idx = shuffled_source_indices[(next_shuffled_sample_idx + j_idx) % len(shuffled_source_indices)]
                            candidates = [source_idx]
                            logical_op_id_to_num_samples[logical_op_id] += 1
                            phys_op_id_to_num_samples[op.get_op_id()] += 1
                        else:
                            if next_shuffled_sample_idx + j_idx == len(shuffled_source_indices):
                                fully_sampled = True
                                break

                            # pick best output from all_outputs from previous logical operator
                            source_idx = shuffled_source_indices[next_shuffled_sample_idx + j_idx]
                            record_sets = all_outputs[prev_logical_op_id][source_idx]
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
                source_idx_to_record_sets, source_idx_to_champion_record_set = self.execute_op_set(op_candidate_pairs)

                # update all_outputs and champion_outputs dictionary
                if logical_op_id not in all_outputs:
                    all_outputs[logical_op_id] = source_idx_to_record_sets
                    champion_outputs[logical_op_id] = source_idx_to_champion_record_set
                else:
                    for source_idx, record_sets in source_idx_to_record_sets.items():
                        if source_idx not in all_outputs[logical_op_id]:
                            all_outputs[logical_op_id][source_idx] = record_sets
                            champion_outputs[logical_op_id][source_idx] = source_idx_to_champion_record_set[source_idx]
                        else:
                            all_outputs[logical_op_id][source_idx].extend(record_sets)
                            # NOTE: short-term solution; in practice we can get multiple champion records from different
                            # sets of operators, so we should try to find a way to only take one
                            champion_outputs[logical_op_id][source_idx] = source_idx_to_champion_record_set[source_idx]

                # flatten lists of records and record_op_stats
                all_records, all_record_op_stats = [], []
                for _, record_sets in source_idx_to_record_sets.items():
                    for record_set in record_sets:
                        all_records.extend(record_set.data_records)
                        all_record_op_stats.extend(record_set.record_op_stats)

                # update plan stats
                plan_stats.add_record_op_stats(all_record_op_stats)

                # add records (which are not filtered) to the cache, if allowed
                if self.cache:
                    for record in all_records:
                        if getattr(record, "passed_operator", True):
                            # self.datadir.append_cache(logical_op_id, record)
                            pass

                # compute quality for each operator
                all_outputs = self.score_quality(
                    op_set,
                    logical_op_id,
                    all_outputs,
                    champion_outputs,
                    expected_outputs,
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
        if self.cache:
            for _, _, _ in plan:
                # self.datadir.close_cache(logical_op_id)
                pass

        # finalize plan stats
        plan_stats.finish()

        return all_outputs, plan_stats
