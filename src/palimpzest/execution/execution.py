from palimpzest.constants import ExecutionStrategy, Model, PlanType, PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.corelib.schemas import SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import DataRecord
from palimpzest.operators import AggregateOp, DataSourcePhysicalOp, LimitScanOp, MarshalAndScanDataOp, PhysicalOperator
from palimpzest.operators.filter import FilterOp
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.qualityestimation import ValidationData, QualityEstimator
from palimpzest.utils.model_helpers import getModels, getVisionModels
from palimpzest.datasources.datasources import DataSource
from .cost_estimator import CostEstimator
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

from palimpzest.dataclasses import OperatorStats, PlanStats

from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional, Union

import multiprocessing
import os
import shutil
import time


def _getAllowedModels(self, subplan: PhysicalPlan) -> List[Model]:
    """
    This function handles the logic of determining which model(s) can be used for a Convert or Filter
    operation during physical plan construction.

    The logic for determining which models can be used is as follows:
    - If model selection is allowed --> then all models may be used
    - If the subplan does not yet have an operator which uses a (non-vision) model --> then all models may be used
    - If the subplan has an operator which uses a (non-vision) model --> only the subplan's model may be used
    """
    # return all models if model selection is allowed
    if self.allow_model_selection:
        return getModels()

    # otherwise, get models used by subplan
    subplan_model, vision_models = None, getVisionModels()
    for phys_op in subplan.operators:
        model = getattr(phys_op, "model", None)
        if model is not None and model not in vision_models:
            subplan_model = model
            break

    # return all models if subplan does not have any models yet
    if subplan_model is None:
        return getModels()

    # otherwise return the subplan model
    return [subplan_model]


class ExecutionEngine:
    def __init__(self,
            num_samples: int=20,
            scan_start_idx: int=0,
            nocache: bool=False,
            include_baselines: bool=False,
            min_plans: Optional[int] = None,
            verbose: bool = False,
            available_models: List[Model] = [],
            allow_bonded_query: List[Model] = True,
            allow_model_selection: bool=True,
            allow_code_synth: bool=True,
            allow_token_reduction: bool=True,
            execution_strategy: bool=ExecutionStrategy.SINGLE_THREADED,
            useParallelOps: bool=False,
            max_workers: Optional[int]=None,
            *args, **kwargs
        ) -> None:
        self.num_samples = num_samples
        self.validation_examples = validation_examples
        self.scan_start_idx = scan_start_idx
        self.nocache = nocache
        self.include_baselines = include_baselines
        self.min_plans = min_plans
        self.verbose = verbose
        self.available_models = available_models
        if not available_models:
            self.available_models = getModels()
        print("Available models: ", self.available_models)
        self.allow_model_selection = allow_model_selection
        self.allow_bonded_query = allow_bonded_query
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.execution_strategy = execution_strategy
        self.max_workers = max_workers
        if self.max_workers is None and self.execution_strategy == ExecutionStrategy.PARALLEL:
            self.max_workers = self.set_max_workers()
        else:
            self.max_workers = 1
        self.datadir = DataDirectory()
        self.useParallelOps = useParallelOps

    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None,
                     max_workers: Optional[int] = None):
        """Initialize the stats and the execute the plan."""
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")
    
    def set_source_dataset_id(self, dataset: Set) -> str:
        """
        Sets the dataset_id of the DataSource for the given dataset.
        """
        # iterate until we reach DataSource
        while isinstance(dataset, Set):
            dataset = dataset._source

        self.source_dataset_id = dataset.dataset_id

    def set_max_workers(self):
        # for now, return the number of system CPUs;
        # in the future, we may want to consider the models the user has access to
        # and whether or not they will encounter rate-limits. If they will, we should
        # set the max workers in a manner that is designed to avoid hitting them.
        # Doing this "right" may require considering their logical, physical plan,
        # and tier status with LLM providers. It may also be worth dynamically
        # changing the max_workers in response to 429 errors.
        return multiprocessing.cpu_count()

    def execute(self,
        dataset: Set,
        policy: Policy,
    ):
        # TODO: we should be able to remove this w/our cache management
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            dspy_cache_dir = os.path.join(
                os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/"
            )
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)

            # remove codegen samples from previous dataset from cache
            cache = self.datadir.getCacheService()
            cache.rmCache()

        # set the the id of the source dataset
        # todo(chjun): If this plan will use Cashe source, we should set it to cache source here.
        self.set_source_dataset_id(dataset)

        # NOTE: this checks if the entire computation is cached; it will re-run
        #       the sentinels even if the computation is partially cached
        # only run sentinels if there isn't a cached result already
        uid = dataset.universalIdentifier()
        run_sentinels = self.nocache or not self.datadir.hasCachedAnswer(uid)

        sentinel_plans, sample_execution_data, sentinel_records = [], [], []

        # initialize logical and physical planner
        # NOTE The Exeuction class MUST KNOW THE PLANNER!
        #  if I disallow code synth for my planning, I will have to disallow it for my execution. Now, I can't do that.
        # get sentinel plans
        logical_planner = LogicalPlanner(self.nocache)
        physical_planner = PhysicalPlanner(
            num_samples=self.num_samples,
            scan_start_idx=0,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_model_selection=self.allow_model_selection,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            useParallelOps=self.useParallelOps,
        )

        if run_sentinels:
            for logical_plan in logical_planner.generate_plans(dataset, sentinels=True):
                for sentinel_plan in physical_planner.generate_plans(
                    logical_plan, sentinels=True
                ):
                    sentinel_plans.append(sentinel_plan)

            # run sentinel plans
            sample_execution_data, sentinel_records = self.run_sentinel_plans(
                sentinel_plans,
                self.verbose,
            )

        # (re-)initialize logical and physical planner
        scan_start_idx = self.num_samples if run_sentinels else 0
        physical_planner.scan_start_idx = scan_start_idx

        # NOTE: in the future we may use operator_estimates below to limit the number of plans
        #       that we need to consider during plan generation. I.e., we may be able to save time
        #       by pre-computing the set of viable models / execution strategies at each operator
        #       based on the sample execution data we get.
        #
        # enumerate all possible physical plans
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)

        # construct the CostEstimator with any sample execution data we've gathered
        cost_estimator = CostEstimator(
            source_dataset_id=self.source_dataset_id,
            sample_execution_data=sample_execution_data,
        )

        # estimate the cost of each plan
        for physical_plan in all_physical_plans:
            total_time, total_cost, quality = cost_estimator.estimate_plan_cost(
                physical_plan
            )
            physical_plan.total_time = total_time
            physical_plan.total_cost = total_cost
            physical_plan.quality = quality

        # deduplicate plans with identical cost estimates
        plans = physical_planner.deduplicate_plans(all_physical_plans)

        # select pareto frontier of plans
        final_plans = physical_planner.select_pareto_optimal_plans(plans)

        # for experimental evaluation, we may want to include baseline plans
        if self.include_baselines:
            final_plans = physical_planner.add_baseline_plans(final_plans)

        if self.min_plans is not None and len(final_plans) < self.min_plans:
            final_plans = physical_planner.add_plans_closest_to_frontier(
                final_plans, plans, self.min_plans
            )

        # choose best plan and execute it
        #TODO return plan_idx ?
        plan = policy.choose(plans)
        new_records, stats = self.execute_plan(plan=plan, 
                                               plan_type=PlanType.FINAL, 
                                               plan_idx=0,
                                               max_workers=self.max_workers)
        all_records = sentinel_records + new_records

        return all_records, plan, stats

    def run_sentinel_plans(
        self,
        sentinel_plans: List[PhysicalPlan],
        verbose: bool = False,
    ):
        # compute number of plans
        num_sentinel_plans = len(sentinel_plans)

        all_sample_execution_data, return_records = [], []

        # results = list(map(lambda x:
        #         self.execute_plan(*x),
        #         [(plan, idx, PlanType.SENTINEL) for idx, plan in enumerate(sentinel_plans)],
        #     )
        # )
        sentinel_workers = min(self.max_workers, num_sentinel_plans)
        with ThreadPoolExecutor(max_workers=sentinel_workers) as executor:
            max_workers_per_plan = max(self.max_workers / num_sentinel_plans, 1)
            results = list(executor.map(lambda x: self.execute_plan(**x),
                    [{"plan":plan, 
                      "plan_type":PlanType.SENTINEL, 
                      "plan_idx": idx,
                      "max_workers":max_workers_per_plan} for idx, plan in enumerate(sentinel_plans)],
                )
            )

        sentinel_records, sentinel_stats = zip(*results)
        for records, plan_stats, plan in zip(
            sentinel_records, sentinel_stats, sentinel_plans
        ):
            # aggregate sentinel est. data
            sample_execution_data = []
            for op_id, operator_stats in plan_stats.operator_stats.items():
                all_sample_execution_data.extend(operator_stats.record_op_stats_lst)

            # set return_records to be records from champion model
            champion_model_name = getChampionModelName()

            # find champion model plan records and add those to all_records
            if champion_model_name in plan.getPlanModelNames():
                return_records = records

        if len(sentinel_records) > 0 and len(return_records) ==0:
            return_records = sentinel_records[0]

class SequentialSingleThreadExecution(ExecutionEngine):

    # NOTE: Adding a few optional arguments for printing, etc.
    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None,
                     max_workers: Optional[int] = None):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"{plan_type.value} {str(plan_idx)}:")
            plan.printPlan()
            print("---")

        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id(), plan_idx = plan_idx) # TODO move into PhysicalPlan.__init__?
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) # TODO: also add op_details here

        # execute the physical plan;
        num_samples = self.num_samples if plan_type == PlanType.SENTINEL else float("inf")

        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0
        current_scan_idx = self.scan_start_idx

        # get handle to DataSource and pre-compute its size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.getRegisteredDataset(self.source_dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.cachedDataIdentifier)
        )
        datasource_len = len(datasource)

        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOp)
        }

        # execute the plan one operator at a time
        for op_idx, operator in enumerate(plan.operators):
            op_id = operator.get_op_id()
            prev_op_id = (
                plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
            )
            next_op_id = (
                plan.operators[op_idx + 1].get_op_id()
                if op_idx + 1 < len(plan.operators)
                else None
            )

            # initialize output records and record_op_stats_lst for this operator
            records, record_op_stats_lst = [], []

            # invoke datasource operator(s) until we run out of source records
            if isinstance(operator, DataSourcePhysicalOp):
                keep_scanning_source_records = True
                while keep_scanning_source_records:
                    # construct input DataRecord for DataSourcePhysicalOp
                    candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
                    candidate.idx = current_scan_idx
                    candidate.get_item_fn = datasource.getItem
                    candidate.cardinality = datasource.cardinality

                    # run DataSourcePhysicalOp on record
                    out_records, out_record_op_stats_lst = operator(candidate)
                    records.extend(out_records)
                    record_op_stats_lst.extend(out_record_op_stats_lst)

                    # update the current scan index
                    current_scan_idx += 1

                    # update whether to keep scanning source records
                    keep_scanning_source_records = (
                        current_scan_idx < datasource_len
                        and len(records) < num_samples
                    )

            # aggregate operators accept all input records at once
            elif isinstance(operator, AggregateOp):
                records, record_op_stats_lst = operator(candidates=processing_queues[op_id])

            # otherwise, process the records in the processing queue for this operator one at a time
            elif len(processing_queues[op_id]) > 0:
                for input_record in processing_queues[op_id]:
                    out_records, out_record_op_stats_lst = operator(input_record)
                    records.extend(out_records)
                    record_op_stats_lst.extend(out_record_op_stats_lst)

            # update plan stats
            op_stats = plan_stats.operator_stats[op_id]
            for record_op_stats in record_op_stats_lst:
                # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                record_op_stats.source_op_id = prev_op_id
                op_stats.record_op_stats_lst.append(record_op_stats)
                op_stats.total_op_time += record_op_stats.time_per_record
                op_stats.total_op_cost += record_op_stats.cost_per_record

            plan_stats.operator_stats[op_id] = op_stats

            # add records (which are not filtered) to the cache, if allowed
            if not self.nocache:
                for record in records:
                    if getattr(record, "_passed_filter", True):
                        self.datadir.appendCache(operator.targetCacheId, record)

            # update processing_queues or output_records
            for record in records:
                if isinstance(operator, FilterOp):
                    if not record._passed_filter:
                        continue
                if next_op_id is not None:
                    processing_queues[next_op_id].append(record)
                else:
                    output_records.append(record)

        # if caching was allowed, close the cache
        if not self.nocache:
            for operator in plan.operators:
                self.datadir.closeCache(operator.targetCacheId)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats


class PipelinedSingleThreadExecution(ExecutionEngine):

    # NOTE: Adding a few optional arguments for printing, etc.
    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None,
                     max_workers: Optional[int] = None):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"{plan_type.value} {str(plan_idx)}:")
            plan.printPlan()
            print("---")

        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id()) # TODO move into PhysicalPlan.__init__?
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) # TODO: also add op_details here

        # execute the physical plan;
        num_samples = self.num_samples if plan_type == PlanType.SENTINEL else float("inf")        

        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0
        current_scan_idx = self.scan_start_idx

        # get handle to DataSource and pre-compute its size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.getRegisteredDataset(self.source_dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.cachedDataIdentifier)
        )
        datasource_len = len(datasource)

        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOp)
        }

        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed
        keep_scanning_source_records = True
        while keep_scanning_source_records:
            output_records_of_root_record = []
            for op_idx, operator in enumerate(plan.operators):
                op_id = operator.get_op_id()

                prev_op_id = (
                    plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
                )
                next_op_id = (
                    plan.operators[op_idx + 1].get_op_id()
                    if op_idx + 1 < len(plan.operators)
                    else None
                )
                records_processed = False

                # invoke datasource operator(s) until we run out of source records
                if isinstance(operator, DataSourcePhysicalOp):
                    if keep_scanning_source_records:
                        # construct input DataRecord for DataSourcePhysicalOp
                        candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
                        candidate.idx = current_scan_idx
                        candidate.get_item_fn = datasource.getItem
                        candidate.cardinality = datasource.cardinality

                        # run DataSourcePhysicalOp on record
                        records, record_op_stats_lst = operator(candidate)

                        # update number of source records scanned and the current index
                        source_records_scanned += len(records)
                        current_scan_idx += 1
                        records_processed = True
                    else:
                        continue

                # only invoke aggregate operator(s) once there are no more source records and all
                # upstream operators' processing queues are empty
                elif isinstance(operator, AggregateOp):
                    upstream_ops_are_finished = True
                    for upstream_op_idx in range(op_idx):
                        upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
                        upstream_ops_are_finished = (
                            upstream_ops_are_finished
                            and len(processing_queues[upstream_op_id]) == 0
                        )
                    if not keep_scanning_source_records and upstream_ops_are_finished:
                        records, record_op_stats_lst = operator(candidates=processing_queues[op_id])
                        records_processed = True

                elif len(processing_queues[op_id]) > 0:
                    input_record = processing_queues[op_id].pop(0)
                    records, record_op_stats_lst = operator(input_record)
                    records_processed = True

                if records_processed:
                    # update plan stats
                    op_stats = plan_stats.operator_stats[op_id]
                    for record_op_stats in record_op_stats_lst:
                        # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                        record_op_stats.source_op_id = prev_op_id
                        op_stats.record_op_stats_lst.append(record_op_stats)
                        op_stats.total_op_time += record_op_stats.time_per_record
                        op_stats.total_op_cost += record_op_stats.cost_per_record

                    plan_stats.operator_stats[op_id] = op_stats

                    # add records (which are not filtered) to the cache, if allowed
                    if not self.nocache:
                        for record in records:
                            if getattr(record, "_passed_filter", True):
                                self.datadir.appendCache(operator.targetCacheId, record)

                    # update processing_queues or output_records
                    for record in records:
                        if isinstance(operator, FilterOp):
                            if not record._passed_filter:
                                continue
                        if next_op_id is not None:
                            processing_queues[next_op_id].append(record)
                        else:
                            output_records.append(record)

            keep_scanning_source_records = (
                current_source_scan_idx < datasource_len
                and source_records_readout < num_samples
            )

            # update finished_executing based on limit
            if isinstance(operator, LimitScanOp):
                keep_scanning_source_records = operator.limit > len(output_records)  

        # if caching was allowed, close the cache
        if not self.nocache:
            for operator in plan.operators:
                self.datadir.closeCache(operator.targetCacheId)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats


class PipelinedParallelExecution(ExecutionEngine):

    @staticmethod
    def execute_op_wrapper(operator: PhysicalOperator, op_input: Union[DataRecord, List[DataRecord]]):
        """
        Wrapper function around operator execution which also and returns the operator.
        This is useful in the parallel setting(s) where operators are executed by a worker pool,
        and it is convenient to return the op_id along with the computation result.
        """
        records, record_op_stats_lst = operator(op_input)

        return records, record_op_stats_lst, operator

    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None,
                     max_workers: Optional[int] = None):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"{plan_type} {str(plan_idx)}:")
            plan.printPlan()
            print("---")

        if len(plan.operators) == 0:
            return [], PlanStats(plan_id=plan.plan_id())

        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(
            plan_id=plan.plan_id()
        )  # TODO move into PhysicalPlan.__init__?
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) # TODO: also add op_details here

        # execute the physical plan;
        num_samples = self.num_samples if plan_type == PlanType.SENTINEL else float("inf")  

        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0

        # initialize data structures to help w/processing DAG
        processing_queue = []
        op_id_to_futures_in_flight = {op.get_op_id(): 0 for op in plan.operators}
        op_id_to_prev_operator = {
            op.get_op_id(): plan.operators[idx - 1] if idx > 0 else None
            for idx, op in enumerate(plan.operators)
        }
        op_id_to_next_operator = {
            op.get_op_id(): plan.operators[idx + 1] if idx + 1 < len(plan.operators) else None
            for idx, op in enumerate(plan.operators)
        }

        # get handle to DataSource and pre-compute its op_id and size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.getRegisteredDataset(self.source_dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.cachedDataIdentifier)
        )
        source_op_id = source_operator.get_op_id()
        datasource_len = len(datasource)

        # compute op_id and limit of final limit operator (if one exists)
        final_limit = plan.operators[-1].limit if isinstance(plan.operators[-1], LimitScanOp) else None

        # create thread pool w/max workers
        futures = []
        current_scan_idx = self.scan_start_idx
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # create initial set of futures to read first source file;
            # construct input DataRecord for DataSourcePhysicalOp
            candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
            candidate.idx = current_scan_idx
            candidate.get_item_fn = datasource.getItem
            candidate.cardinality = datasource.cardinality
            futures.append(executor.submit(PipelinedParallelExecution.execute_op_wrapper, source_operator, candidate))
            op_id_to_futures_in_flight[source_op_id] += 1
            current_scan_idx += 1   

            # iterate until we have processed all operators on all records or come to an early stopping condition
            while len(futures) > 0:
                # get the set of futures that have (and have not) finished in the last PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

                # cast not_done_futures from a set to a list so we can append to it
                not_done_futures = list(not_done_futures)

                # process finished futures, creating new ones as needed
                new_futures = []
                for future in done_futures:
                    # get the result
                    records, record_op_stats_lst, operator = future.result()
                    op_id = operator.get_op_id()

                    # decrement future from mapping of futures in-flight
                    op_id_to_futures_in_flight[op_id] -= 1

                    # update plan stats
                    op_stats = plan_stats.operator_stats[op_id]
                    for record_op_stats in record_op_stats_lst:
                        # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                        prev_operator = op_id_to_prev_operator[op_id]
                        record_op_stats.source_op_id = prev_operator.get_op_id() if prev_operator is not None else None
                        op_stats.record_op_stats_lst.append(record_op_stats)
                        op_stats.total_op_time += record_op_stats.time_per_record
                        op_stats.total_op_cost += record_op_stats.cost_per_record

                    plan_stats.operator_stats[op_id] = op_stats

                    # process each record output by the future's operator
                    for record in records:
                        # skip records which are filtered out
                        if not getattr(record, "_passed_filter", True):
                            continue

                        # add records (which are not filtered) to the cache, if allowed
                        if not self.nocache:
                            self.datadir.appendCache(operator.targetCacheId, record)

                        # add records to processing queue if there is a next_operator; otherwise add to output_records
                        next_operator = op_id_to_next_operator[op_id]
                        if next_operator is not None:
                            processing_queue.append((next_operator, record))
                        else:
                            output_records.append(record)

                    # if this operator was a source scan, update the number of source records scanned
                    if op_id == source_op_id:
                        source_records_scanned += len(records)

                        # scan next record if we can still draw records from source
                        if source_records_scanned < num_samples and current_scan_idx < datasource_len:
                            # construct input DataRecord for DataSourcePhysicalOp
                            candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
                            candidate.idx = current_scan_idx
                            candidate.get_item_fn = datasource.getItem
                            candidate.cardinality = datasource.cardinality
                            new_futures.append(executor.submit(PipelinedParallelExecution.execute_op_wrapper, source_operator, candidate))
                            op_id_to_futures_in_flight[source_op_id] += 1
                            current_scan_idx += 1

                    # check early stopping condition based on final limit
                    if final_limit is not None and len(output_records) >= final_limit:
                        output_records = output_records[:final_limit]
                        futures = []
                        break

                    # only invoke aggregate operator(s) once all upstream operators' processing queues are empty
                    # and their in-flight futures are finished
                    if isinstance(operator, AggregateOp):
                        this_op_idx = 0
                        while op_id != plan.operators[this_op_idx].get_op_id():
                            this_op_idx += 1

                        upstream_ops_are_finished = True
                        for upstream_op_idx in range(this_op_idx):
                            upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
                            upstream_op_id_queue = list(filter(lambda tup: tup[0].get_op_id() == upstream_op_id, processing_queue))

                            upstream_ops_are_finished = (
                                upstream_ops_are_finished
                                and len(upstream_op_id_queue) == 0
                                and op_id_to_futures_in_flight[upstream_op_id] == 0
                            )

                        if upstream_ops_are_finished:
                            candidates = list(filter(lambda tup: tup[0].get_op_id() == op_id, processing_queue))
                            candidates = list(map(lambda tup: tup[1], candidates))
                            future = executor.submit(PipelinedParallelExecution.execute_op_wrapper, operator, candidates)
                            new_futures.append(future)
                            op_id_to_futures_in_flight[op_id] += 1
                            processing_queue = list(filter(lambda tup: tup[0].get_op_id() != op_id, processing_queue))

                    # otherwise, process all the records in the processing queue
                    else:
                        for operator, candidate in processing_queue:
                            future = executor.submit(PipelinedParallelExecution.execute_op_wrapper, operator, candidate)
                            new_futures.append(future)
                            op_id_to_futures_in_flight[op_id] += 1

                        processing_queue = []

                # update list of futures
                not_done_futures.extend(new_futures)
                futures = not_done_futures

        # if caching was allowed, close the cache
        if not self.nocache:
            for operator in plan.operators:
                self.datadir.closeCache(operator.targetCacheId)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats


class Execute:
    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int = 20,
        nocache: bool = False,
        include_baselines: bool = False,
        min_plans: Optional[int] = None,
        verbose: bool = False,
        available_models: Optional[List[Model]] = [],
        allow_bonded_query: Optional[List[Model]] = [],
        allow_model_selection: Optional[bool]=True,
        allow_code_synth: Optional[bool]=True,
        allow_token_reduction: Optional[bool]=True,
        useParallelOps: Optional[bool]=False,
        execution_engine: ExecutionEngine = SequentialSingleThreadExecution,
        *args,
        **kwargs,
    ):
        return execution_engine(
            num_samples=num_samples,
            nocache=nocache,
            include_baselines=include_baselines,
            min_plans=min_plans,
            verbose=verbose,
            available_models=available_models,
            allow_bonded_query=allow_bonded_query,
            allow_code_synth=allow_code_synth,
            allow_model_selection=allow_model_selection,
            allow_token_reduction=allow_token_reduction,
            useParallelOps=useParallelOps,
            *args,
            **kwargs,
        ).execute(dataset=dataset, policy=policy)
