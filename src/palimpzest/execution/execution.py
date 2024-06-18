import time
from palimpzest.constants import Model, PlanType
from palimpzest.corelib.schemas import SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import DataRecord
from palimpzest.operators import AggregateOp, DataSourcePhysicalOp, LimitScanOp, MarshalAndScanDataOp
from palimpzest.operators.filter import FilterOp
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.utils.model_helpers import getModels, getVisionModels
from .cost_estimator import CostEstimator
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

from palimpzest.dataclasses import OperatorStats, PlanStats, SampleExecutionData

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import os
import shutil


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
    def __init__(self) -> None:
        raise NotImplementedError

class SimpleExecution(ExecutionEngine):

    def __init__(self,
            num_samples: int=20,
            scan_start_idx: int=0,
            nocache: bool=False,
            include_baselines: bool=False,
            min_plans: Optional[int] = None,
            verbose: bool = False,
            available_models: Optional[List[Model]] = [],
            allow_model_selection: Optional[bool]=True,
            allow_code_synth: Optional[bool]=True,
            allow_token_reduction: Optional[bool]=True,
            useParallelOps: Optional[bool]=False,
            *args, **kwargs
        ) -> None:
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.nocache = nocache
        self.include_baselines = include_baselines
        self.min_plans = min_plans
        self.verbose = verbose
        self.available_models = available_models
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.useParallelOps = useParallelOps
        self.datadir = DataDirectory()

    def set_source_dataset_id(self, dataset: Set) -> str:
        """
        Sets the dataset_id of the DataSource for the given dataset.
        """
        # iterate until we reach DataSource
        while isinstance(dataset, Set):
            dataset = dataset._source

        self.source_dataset_id = dataset.dataset_id

    def execute(self,
        dataset: Set,
        policy: Policy,
    ):
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
            allow_model_selection=self.allow_model_selection,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            useParallelOps=self.useParallelOps,
            useStrategies=True,
        )

        if run_sentinels:
            for logical_plan in logical_planner.generate_plans(dataset, sentinels=True):
                for sentinel_plan in physical_planner.generate_plans(logical_plan, sentinels=True):
                    sentinel_plans.append(sentinel_plan)

            # run sentinel plans
            sample_execution_data, sentinel_records = self.run_sentinel_plans(
                sentinel_plans, self.verbose
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
        cost_estimator = CostEstimator(source_dataset_id=self.source_dataset_id,
                                       sample_execution_data=sample_execution_data)

        # estimate the cost of each plan
        for physical_plan in all_physical_plans:
            total_time, total_cost, quality = cost_estimator.estimate_plan_cost(physical_plan)
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
            final_plans = physical_planner.add_plans_closest_to_frontier(final_plans, plans, self.min_plans)

        # choose best plan and execute it
        plan = policy.choose(plans)
        new_records, stats = self.execute_plan(plan, plan_type=PlanType.FINAL) # TODO: Still WIP
        all_records = sentinel_records + new_records

        return all_records, plan, stats

    def run_sentinel_plans(
        self, sentinel_plans: List[PhysicalPlan], verbose: bool = False
    ):
        # compute number of plans
        num_sentinel_plans = len(sentinel_plans)

        all_sample_execution_data, return_records = [], []
        results = list(map(lambda x:
                self.execute_plan(*x),
                [(plan, idx, PlanType.SENTINEL) for idx, plan in enumerate(sentinel_plans)],
            )
        )
        # with ThreadPoolExecutor(max_workers=num_sentinel_plans) as executor:
        #     results = list(executor.map(lambda x:
        #             self.execute_plan(*x),
        #             [(plan, idx, PlanType.SENTINEL) for idx, plan in enumerate(sentinel_plans)],
        #         )
        #     )

        sentinel_records, sentinel_stats = zip(*results)
        for records, plan_stats, plan in zip(sentinel_records, sentinel_stats, sentinel_plans):
            # aggregate sentinel est. data
            sample_execution_data = self.getSampleExecutionData(plan_stats)
            all_sample_execution_data.extend(sample_execution_data)

            # set return_records to be records from champion model
            champion_model_name = getChampionModelName()

            # find champion model plan records and add those to all_records
            if champion_model_name in plan.getPlanModelNames():
                return_records = records

        return all_sample_execution_data, return_records # TODO: make sure you capture cost of sentinel plans.

    def _execute_dag(self, plan: PhysicalPlan, plan_stats: PlanStats, num_samples: Optional[int] = None):
        """
        Helper function which executes the physical plan. This function is overly complex for today's
        plans which are simple cascades -- but is designed with an eye towards 
        """
        output_records = []
        current_scan_idx = self.scan_start_idx
        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed

        for idx, operator in enumerate(plan.operators):
            op_id = operator.get_op_id()
            # TODO: Is it okay to have call return a list?
            if isinstance(operator, DataSourcePhysicalOp):
                idx=0
                out_records, out_stats_lst = [], []
                num_samples = num_samples if num_samples else float("inf")

                ds = operator.datadir.getRegisteredDataset(plan.datasetIdentifier)
                for filename in sorted(os.listdir(ds.path)):
                    file_path = os.path.join(ds.path, filename)
                    if os.path.isfile(file_path):
                        if idx > num_samples:
                            break
                        datasource = (
                            self.datadir.getRegisteredDataset(self.source_dataset_id)
                            if isinstance(operator, MarshalAndScanDataOp)
                            else self.datadir.getCachedResult(operator.cachedDataIdentifier)
                        )
                        candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
                        candidate.idx = current_scan_idx
                        candidate.get_item_fn = datasource.getItem
                        candidate.cardinality = datasource.cardinality
                        record, record_op_stats_lst = operator(candidate)
                        out_records.extend(record)
                        out_stats_lst.extend(record_op_stats_lst)
                        # Incrementing here bc folder may contain subfolders
                        idx += 1
            else:
                input_records = out_records
                out_records = []
                out_stats_lst = []
                for idx,input_record in enumerate(input_records):
                    if isinstance(operator, LimitScanOp) and idx == operator.limit:
                        break

                    records, record_op_stats = operator(input_record)
                    if records is None:
                        records = []
                        continue
                    elif type(records) != type([]):
                        records = [records]

                    for record in records:
                        if isinstance(operator, FilterOp):
                            if not record._passed_filter:
                                continue
                        out_records.append(record)
                    out_stats_lst.append(record_op_stats)

            # TODO code a nice __add__ function for OperatorStats and RecordOpStats
            for record_op_stats in out_stats_lst:
                x = plan_stats.operator_stats[op_id]
                x.record_op_stats_lst.append(record_op_stats)
                x.total_op_time += record_op_stats.op_time
                x.total_op_cost += record_op_stats.op_cost
                plan_stats.operator_stats[op_id] = x

        return out_records, plan_stats

    # TODO The dag style execution is not really working. I am implementing a per-records execution
    def execute_dag(self, plan: PhysicalPlan, plan_stats: PlanStats, num_samples: Optional[int] = None):
        """
        Helper function which executes the physical plan. This function is overly complex for today's
        plans which are simple cascades -- but is designed with an eye towards the future.
        """
        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0
        datasource_len = 0
        current_scan_idx = self.scan_start_idx

        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOp)
        }

        # if num_samples is not provided, set it to infinity
        num_samples = num_samples if num_samples is not None else float("inf")

        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed
        finished_executing, keep_scanning_source_records = False, True
        while not finished_executing:
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

                # TODO: if self.useParallelOps is True; execute each operator with parallelism
                # invoke datasource operator(s) until we run out of source records
                if isinstance(operator, DataSourcePhysicalOp) and keep_scanning_source_records:
                    # get handle to DataSource and pre-compute its size
                    datasource = (
                        self.datadir.getRegisteredDataset(self.source_dataset_id)
                        if isinstance(operator, MarshalAndScanDataOp)
                        else self.datadir.getCachedResult(operator.cachedDataIdentifier)
                    )
                    datasource_len = len(datasource)

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

                # only invoke aggregate operator(s) once there are no more source records and all
                # upstream operators' processing queues are empty
                elif isinstance(operator, AggregateOp):
                    upstream_queues_are_empty = True
                    for upstream_op_idx in range(op_idx):
                        upstream_queues_are_empty = (
                            upstream_queues_are_empty
                            and len(processing_queues[upstream_op_idx]) == 0
                        )
                    if not keep_scanning_source_records and upstream_queues_are_empty:
                        records, record_op_stats_lst = operator(candidates=processing_queues[op_idx])

                # otherwise, process the next record in the processing queue for this operator
                elif len(processing_queues[op_id]) > 0:
                    print(f"Processing operator {op_id} - queue length: {len(processing_queues[op_id])}")
                    input_record = processing_queues[op_id].pop(0)
                    records, record_op_stats_lst = operator(input_record)

                # update plan stats
                op_stats = plan_stats.operator_stats[op_id]
                for record_op_stats in record_op_stats_lst:
                    # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                    record_op_stats.source_op_id = prev_op_id
                    op_stats.record_op_stats_lst.append(record_op_stats)
                    op_stats.total_op_time += record_op_stats.op_time
                    op_stats.total_op_cost += record_op_stats.op_cost

                plan_stats.operator_stats[op_id] = op_stats

                # TODO some operator is not returning a singleton list
                if type(records) != type([]):
                    records = [records]

                # TODO: manage the cache here

                # update processing_queues or output_records
                for record in records:
                    if isinstance(operator, FilterOp):
                        if not record._passed_filter:
                            continue
                    if next_op_id is not None:
                        processing_queues[next_op_id].append(record)
                    else:
                        output_records.append(record)

            # update finished_executing based on whether all records have been processed
            still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
            keep_scanning_source_records = (
                current_scan_idx < datasource_len
                and source_records_scanned < num_samples
            )
            finished_executing = not keep_scanning_source_records and not still_processing

            # update finished_executing based on limit
            if isinstance(operator, LimitScanOp):
                finished_executing = (len(output_records) == operator.limit)

        return output_records, plan_stats

    # NOTE: Adding a few optional arguments for printing, etc.
    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None):
        """Initialize the stats and invoke execute_dag() to execute the plan."""
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
        # NOTE: I am writing this execution helper function with the goal of supporting future
        #       physical plans that may have joins and/or other operations with multiple sources.
        #       Thus, the implementation is overkill for today's plans, but hopefully this will
        #       avoid the need for a lot of refactoring in the future.
        # execute the physical plan;
        num_samples = self.num_samples if plan_type == PlanType.SENTINEL else None
        output_records, plan_stats = self.execute_dag(plan, plan_stats, num_samples)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats

    def getSampleExecutionData(self, plan_stats: PlanStats) -> List[RecordOpStats]:
        """Compute and return all sample execution data collected by this plan so far."""
        # construct table of observation data from sample batch of processed records
        sample_execution_data, source_op_id = [], None
        for op_id, operator_stats in plan_stats.operator_stats.items():
            # append observation data for each record
            sample_execution_data.extend(operator_stats.record_op_stats_lst)
            continue
            for record_op_stats in operator_stats.record_op_stats_lst:
                # compute minimal observation which is supported by all operators
                # TODO: one issue with this setup is that cache_scans of previously computed queries
                #       may not match w/these observations due to the diff. op_name
                observation_arguments = {
                    "record_uuid": record_op_stats.record_uuid,
                    "record_parent_uuid": record_op_stats.record_parent_uuid,
                    "op_id": op_id,
                    "op_name": record_op_stats.op_name,
                    "op_cost": record_op_stats.op_cost,
                    "source_op_id": source_op_id,
                    "op_time": record_op_stats.op_time,
                }
                if "record_details" in record_op_stats.__dict__:
                    observation_arguments.update({
                    "model_name": (
                        record_op_stats.record_details["record_details"]
                        if "model_name" in record_op_stats.op_details
                        else None
                    ),
                    "filter_str": (
                        record_op_stats.record_details["filter_str"]
                        if "filter_str" in record_op_stats.op_details
                        else None
                    ),
                    "input_fields_str": (
                        "-".join(sorted(record_op_stats.op_details["input_fields"]))
                        if "input_fields" in record_op_stats.op_details
                        else None
                    ),
                    "generated_fields_str": (
                        "-".join(sorted(record_op_stats.op_details["generated_fields"]))
                        if "generated_fields" in record_op_stats.op_details
                        else None
                    ),}
                    )
                    answer = {}
                    # return key->value mapping for generated fields for induce
                    if "generated_fields" in record_op_stats.op_details:
                        for field in record_op_stats.op_details["generated_fields"]:
                            answer[field] = record_op_stats.record_state[field]
                    observation_arguments["answer"] = answer

                if 'record_state' in record_op_stats.__dict__:
                    observation_arguments.update({
                    "total_input_tokens": (
                        record_op_stats.record_details["total_input_tokens"]
                        if "total_input_tokens" in record_op_stats.record_state
                        else None
                    ),
                    "total_output_tokens": (
                        record_op_stats.record_details["total_output_tokens"]
                        if "total_output_tokens" in record_op_stats.record_state
                        else None
                    ),
                    "total_input_cost": (
                        record_op_stats.record_details["total_input_cost"]
                        if "total_input_cost" in record_op_stats.record_state
                        else None
                    ),
                    "total_output_cost": (
                        record_op_stats.record_details["total_output_cost"]
                        if "total_output_cost" in record_op_stats.record_state
                        else None
                    ),
                    "answer": (record_op_stats.record_state["answer"] 
                        if "answer" in record_op_stats.record_state
                        else None),
                    "passed_filter": (
                        record_op_stats.record_state["_passed_filter"]
                        if "_passed_filter" in record_op_stats.record_state
                        else None
                    ),
                    })

                    # return T/F for filter
                    if "_passed_filter" in record_op_stats.record_state:
                        observation_arguments["answer"] = record_op_stats.record_state["_passed_filter"]

                observation = SampleExecutionData(**observation_arguments)
                # add observation to list of observations
                sample_execution_data.append(observation)

            # update source_op_id
            source_op_id = op_id

        return sample_execution_data

class Execute:
    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int=20,
        nocache: bool=False,
        include_baselines: bool=False,
        min_plans: Optional[int] = None,
        verbose: bool = False,
        available_models: Optional[List[Model]] = [],
        allow_model_selection: Optional[bool]=True,
        allow_code_synth: Optional[bool]=True,
        allow_token_reduction: Optional[bool]=True,
        useParallelOps: Optional[bool]=False,
        execution_engine: ExecutionEngine = SimpleExecution,
        *args,
        **kwargs
    ):

        return execution_engine(
            num_samples=num_samples,
            nocache=nocache,
            include_baselines=include_baselines,
            min_plans=min_plans,
            verbose=verbose,
            available_models=available_models,
            allow_code_synth=allow_code_synth,
            allow_model_selection=allow_model_selection,
            allow_token_reduction=allow_token_reduction,
            useParallelOps=useParallelOps,
            *args,
            **kwargs
        ).execute(
            dataset=dataset,
            policy=policy
        )
