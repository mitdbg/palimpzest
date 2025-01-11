import time

from palimpzest.core.data.dataclasses import OperatorStats, PlanStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import SourceRecord
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_engine import ExecutionEngine
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.datasource import DataSourcePhysicalOp, MarshalAndScanDataOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.sets import Dataset


class StreamingSequentialExecution(ExecutionEngine):
    """This class can be used for a streaming, record-based execution.
    Results are returned as an iterable that can be consumed by the caller."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plan: PhysicalPlan | None = None
        self._plan_stats: PlanStats | None = None
        self.last_record = False
        self.current_scan_idx: int = 0
        self.plan_generated: bool = False
        self.records_count: int = 0

    @property
    def plan(self) -> PhysicalPlan:
        if self._plan is None:
            raise Exception("Plan has not been generated yet.")
        return self._plan

    @plan.setter
    def plan(self, plan: PhysicalPlan):
        self._plan = plan

    @property
    def plan_stats(self) -> PlanStats:
        if self._plan_stats is None:
            raise Exception("Plan stats have not been generated yet.")
        return self._plan_stats

    @plan_stats.setter
    def plan_stats(self, plan_stats: PlanStats):
        self._plan_stats = plan_stats

    def generate_plan(self, dataset: Dataset, policy: Policy):
        self.clear_cached_examples()
        start_time = time.time()

        cost_model = CostModel()
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
        )

        # Effectively always use the optimal strategy
        plans = optimizer.optimize(dataset, policy)
        self.plan = plans[0]
        self.plan_stats = PlanStats(plan_id=self.plan.plan_id)
        for op in self.plan.operators:
            if isinstance(op, AggregateOp):
                raise Exception("You cannot have a Streaming Execution if there is an Aggregation Operator")
            op_id = op.get_op_id()
            op_name = op.op_name()
            op_details = {k: str(v) for k, v in op.get_id_params().items()}
            self.plan_stats.operator_stats[op_id] = OperatorStats(op_id=op_id, op_name=op_name, op_details=op_details) 
        print("Time for planning: ", time.time() - start_time)
        self.plan_generated = True
        return self.plan

    def execute(
        self,
        dataset: Dataset,
        policy: Policy,
    ):
        start_time = time.time()
        # Always delete cache
        if not self.plan_generated:
            self.generate_plan(dataset, policy)

        input_records = self.get_input_records()
        for idx, record in enumerate(input_records):
            # print("Iteration number: ", idx+1, "out of", len(input_records))
            output_records = self.execute_opstream(self.plan, record)
            if idx == len(input_records) - 1:
                total_plan_time = time.time() - start_time
                self.plan_stats.finalize(total_plan_time)

            yield output_records, self.plan, self.plan_stats

    def get_input_records(self):
        scan_operator = self.plan.operators[0]
        datasource = (
            self.datadir.get_registered_dataset(scan_operator.dataset_id)
            if isinstance(scan_operator, MarshalAndScanDataOp)
            else self.datadir.get_cached_result(scan_operator.dataset_id)
        )
        if not datasource:
            raise Exception("Data source not found")
        datasource_len = len(datasource)

        input_records = []
        record_op_stats = []
        for idx in range(datasource_len):
            # NOTE: this DataRecord will be discarded and replaced by the scan_operator;
            #       it is simply a vessel to inform the scan_operator which record to fetch
            candidate = DataRecord(schema=SourceRecord, source_id=idx)
            candidate.idx = idx
            candidate.get_item_fn = datasource.get_item
            records, record_op_stats_lst = scan_operator(candidate)
            input_records += records
            record_op_stats += record_op_stats_lst

        op_id = scan_operator.get_op_id()
        self.plan_stats.operator_stats[op_id].add_record_op_stats(
            record_op_stats,
            source_op_id=None,
            plan_id=self.plan.plan_id,
        )

        return input_records

    def execute_opstream(self, plan, record):
        # initialize list of output records and intermediate variables
        input_records = [record]
        record_op_stats_lst = []

        for op_idx, operator in enumerate(plan.operators):
            # TODO: this being defined in the for loop potentially makes the return
            # unbounded if plan.operators is empty. This should be defined outside the loop
            # and the loop refactored to account for not redeclaring this for each operator
            output_records = []
            op_id = operator.get_op_id()
            prev_op_id = plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None

            if isinstance(operator, DataSourcePhysicalOp):
                continue
            # only invoke aggregate operator(s) once there are no more source records and all
            # upstream operators' processing queues are empty
            # elif isinstance(operator, AggregateOp):
            # output_records, record_op_stats_lst = operator(candidates=input_records)
            elif isinstance(operator, LimitScanOp):
                if self.records_count >= operator.limit:
                    break
            else:
                for r in input_records:
                    record_out, stats = operator(r)
                    output_records += record_out
                    record_op_stats_lst += stats

                if isinstance(operator, FilterOp):
                    # delete all records that did not pass the filter
                    output_records = [r for r in output_records if r.passed_operator]
                    if not output_records:
                        break

            self.plan_stats.operator_stats[op_id].add_record_op_stats(
                record_op_stats_lst,
                source_op_id=prev_op_id,
                plan_id=plan.plan_id,
            )
            input_records = output_records
            self.records_count += len(output_records)

        return output_records
