import logging
import time

from palimpzest.core.data.dataclasses import PlanStats
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.policy import Policy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.sets import Dataset

logger = logging.getLogger(__name__)

class StreamingQueryProcessor(QueryProcessor):
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
        logger.info("Initialized StreamingQueryProcessor")

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
        # self.clear_cached_examples()
        start_time = time.time()

        # check that the plan does not contain any aggregation operators
        for op in self.plan.operators:
            if isinstance(op, AggregateOp):
                raise Exception("You cannot have a Streaming Execution if there is an Aggregation Operator")

        # TODO: Do we need to re-initialize the optimizer here? 
        # Effectively always use the optimal strategy   
        optimizer = self.optimizer.deepcopy_clean()
        plans = optimizer.optimize(dataset, policy)
        self.plan = plans[0]
        self.plan_stats = PlanStats.from_plan(self.plan)
        self.plan_stats.start()
        logger.info(f"Time for planning: {time.time() - start_time:.2f} seconds")
        self.plan_generated = True
        logger.info(f"Generated plan:\n{self.plan}")
        return self.plan

    def execute(self):
        logger.info("Executing StreamingQueryProcessor")
        # Always delete cache
        if not self.plan_generated:
            self.generate_plan(self.dataset, self.policy)

        # if dry_run:
        #     yield [], self.plan, self.plan_stats
        #     return

        input_records = self.get_input_records()
        for idx, record in enumerate(input_records):
            # print("Iteration number: ", idx+1, "out of", len(input_records))
            output_records = self.execute_opstream(self.plan, record)
            if idx == len(input_records) - 1:
                # finalize plan stats
                self.plan_stats.finish()
            self.plan_stats.plan_str = str(self.plan)
            yield DataRecordCollection(output_records, plan_stats=self.plan_stats)

        logger.info("Done executing StreamingQueryProcessor")


    def get_input_records(self):
        scan_operator = self.plan.operators[0]
        assert isinstance(scan_operator, ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
        datareader = scan_operator.datareader
        if not datareader:
            raise Exception("DataReader not found")
        datareader_len = len(datareader)

        input_records = []
        record_op_stats = []
        for source_idx in range(datareader_len):
            record_set = scan_operator(source_idx)
            input_records += record_set.data_records
            record_op_stats += record_set.record_op_stats

        self.plan_stats.add_record_op_stats(record_op_stats)

        return input_records

    def execute_opstream(self, plan, record):
        # initialize list of output records and intermediate variables
        input_records = [record]
        record_op_stats_lst = []

        for operator in plan.operators:
            # TODO: this being defined in the for loop potentially makes the return
            # unbounded if plan.operators is empty. This should be defined outside the loop
            # and the loop refactored to account for not redeclaring this for each operator
            output_records = []

            if isinstance(operator, ScanPhysicalOp):
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
                    record_set = operator(r)
                    output_records += record_set.data_records
                    record_op_stats_lst += record_set.record_op_stats

                if isinstance(operator, FilterOp):
                    # delete all records that did not pass the filter
                    output_records = [r for r in output_records if r.passed_operator]
                    if not output_records:
                        break

            self.plan_stats.add_record_op_stats(record_op_stats_lst)
            input_records = output_records
            self.records_count += len(output_records)

        return output_records
