import logging
import time

from palimpzest.core.data.dataclasses import ExecutionStats
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.execution.parallel_execution_strategy import ParallelExecutionStrategy
from palimpzest.query.execution.single_threaded_execution_strategy import (
    PipelinedSingleThreadExecutionStrategy,
    SequentialSingleThreadExecutionStrategy,
)
from palimpzest.query.processor.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class NoSentinelQueryProcessor(QueryProcessor):
    """
    Specialized query processor that implements no sentinel strategy
    for coordinating optimization and execution.
    """

    # TODO: Consider to support dry_run.
    def execute(self) -> DataRecordCollection:
        logger.info("Executing NoSentinelQueryProcessor")
        execution_start_time = time.time()

        # if cache is False, make sure we do not re-use codegen examples
        if not self.cache:
            # self.clear_cached_examples()
            pass

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_best_plan(self.dataset, self.policy, self.optimizer)

        # aggregate plan stats
        aggregate_plan_stats = self.aggregate_plan_stats(plan_stats)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats=aggregate_plan_stats,
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(
                list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))
            ),
            plan_strs={plan_id: plan_stats.plan_str for plan_id, plan_stats in aggregate_plan_stats.items()},
        )

        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info("Done executing NoSentinelQueryProcessor")
        logger.debug(f"Result: {result}")
        return result


class NoSentinelSequentialSingleThreadProcessor(NoSentinelQueryProcessor, SequentialSingleThreadExecutionStrategy):
    """
    This class performs non-sample based execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelQueryProcessor.__init__(self, *args, **kwargs)
        SequentialSingleThreadExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            cache=self.cache,
            verbose=self.verbose,
            progress=self.progress,
        )
        self.progress_manager = None
        logger.info("Created NoSentinelSequentialSingleThreadProcessor")


class NoSentinelPipelinedSingleThreadProcessor(NoSentinelQueryProcessor, PipelinedSingleThreadExecutionStrategy):
    """
    This class performs non-sample based execution while executing plans in a pipelined fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelQueryProcessor.__init__(self, *args, **kwargs)
        PipelinedSingleThreadExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            cache=self.cache,
            verbose=self.verbose,
            progress=self.progress,
        )
        self.progress_manager = None
        logger.info("Created NoSentinelPipelinedSingleThreadProcessor")


class NoSentinelParallelProcessor(NoSentinelQueryProcessor, ParallelExecutionStrategy):
    """
    This class performs non-sample based execution while executing plans in a parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelQueryProcessor.__init__(self, *args, **kwargs)
        ParallelExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            cache=self.cache,
            verbose=self.verbose,
            progress=self.progress,
        )
        self.progress_manager = None
        logger.info("Created NoSentinelParallelProcessor")
