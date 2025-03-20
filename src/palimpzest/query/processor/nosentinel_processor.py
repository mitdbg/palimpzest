import logging

from palimpzest.core.data.dataclasses import ExecutionStats
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.processor.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class NoSentinelQueryProcessor(QueryProcessor):
    """
    Query processor that uses naive cost estimates to select the best plan.
    """

    # TODO: Consider to support dry_run.
    def execute(self) -> DataRecordCollection:
        logger.info("Executing NoSentinelQueryProcessor")

        # create execution stats
        execution_stats = ExecutionStats(execution_id=self.execution_id())
        execution_stats.start()

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_best_plan(self.dataset, self.optimizer)

        # update the execution stats to account for the work to execute the final plan
        execution_stats.add_plan_stats(plan_stats)
        execution_stats.finish()

        # construct and return the DataRecordCollection
        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info("Done executing NoSentinelQueryProcessor")

        return result
