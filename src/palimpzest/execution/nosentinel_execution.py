import time

from palimpzest.constants import OptimizationStrategy
from palimpzest.dataclasses import ExecutionStats
from palimpzest.execution.execution_engine import ExecutionEngine
from palimpzest.execution.plan_executors.parallel_plan_execution import (
    PipelinedParallelPlanExecutor,
)
from palimpzest.execution.plan_executors.single_threaded_plan_execution import (
    PipelinedSingleThreadPlanExecutor,
    SequentialSingleThreadPlanExecutor,
)
from palimpzest.optimizer.cost_model import CostModel
from palimpzest.optimizer.optimizer import Optimizer
from palimpzest.policy import Policy
from palimpzest.sets import Set


class NoSentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the execute_plan() method.
    """

    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            self.clear_cached_responses_and_examples()

        # construct the CostModel
        cost_model = CostModel()

        # initialize the optimizer
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
            optimization_strategy=self.optimization_strategy,
        )

        # execute plan(s) according to the optimization strategy
        records, plan_stats = [], []
        if self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            records, plan_stats = self.execute_confidence_interval_strategy(dataset, policy, optimizer)
        
        else:
            records, plan_stats = self.execute_strategy(dataset, policy, optimizer)

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

        return records, execution_stats

class NoSentinelSequentialSingleThreadExecution(NoSentinelExecutionEngine, SequentialSingleThreadPlanExecutor):
    """
    This class performs non-sample based execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelExecutionEngine.__init__(self, *args, **kwargs)
        SequentialSingleThreadPlanExecutor.__init__(self, *args, **kwargs)


class NoSentinelPipelinedSingleThreadExecution(NoSentinelExecutionEngine, PipelinedSingleThreadPlanExecutor):
    """
    This class performs non-sample based execution while executing plans in a pipelined, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelExecutionEngine.__init__(self, *args, **kwargs)
        PipelinedSingleThreadPlanExecutor.__init__(self, *args, **kwargs)


class NoSentinelPipelinedParallelExecution(NoSentinelExecutionEngine, PipelinedParallelPlanExecutor):
    """
    This class performs non-sample based execution while executing plans in a pipelined, parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelExecutionEngine.__init__(self, *args, **kwargs)
        PipelinedParallelPlanExecutor.__init__(self, *args, **kwargs)
