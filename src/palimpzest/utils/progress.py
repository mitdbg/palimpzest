import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as RichProgress

from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan


@dataclass
class ProgressStats:
    """Statistics tracked for progress reporting"""
    start_time: float = 0.0
    total_cost: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    current_operation: str = ""
    memory_usage_mb: float = 0.0
    recent_text: str = ""

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

# NOTE: right now we only need to support single plan execution; in a multi-plan setting, we will
#       need to modify the semantics of the progress manager to support multiple plans
class ProgressManager(ABC):
    """Abstract base class for progress managers for plan execution"""

    def __init__(self, plan: PhysicalPlan | SentinelPlan, num_samples: int | None = None):
        """
        Initialize the progress manager for the given plan. This function takes in a plan,
        the number of samples to process (if specified).

        If `num_samples` is None, then the entire DataReader will be scanned.

        For each operator which is not an `AggregateOp` or `LimitScanOp`, we set its task `total`
        to the number of inputs to be processed by the plan. As intermediate operators process
        their inputs, the ProgressManager will update the `total` for their downstream operators.
        """
        # initialize progress object
        self.progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            #TextColumn("[green]Success: {task.fields[success]}"),
            #TextColumn("[red]Failed: {task.fields[failed]}"),
            #TextColumn("[cyan]Mem: {task.fields[memory]:.1f}MB"),
            TextColumn("[green]Cost: ${task.fields[cost]:.4f}"),
            TextColumn("\n[white]{task.fields[recent]}"),  # Recent text on new line
            refresh_per_second=10,
            expand=True,   # Use full width
        )

        # initialize mapping from op_id --> ProgressStats
        self.op_id_to_stats: dict[str, ProgressStats] = {}

        # initialize mapping from op_id --> task
        self.op_id_to_task = {}

        # initialize start time
        self.start_time = None

        # create mapping from op_id --> next_op
        self.op_id_to_next_op: dict[str, PhysicalOperator] = {}
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            next_op = plan.operators[op_idx + 1] if op_idx + 1 < len(plan.operators) else None
            self.op_id_to_next_op[op_id] = next_op

        # compute the total number of inputs to be processed by the plan
        datareader_len = len(plan.operators[0].datareader)
        total = datareader_len if num_samples is None else min(num_samples, datareader_len)

        # add a task to the progress manager for each operator in the plan
        for op in plan.operators:
            # get the op id and a short string representation of the op; (str(op) is too long)
            op_id = op.get_op_id()
            op_str = f"{op.op_name()} ({op_id})"

            # update the `total` if we encounter an AggregateOp or LimitScanOp
            if isinstance(op, AggregateOp):
                total = 1
            elif isinstance(op, LimitScanOp):
                total = op.limit

            self.add_task(op_id, op_str, total)

    def get_task_total(self, op_id: str) -> int:
        """Return the current total value for the given task."""
        task = self.op_id_to_task[op_id]
        return self.progress._tasks[task].total

    def get_task_description(self, op_id: str) -> str:
        """Return the current description for the given task."""
        task = self.op_id_to_task[op_id]
        return self.progress._tasks[task].description

    @abstractmethod
    def add_task(self, op_id: str, op_str: str, total: int):
        """Initialize progress tracking for operator execution with total items"""
        pass

    @abstractmethod
    def start(self):
        """Start the progress bar(s)"""
        pass

    @abstractmethod
    def incr(self, op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        """
        Advance the progress bar for the given operator by one. Modify the downstream operators'
        progress bar `total` to reflect the number of outputs produced by this operator.

        NOTE: The semantics of this function are that every time it is executed we advance the
        progress bar by 1. This is because the progress bar represents what fraction of the inputs
        have been processed by the operator. `num_outputs` specifies how many outputs were generated
        by the operator when processing the input for which `incr()` was called. E.g. a filter which
        filters an input record will advance its progress bar by 1, but the next operator will now
        have 1 fewer inputs to process. Alternatively, a convert which generates 3 `num_outputs` will
        increase the inputs for the next operator by `delta = num_outputs - 1 = 2`.
        """
        pass

    @abstractmethod
    def finish(self):
        """Clean up and finalize progress tracking"""
        pass

    def update_stats(self, op_id: str, **kwargs):
        """Update progress statistics"""
        for key, value in kwargs.items():
            if hasattr(self.op_id_to_stats[op_id], key):
                if key != "total_cost":
                    setattr(self.op_id_to_stats[op_id], key, value)
                else:
                    self.op_id_to_stats[op_id].total_cost += value
        self.op_id_to_stats[op_id].memory_usage_mb = get_memory_usage()


class MockProgressManager(ProgressManager):
    """Mock progress manager for testing purposes"""

    def __init__(self, plan: PhysicalPlan | SentinelPlan, num_samples: int | None = None):
        pass

    def add_task(self, op_id: str, op_str: str, total: int):
        pass

    def start(self):
        pass

    def incr(self, op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        pass

    def finish(self):
        pass


class PZProgressManager(ProgressManager):
    """Progress manager for command line interface using rich"""
    
    def __init__(self, plan: PhysicalPlan | SentinelPlan, num_samples: int | None = None):
        super().__init__(plan, num_samples)
        self.console = Console()

    def add_task(self, op_id: str, op_str: str, total: int):
        """Add a new task to the progress bar"""
        task = self.progress.add_task(
            f"[blue]{op_str}", 
            total=total,
            cost=0.0,
            success=0,
            failed=0,
            memory=0.0,
            recent="",
        )

        # store the mapping of operator ID to task ID
        self.op_id_to_task[op_id] = task

        # initialize the stats for this operation
        self.op_id_to_stats[op_id] = ProgressStats(start_time=time.time())

    def start(self):
        # print a newline before starting to separate from previous output
        print()

        # set start time
        self.start_time = time.time()

        # start progress bar
        self.progress.start()

    def incr(self, op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        # get the task for the given operation
        task = self.op_id_to_task.get(op_id)

        # update statistics with any additional keyword arguments
        if kwargs != {}:
            self.update_stats(op_id, **kwargs)

        # update progress bar and recent text in one update
        if display_text is not None:
            self.op_id_to_stats[op_id].recent_text = display_text

        # if num_outputs is not 1, update the downstream operators' progress bar total for any
        # operator which is not an AggregateOp or LimitScanOp
        delta = num_outputs - 1
        if delta != 0:
            next_op = self.op_id_to_next_op[op_id]
            while next_op is not None:
                if not isinstance(next_op, (AggregateOp, LimitScanOp)):
                    next_op_id = next_op.get_op_id()
                    next_task = self.op_id_to_task[next_op_id]
                    self.progress.update(next_task, total=self.get_task_total(next_op_id) + delta)

                next_op = self.op_id_to_next_op[next_op_id]

        # advance the progress bar for this task
        self.progress.update(
            task,
            advance=1,
            description=f"[bold blue]{self.get_task_description(op_id)}",
            cost=self.op_id_to_stats[op_id].total_cost,
            success=self.op_id_to_stats[op_id].success_count,
            failed=self.op_id_to_stats[op_id].failure_count,
            memory=get_memory_usage(),
            recent=f"{self.op_id_to_stats[op_id].recent_text}" if display_text is not None else "",
            refresh=True,
        )

    def finish(self):
        self.progress.stop()

        # compute total cost, success, and failure
        total_cost = sum(stats.total_cost for stats in self.op_id_to_stats.values())
        # success_count = sum(stats.success_count for stats in self.op_id_to_stats.values())
        # failure_count = sum(stats.failure_count for stats in self.op_id_to_stats.values())

        # Print final stats on new lines after progress display
        print(f"Total time: {time.time() - self.start_time:.2f}s")
        print(f"Total cost: ${total_cost:.4f}")
        # print(f"Success rate: {success_count}/{success_count + failure_count}")


def create_progress_manager(
    plan: PhysicalPlan | SentinelPlan,
    num_samples: int | None = None,
    progress: bool = True,
) -> ProgressManager:
    """Factory function to create appropriate progress manager based on environment"""
    if not progress:
        return MockProgressManager(plan, num_samples)
    return PZProgressManager(plan, num_samples)
