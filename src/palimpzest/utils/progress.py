import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from chromadb.api.models.Collection import Collection
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
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
from rich.table import Table

from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
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

        # initialize mapping from full_op_id --> ProgressStats
        self.full_op_id_to_stats: dict[str, ProgressStats] = {}

        # initialize mapping from full_op_id --> task
        self.full_op_id_to_task = {}

        # initialize start time
        self.start_time = None

        # create mapping from full_op_id --> next_op
        self.full_op_id_to_next_op: dict[str, PhysicalOperator] = {}
        for op_idx, op in enumerate(plan.operators):
            full_op_id = op.get_full_op_id()
            next_op = plan.operators[op_idx + 1] if op_idx + 1 < len(plan.operators) else None
            self.full_op_id_to_next_op[full_op_id] = next_op

        # compute the total number of inputs to be processed by the plan
        datareader_len = len(plan.operators[0].datareader)
        total = datareader_len if num_samples is None else min(num_samples, datareader_len)

        # add a task to the progress manager for each operator in the plan
        for op in plan.operators:
            # get the op id and a short string representation of the op; (str(op) is too long)
            op_str = f"{op.op_name()} ({op.get_op_id()})"

            # update the `total` if we encounter an AggregateOp or LimitScanOp
            if isinstance(op, AggregateOp):
                total = 1
            elif isinstance(op, LimitScanOp):
                total = op.limit

            self.add_task(op.get_full_op_id(), op_str, total)

    def get_task_total(self, full_op_id: str) -> int:
        """Return the current total value for the given task."""
        task = self.full_op_id_to_task[full_op_id]
        return self.progress._tasks[task].total

    def get_task_description(self, full_op_id: str) -> str:
        """Return the current description for the given task."""
        task = self.full_op_id_to_task[full_op_id]
        return self.progress._tasks[task].description

    @abstractmethod
    def add_task(self, full_op_id: str, op_str: str, total: int):
        """Initialize progress tracking for operator execution with total items"""
        pass

    @abstractmethod
    def start(self):
        """Start the progress bar(s)"""
        pass

    @abstractmethod
    def incr(self, full_op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
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


class MockProgressManager(ProgressManager):
    """Mock progress manager for testing purposes"""

    def __init__(self, plan: PhysicalPlan | SentinelPlan, num_samples: int | None = None):
        pass

    def add_task(self, full_op_id: str, op_str: str, total: int):
        pass

    def start(self):
        pass

    def incr(self, full_op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        pass

    def finish(self):
        pass

class PZProgressManager(ProgressManager):
    """Progress manager for command line interface using rich"""
    
    def __init__(self, plan: PhysicalPlan, num_samples: int | None = None):
        super().__init__(plan, num_samples)
        self.console = Console()

    def add_task(self, full_op_id: str, op_str: str, total: int):
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
        self.full_op_id_to_task[full_op_id] = task

        # initialize the stats for this operation
        self.full_op_id_to_stats[full_op_id] = ProgressStats(start_time=time.time())

    def start(self):
        # print a newline before starting to separate from previous output
        print()

        # set start time
        self.start_time = time.time()

        # start progress bar
        self.progress.start()

    def incr(self, full_op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        # get the task for the given operation
        task = self.full_op_id_to_task.get(full_op_id)

        # update statistics with any additional keyword arguments
        if kwargs != {}:
            self.update_stats(full_op_id, **kwargs)

        # update progress bar and recent text in one update
        if display_text is not None:
            self.full_op_id_to_stats[full_op_id].recent_text = display_text

        # if num_outputs is not 1, update the downstream operators' progress bar total for any
        # operator which is not an AggregateOp or LimitScanOp
        delta = num_outputs - 1
        if delta != 0:
            next_op = self.full_op_id_to_next_op[full_op_id]
            while next_op is not None:
                if not isinstance(next_op, (AggregateOp, LimitScanOp)):
                    next_full_op_id = next_op.get_full_op_id()
                    next_task = self.full_op_id_to_task[next_full_op_id]
                    self.progress.update(next_task, total=self.get_task_total(next_full_op_id) + delta)

                next_op = self.full_op_id_to_next_op[next_full_op_id]

        # advance the progress bar for this task
        self.progress.update(
            task,
            advance=1,
            description=f"[bold blue]{self.get_task_description(full_op_id)}",
            cost=self.full_op_id_to_stats[full_op_id].total_cost,
            success=self.full_op_id_to_stats[full_op_id].success_count,
            failed=self.full_op_id_to_stats[full_op_id].failure_count,
            memory=get_memory_usage(),
            recent=f"{self.full_op_id_to_stats[full_op_id].recent_text}" if display_text is not None else "",
            refresh=True,
        )

    def finish(self):
        self.progress.stop()

        # compute total cost, success, and failure
        total_cost = sum(stats.total_cost for stats in self.full_op_id_to_stats.values())
        # success_count = sum(stats.success_count for stats in self.full_op_id_to_stats.values())
        # failure_count = sum(stats.failure_count for stats in self.full_op_id_to_stats.values())

        # Print final stats on new lines after progress display
        print(f"Total time: {time.time() - self.start_time:.2f}s")
        print(f"Total cost: ${total_cost:.4f}")
        # print(f"Success rate: {success_count}/{success_count + failure_count}")

    def update_stats(self, full_op_id: str, **kwargs):
        """Update progress statistics"""
        for key, value in kwargs.items():
            if hasattr(self.full_op_id_to_stats[full_op_id], key):
                if key != "total_cost":
                    setattr(self.full_op_id_to_stats[full_op_id], key, value)
                else:
                    self.full_op_id_to_stats[full_op_id].total_cost += value
        self.full_op_id_to_stats[full_op_id].memory_usage_mb = get_memory_usage()

class PZSentinelProgressManager(ProgressManager):
    def __init__(self, plan: SentinelPlan, sample_budget: int):
        # overall progress bar
        self.overall_progress = RichProgress(
            SpinnerColumn(),
            TextColumn("{task.description}"),  # TODO: fixed string?
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[green]Cost: ${task.fields[cost]:.4f}"),
            TextColumn("\n[white]{task.fields[recent]}"),  # Recent text on new line
            refresh_per_second=10,
            expand=True,   # Use full width
        )
        self.overall_task_id = self.overall_progress.add_task("", total=sample_budget, cost=0.0, recent="")

        # logical operator progress bars
        self.op_progress = RichProgress(
            SpinnerColumn(),
            "{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]Cost: ${task.fields[cost]:.4f}"),
            TextColumn("\n[white]{task.fields[recent]}"),  # Recent text on new line
            refresh_per_second=10,
            expand=True,   # Use full width
        )

        # organize progress bars into nice display
        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(self.op_progress, title="[b]Sample Allocation", border_style="red", padding=(1, 2)),
        )
        self.progress_table.add_row(
            Panel.fit(
                self.overall_progress, title="Optimization Progress", border_style="green", padding=(2, 2)
            )
        )
        self.live_display = Live(self.progress_table, refresh_per_second=10)

        # initialize mapping from logical_op_id --> ProgressStats
        self.logical_op_id_to_stats: dict[str, ProgressStats] = {}

        # initialize mapping from logical_op_id --> task
        self.logical_op_id_to_task = {}

        # initialize start time
        self.start_time = None

        # add a task to the progress manager for each operator in the plan
        for logical_op_id, op_set in plan:
            physical_op = op_set[0]
            is_llm_convert = isinstance(physical_op, LLMConvert)
            is_llm_filter = isinstance(physical_op, LLMFilter)
            op_name = "LLMConvert" if is_llm_convert else "LLMFilter" if is_llm_filter else physical_op.op_name()
            op_str = f"{op_name} ({logical_op_id})"
            total = sample_budget if self._is_llm_op(op_set[0]) else 0
            self.add_task(logical_op_id, op_str, total)

        self.console = Console()

    def _is_llm_op(self, physical_op: PhysicalOperator) -> bool:
        is_llm_convert = isinstance(physical_op, LLMConvert)
        is_llm_filter = isinstance(physical_op, LLMFilter)
        is_llm_retrieve = isinstance(physical_op, RetrieveOp) and isinstance(physical_op.index, Collection)
        return is_llm_convert or is_llm_filter or is_llm_retrieve

    def get_task_description(self, logical_op_id: str) -> str:
        """Return the current description for the given task."""
        task = self.logical_op_id_to_task[logical_op_id]
        return self.op_progress._tasks[task].description

    def add_task(self, logical_op_id: str, op_str: str, total: int):
        """Add a new task to the op progress bars"""
        task = self.op_progress.add_task(
            f"[blue]{op_str}", 
            total=total,
            cost=0.0,
            success=0,
            failed=0,
            memory=0.0,
            recent="",
        )

        # store the mapping of operator ID to task ID
        self.logical_op_id_to_task[logical_op_id] = task

        # initialize the stats for this operation
        self.logical_op_id_to_stats[logical_op_id] = ProgressStats(start_time=time.time())

    def start(self):
        # print a newline before starting to separate from previous output
        print()

        # set start time
        self.start_time = time.time()

        # start progress bars
        self.live_display.start()

    def incr(self, logical_op_id: str, num_samples: int, display_text: str | None = None, **kwargs):
        # TODO: (above) organize progress bars into a Live / Table / Panel or something
        # get the task for the given operation
        task = self.logical_op_id_to_task.get(logical_op_id)

        # update statistics with any additional keyword arguments
        if kwargs != {}:
            self.update_stats(logical_op_id, **kwargs)

        # update progress bar and recent text in one update
        if display_text is not None:
            self.logical_op_id_to_stats[logical_op_id].recent_text = display_text

        # advance the op progress bar for this logical_op_id
        self.op_progress.update(
            task,
            advance=num_samples,
            description=f"[bold blue]{self.get_task_description(logical_op_id)}",
            cost=self.logical_op_id_to_stats[logical_op_id].total_cost,
            success=self.logical_op_id_to_stats[logical_op_id].success_count,
            failed=self.logical_op_id_to_stats[logical_op_id].failure_count,
            memory=get_memory_usage(),
            recent=f"{self.logical_op_id_to_stats[logical_op_id].recent_text}" if display_text is not None else "",
            refresh=True,
        )

        # advance the overall progress bar
        self.overall_progress.update(
            self.overall_task_id,
            advance=num_samples,
            cost=sum(stats.total_cost for _, stats in self.logical_op_id_to_stats.items()),
            refresh=True,
        )

        # force the live display to refresh
        self.live_display.refresh()

    def finish(self):
        self.live_display.stop()

        # compute total cost, success, and failure
        total_cost = sum(stats.total_cost for stats in self.logical_op_id_to_stats.values())
        # success_count = sum(stats.success_count for stats in self.logical_op_id_to_stats.values())
        # failure_count = sum(stats.failure_count for stats in self.logical_op_id_to_stats.values())

        # Print final stats on new lines after progress display
        print(f"Total opt. time: {time.time() - self.start_time:.2f}s")
        print(f"Total opt. cost: ${total_cost:.4f}")
        # print(f"Success rate: {success_count}/{success_count + failure_count}")

    def update_stats(self, logical_op_id: str, **kwargs):
        """Update progress statistics"""
        for key, value in kwargs.items():
            if hasattr(self.logical_op_id_to_stats[logical_op_id], key):
                if key != "total_cost":
                    setattr(self.logical_op_id_to_stats[logical_op_id], key, value)
                else:
                    self.logical_op_id_to_stats[logical_op_id].total_cost += value
        self.logical_op_id_to_stats[logical_op_id].memory_usage_mb = get_memory_usage()

def create_progress_manager(
    plan: PhysicalPlan | SentinelPlan,
    num_samples: int | None = None,
    sample_budget: int | None = None,
    progress: bool = True,
) -> ProgressManager:
    """Factory function to create appropriate progress manager based on environment"""
    if not progress:
        return MockProgressManager(plan, num_samples)

    if isinstance(plan, SentinelPlan):
        assert sample_budget is not None, "Sample budget must be specified for SentinelPlan progress manager"
        return PZSentinelProgressManager(plan, sample_budget)

    return PZProgressManager(plan, num_samples)
