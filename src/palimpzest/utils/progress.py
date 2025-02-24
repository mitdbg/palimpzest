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
from palimpzest.query.optimizer.plan import PhysicalPlan

try:
    import ipywidgets as widgets
    from IPython.display import display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

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

def in_jupyter_notebook():
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config
    except Exception:
        return False

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

class ProgressManager(ABC):
    """Abstract base class for progress managers for plan execution"""

    def __init__(self, plan: PhysicalPlan, num_samples: int | None = None):
        """
        Initialize the progress manager for the given plan. This function takes in a plan,
        the number of samples to process (if specified), and a boolean indicating whether the
        execution of plan operators will be in sequence (as opposed to in a pipeline).

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
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            #TextColumn("[yellow]Cost: ${task.fields[cost]:.4f}"),
            #TextColumn("[green]Success: {task.fields[success]}"),
            #TextColumn("[red]Failed: {task.fields[failed]}"),
            TextColumn("[cyan]Mem: {task.fields[memory]:.1f}MB"),
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

    def get_task_total(self, task) -> int:
        """Return the current total value for the given task."""
        return self.progress._tasks[task].total

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
                setattr(self.op_id_to_stats[op_id], key, value)
        self.op_id_to_stats[op_id].memory_usage_mb = get_memory_usage()

class CLIProgressManager(ProgressManager):
    """Progress manager for command line interface using rich"""
    
    def __init__(self):
        super().__init__()
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

    # TODO: update cost, success, failed
    def incr(self, op_id: str, num_outputs: int = 1, display_text: str | None = None, **kwargs):
        # get the task for the given operation
        task = self.op_id_to_task.get(op_id)

        # update statistics with any additional keyword arguments
        if kwargs != {}:
            self.update_stats(op_id, **kwargs)

        # update progress bar and recent text in one update
        if display_text is not None:
            self.op_id_to_stats[op_id].recent_text = display_text

        # advance the progress bar for this task
        self.progress.update(
            task,
            advance=1,
            description=f"[bold blue]{self.op_id_to_stats[op_id].current_operation}",
            cost=self.op_id_to_stats[op_id].total_cost,
            success=self.op_id_to_stats[op_id].success_count,
            failed=self.op_id_to_stats[op_id].failure_count,
            memory=get_memory_usage(),
            recent=f"Recent: {self.op_id_to_stats[op_id].recent_text}",
        )

        # if num_outputs is not 1, update the downstream operators' progress bar total for any
        # operator which is not an AggregateOp or LimitScanOp
        delta = num_outputs - 1
        if delta != 0:
            next_op = self.op_id_to_next_op[op_id]
            while next_op is not None:
                if not isinstance(next_op, (AggregateOp, LimitScanOp)):
                    next_op_id = next_op.get_op_id()
                    next_task = self.op_id_to_task[next_op_id]
                    self.progress.update(next_task, total=self.get_task_total(next_task) + delta)

                next_op = self.op_id_to_next_op[next_op_id]

    def finish(self):
        self.progress.stop()

        # compute total cost, success, and failure
        total_cost = sum(stats.total_cost for stats in self.op_id_to_stats.values())
        success_count = sum(stats.success_count for stats in self.op_id_to_stats.values())
        failure_count = sum(stats.failure_count for stats in self.op_id_to_stats.values())

        # Print final stats on new lines after progress display
        print(f"Total time: {time.time() - self.start_time:.2f}s")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Success rate: {success_count}/{success_count + failure_count}")


# TODO: do we need this?
class NotebookProgressManager(ProgressManager):
    """Progress manager for Jupyter notebooks using ipywidgets"""
    
    def __init__(self, plan: PhysicalPlan, num_samples: int | None = None):
        super().__init__()
        if not JUPYTER_AVAILABLE:
            raise ImportError("ipywidgets not available. Install with: pip install ipywidgets")
            
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            description='Processing:',
            bar_style='info',
            orientation='horizontal'
        )
        
        self.stats_html = widgets.HTML(
            value="<pre>Initializing...</pre>"
        )
        
        self.recent_html = widgets.HTML(
            value="<pre>Recent: </pre>"
        )
        
        self.container = widgets.VBox([
            self.progress_bar,
            self.stats_html,
            self.recent_html
        ])
        
    def start(self, total: int):
        self.progress_bar.max = total
        display(self.container)
        
    def update(self, current: int, sample: str | None = None, **kwargs):
        self.update_stats(**kwargs)
        self.progress_bar.value = current
        
        # Update stats display
#        Total Cost: ${self.stats.total_cost:.4f}
#        Success/Total: {self.stats.success_count}/{self.stats.success_count + self.stats.failure_count}


        stats_text = f"""
        <pre>
        Operation: {self.stats.current_operation}
        Time Elapsed: {time.time() - self.stats.start_time:.1f}s
        Memory Usage: {self.stats.memory_usage_mb:.1f}MB
        </pre>
        """
        self.stats_html.value = stats_text
        
        # Update recent text
        if sample:
            self.stats.recent_text = sample
        self.recent_html.value = f"<pre>Recent: {self.stats.recent_text}</pre>"
                
    def finish(self):
        self.progress_bar.bar_style = 'success'
        #
        # Total Cost: ${self.stats.total_cost:.4f}
        # Final Success Rate: {self.stats.success_count}/{self.stats.success_count + self.stats.failure_count}
        stats_text = f"""
        <pre>
        Completed!
        Total Time: {time.time() - self.stats.start_time:.1f}s
        Peak Memory Usage: {self.stats.memory_usage_mb:.1f}MB
        </pre>
        """
        self.stats_html.value = stats_text
        self.recent_html.value = "<pre>Completed</pre>"

def create_progress_manager(plan: PhysicalPlan, num_samples: int | None = None) -> ProgressManager:
    """Factory function to create appropriate progress manager based on environment"""
    if in_jupyter_notebook():
        try:
            return NotebookProgressManager(plan, num_samples)
        except ImportError:
            return CLIProgressManager(plan, num_samples)
    return CLIProgressManager(plan, num_samples)
