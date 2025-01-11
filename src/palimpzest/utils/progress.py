import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as RichProgress

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
    """Abstract base class for progress managers"""
    
    def __init__(self):
        self.stats = ProgressStats(start_time=time.time())
    
    @abstractmethod
    def start(self, total: int):
        """Initialize progress tracking with total items"""
        pass
    
    @abstractmethod
    def update(self, current: int, sample: str | None = None, **kwargs):
        """Update progress with current count and optional sample"""
        pass
    
    @abstractmethod
    def finish(self):
        """Clean up and finalize progress tracking"""
        pass

    def update_stats(self, **kwargs):
        """Update progress statistics"""
        for key, value in kwargs.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)
        self.stats.memory_usage_mb = get_memory_usage()

class CLIProgressManager(ProgressManager):
    """Progress manager for command line interface using rich"""
    
    def __init__(self):
        super().__init__()
        self.console = Console()
        
        # Create single progress bar that includes both progress and recent text
        self.progress = RichProgress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            #TextColumn("[yellow]Cost: ${task.fields[cost]:.4f}"),
            #TextColumn("[green]Success: {task.fields[success]}"),
            #TextColumn("[red]Failed: {task.fields[failed]}"),
            TextColumn("[cyan]Mem: {task.fields[memory]:.1f}MB"),
            TextColumn("\n[white]{task.fields[recent]}"),  # Recent text on new line
            refresh_per_second=10,
            expand=True,   # Use full width
        )
        self.task_id = None
        
    def start(self, total: int):
        # Print a newline before starting to separate from previous output
        print()
        
        self.task_id = self.progress.add_task(
            "Processing", 
            total=total,
            cost=0.0,
            success=0,
            failed=0,
            memory=0.0,
            recent=""
        )
        
        # Start progress bar
        self.progress.start()
        
    def update(self, current: int, sample: str | None = None, **kwargs):
        self.update_stats(**kwargs)
        
        # Update progress bar and recent text in one update
        if sample:
            self.stats.recent_text = sample
            
        self.progress.update(
            self.task_id, 
            completed=current,
            description=f"[bold blue]{self.stats.current_operation}",
            cost=self.stats.total_cost,
            success=self.stats.success_count,
            failed=self.stats.failure_count,
            memory=self.stats.memory_usage_mb,
            recent=f"Recent: {self.stats.recent_text}"
        )
        
    def finish(self):
        self.progress.stop()
        
        # Print final stats on new lines after progress display
        print(f"Total time: {time.time() - self.stats.start_time:.2f}s")
        #print(f"Total cost: ${self.stats.total_cost:.4f}")
        print(f"Success rate: {self.stats.success_count}/{self.stats.success_count + self.stats.failure_count}")

class NotebookProgressManager(ProgressManager):
    """Progress manager for Jupyter notebooks using ipywidgets"""
    
    def __init__(self):
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

def create_progress_manager() -> ProgressManager:
    """Factory function to create appropriate progress manager based on environment"""
    if in_jupyter_notebook():
        try:
            return NotebookProgressManager()
        except ImportError:
            return CLIProgressManager()
    return CLIProgressManager() 