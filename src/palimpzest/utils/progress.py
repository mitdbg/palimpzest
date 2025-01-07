from abc import ABC, abstractmethod
import threading
from typing import Optional

from rich.progress import Progress as RichProgress
from rich.progress import TextColumn, BarColumn, TaskProgressColumn
try:
    import ipywidgets as widgets
    from IPython.display import display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

def in_jupyter_notebook():
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config
    except:
        return False

class ProgressManager(ABC):
    """Abstract base class for progress managers"""
    
    @abstractmethod
    def start(self, total: int):
        """Initialize progress tracking with total items"""
        pass
    
    @abstractmethod
    def update(self, current: int, sample: Optional[str] = None):
        """Update progress with current count and optional sample"""
        pass
    
    @abstractmethod
    def finish(self):
        """Clean up and finalize progress tracking"""
        pass

class CLIProgressManager(ProgressManager):
    """Progress manager for command line interface using rich"""
    
    def __init__(self):
        self.progress = RichProgress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            refresh_per_second=10
        )
        self.task_id = None
        
    def start(self, total: int):
        self.task_id = self.progress.add_task("Processing", total=total)
        self.progress.start()
        
    def update(self, current: int, sample: Optional[str] = None):
        self.progress.update(self.task_id, completed=current)
        if sample:
            print(f"Sample: {sample}")
            
    def finish(self):
        self.progress.stop()

class NotebookProgressManager(ProgressManager):
    """Progress manager for Jupyter notebooks using ipywidgets"""
    
    def __init__(self):
        if not JUPYTER_AVAILABLE:
            raise ImportError("ipywidgets not available. Install with: pip install ipywidgets")
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            description='Processing:',
            bar_style='info',
            orientation='horizontal'
        )
        self.output = widgets.Output()
        
    def start(self, total: int):
        self.progress_bar.max = total
        display(self.progress_bar)
        display(self.output)
        
    def update(self, current: int, sample: Optional[str] = None):
        self.progress_bar.value = current
        if sample:
            with self.output:
                print(f"Sample: {sample}")
                
    def finish(self):
        self.progress_bar.bar_style = 'success'

def create_progress_manager() -> ProgressManager:
    """Factory function to create appropriate progress manager based on environment"""
    if in_jupyter_notebook():
        try:
            return NotebookProgressManager()
        except ImportError:
            return CLIProgressManager()
    return CLIProgressManager() 