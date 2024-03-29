from dataclasses import dataclass

# probably will need multiple types of stats objects
# will likely want to_dict() method for each of them
@dataclass
class Stats:
    """Dataclass for storing statistics captured during the execution of a physical plan."""
    pass


class StatsProcessor:
    """
    This class implements a set of standardized functions for processing profiling statistics
    collected by PZ.
    """

    def __init__(self, stats: Stats) -> None:
        self.stats = stats
