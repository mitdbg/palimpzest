from __future__ import annotations

from palimpzest.constants import PZ_PROFILING_ENV_VAR
from palimpzest.elements import DataRecord

from functools import wraps
from typing import Callable

import os
import time

# DEFINITIONS
IteratorFn = Callable[[], DataRecord]

class profiler:
    """
    This class maintains a set of utility tools and functions to help
    with profiling PZ programs.
    """

    def __init__(self):
        pass

    @property
    def is_profiling(self):
        profile_pz = os.getenv(PZ_PROFILING_ENV_VAR)
        return profile_pz is not None and profile_pz.lower() == "true"

    @staticmethod
    def iter_profiler(name: str):
        """
        iter_profiler is a decorator factory. This function takes in a `name` argument
        and returns a decorator which will decorate an iterator. In practice, this
        looks almost identical to how you would normally use a decorator. The only
        difference is that now we can use the `name` inside of our decorated function
        to identify the profiling information on a per-iterator basis.

        To use the profiler, simply apply it to an iterator as follows:

        @profiler(name="foo")
        def someIterator():
            # do normal iterator things
            yield dr
        """
        def profile_decorator(iterator: IteratorFn) -> IteratorFn:
            # return iterator if profiling is not set to True
            if not profiler.is_profiling:
                return iterator

            # TODO: need to handle parallel iterators differently b/c all of
            #       the time will be spent waiting for the first tuple and then
            #       subsequent tuples will come immediately afterwards.
            #
            #       the history dict. is actually good;
            #
            #       most of the info we want to capture is in the _attemptMapping()
            #       and/or _passesFilter() fcn. call(s); we need to have logic in those
            #       functions that adds info to `record` directly in order to preserve
            #       property that we don't need to change iterator function bodies/signatures
            #       in order to handle profiling.
            @wraps(iterator)
            def timed_iterator():
                t_start = time.time()
                for idx, record in enumerate(iterator()):
                    t_end = time.time()

                    # capture time spent in iteration for operator
                    record._stats[f"{name}_iter_time"] = t_end - t_start

                    # add transformation to history of computation if this is an induce
                    if "induce" in name:
                        record._history[f"{name}"] = record.asJSON()

                    yield record

                    # start timer for next iteration
                    t_start = time.time()

            return timed_iterator
        return profile_decorator