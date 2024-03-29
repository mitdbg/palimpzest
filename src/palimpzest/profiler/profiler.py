from __future__ import annotations

from palimpzest.constants import MODEL_CARDS
from palimpzest.profiler import Stats, AggOperatorStats, FilterLLMStats, InduceLLMStats

from functools import wraps
from typing import Any, Dict

import os
import time

# DEFINITIONS
PZ_PROFILING_ENV_VAR = "PZ_PROFILING"
# IteratorFn = Callable[[], DataRecord]

class Profiler:
    """
    This class maintains a set of utility tools and functions to help with profiling PZ programs.
    """

    def __init__(self, op_id: str):
        # store op_id for the operation associated with this profiler
        self.op_id = op_id

        # dictionary with aggregate statistics for this operator
        self.agg_operator_stats = AggOperatorStats(op_id=self.op_id)

        # list of records' states after being operated on by the operator associated with
        # this profiler; each record will have its individual stats and state stored;
        # we can then get the lineage / history of computation by looking at records'
        # _state across computations
        self.records = []

    def _update_agg_stats(self, stats: Stats) -> None:
        """
        Given the stats object for a single record which was operated on by this operator,
        update the aggregate stats for this operator in place.
        """
        # update aggregate statistics
        self.agg_operator_stats["total_input_tokens"] += stats["usage"]["prompt_tokens"]
        self.agg_operator_stats["total_output_tokens"] += stats["usage"]["completion_tokens"]
        self.agg_operator_stats["total_api_call_duration"] += stats["api_call_duration"]

        finish_reason = stats["finish_reason"]
        if finish_reason in self.agg_operator_stats["finish_reasons"]:
            self.agg_operator_stats["finish_reasons"][finish_reason] += 1
        else:
            self.agg_operator_stats["finish_reasons"][finish_reason] = 1

        if field_name is not None:
            # initialize aggregate field stats sub-dictionary if not already present
            if field_name not in self.agg_operator_stats:
                self.agg_operator_stats[field_name] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_api_call_duration": 0.0,
                    "finish_reasons": {},
                }

            # update field aggregate statistics
            self.agg_operator_stats[field_name]["total_input_tokens"] += stats["usage"]["prompt_tokens"]
            self.agg_operator_stats[field_name]["total_output_tokens"] += stats["usage"]["completion_tokens"]
            self.agg_operator_stats[field_name]["total_api_call_duration"] += stats["api_call_duration"]
            finish_reason = stats["finish_reason"]
            if finish_reason in self.agg_operator_stats[field_name]["finish_reasons"]:
                self.agg_operator_stats[field_name]["finish_reasons"][finish_reason] += 1
            else:
                self.agg_operator_stats[field_name]["finish_reasons"][finish_reason] = 1

    @staticmethod
    def profiling_on() -> bool:
        """
        Returns a boolean indicating whether user has turned on profiling.
        """
        return os.getenv(PZ_PROFILING_ENV_VAR) is not None and os.getenv(PZ_PROFILING_ENV_VAR).lower() == "true"

    def get_data(self, model_name: str=None) -> Dict[str, Any]:
        """
        Compute and return the aggregate operator statistics as well as the per-record statistics.
        """
        # compute final aggregate operator stats
        total_iter_time = self.agg_operator_stats["total_iter_time"]
        total_records = self.agg_operator_stats["total_records"]
        self.agg_operator_stats["avg_record_iter_time"] = total_iter_time / total_records

        # compute total usd spent if using induce or filter operation
        if model_name is not None:
            usd_per_input_token = MODEL_CARDS[model_name]["usd_per_input_token"]
            usd_per_output_token = MODEL_CARDS[model_name]["usd_per_output_token"]
            self.agg_operator_stats["total_input_usd"] = usd_per_input_token * self.agg_operator_stats["total_input_tokens"]
            self.agg_operator_stats["total_output_usd"] = usd_per_output_token * self.agg_operator_stats["total_output_tokens"]
            self.agg_operator_stats["total_usd"] = self.agg_operator_stats["total_input_usd"] + self.agg_operator_stats["total_output_usd"]

        # prepare final dictionary and return
        full_stats_dict = {
            "agg_operator_stats": self.agg_operator_stats,
            "records": self.records,
        }

        return full_stats_dict

    def iter_profiler(self, name: str, shouldProfile: bool = False):
        """
        iter_profiler is a decorator factory. This function takes in a `name` argument
        and returns a decorator which will decorate an iterator. In practice, this
        looks almost identical to how you would normally use a decorator. The only
        difference is that now we can use the `name` inside of our decorated function
        to identify the profiling information on a per-iterator basis.

        To use the profiler, simply apply it to an iterator as follows:

        @profiler(name="foo", shouldProfile=True)
        def someIterator():
            # do normal iterator things
            yield dr
        """
        def profile_decorator(iterator):
            # return iterator if profiling is not set to True
            if not shouldProfile:
                return iterator

            @wraps(iterator)
            def timed_iterator():
                self.agg_operator_stats.op_name = name
                t_record_start = time.time()
                for record in iterator():
                    t_record_end = time.time()
                    self.agg_operator_stats.total_records += 1

                    # for non-induce/filter operators, we need to create an empty Stats object for the op_id
                    if self.op_id not in record._stats:
                        record._stats[self.op_id] = Stats()

                    # add time spent waiting for iterator to yield record; this measures the
                    # time spent by the record in this operator and all source operators
                    record._stats[self.op_id].cumulative_iter_time = t_record_end - t_record_start

                    # update state of record for complete history of computation
                    record._state[self.op_id] = {
                        "name": name,
                        "uuid": record.uuid,
                        "parent_uuid": record.parent_uuid,
                        "stats": record._stats[self.op_id],
                        "record_state": record.asDict(include_bytes=False),
                    }

                    # add record state to set of records computed by this operator
                    self.records.append(record._state[self.op_id])

                    # update aggregate stats for operators with LLM workloads
                    self._update_agg_stats(stats, record._stats[self.op_id])

                    # track time spent in profiler
                    self.agg_operator_stats.total_time_in_profiler += time.time() - t_record_end

                    # if this is a filter operation and the record did not pass the filter,
                    # then this record is meant to be filtered out, so do not yield it
                    if "filter" in name and record._passed_filter is False:
                        t_record_start = time.time()
                        continue

                    yield record

                    # start timer for next iteration
                    t_record_start = time.time()

            return timed_iterator
        return profile_decorator
