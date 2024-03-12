from __future__ import annotations

from palimpzest.constants import MODEL_CARDS

from functools import wraps
from typing import Any, Dict

import os
import time

# DEFINITIONS
PZ_PROFILING_ENV_VAR = "PZ_PROFILING"
# IteratorFn = Callable[[], DataRecord]

class Profiler:
    """
    This class maintains a set of utility tools and functions to help
    with profiling PZ programs.
    """

    def __init__(self):
        # dictionary with aggregate statistics for this operator
        self.agg_operator_stats = {
            # total number of records returned by the iterator for this operator
            "total_records": 0,

            # total time spent in this iterator; this will include time spent in input operators
            "total_iter_time": 0.0,

            # usage statistics computed for induce and filter operations
            "total_input_tokens": 0,
            "total_output_tokens": 0,

            # time spent waiting for API calls to return
            "total_api_call_duration": 0.0,

            # keep track of finish reasons and prompts used
            "finish_reasons": {},
            "prompts": {},

            # keep track of the total time spent inside of the profiler
            "total_time_in_profiler": 0.0,
        }

        # list of records computed by this operator; each record will have
        # its individual stats and computation history stored
        self.records = []

    def _update_agg_stats(self, stats: Dict[str, Any], field_name: str=None) -> None:
        """
        Given the stats dictionary from either (1) a `run_cot_bool` function call or
        (2) a single field in a `run_cot_qa` function all -- update the aggregate stats
        in place. When a `field_name` is provided, we update the total aggregate stats
        as well as the per-field aggregate stats.
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

        prompt = stats["prompt"]
        if prompt in self.agg_operator_stats["prompts"]:
            self.agg_operator_stats["prompts"][prompt] += 1
        else:
            self.agg_operator_stats["prompts"][prompt] = 1

        if field_name is not None:
            # initialize aggregate field stats sub-dictionary if not already present
            if field_name not in self.agg_operator_stats:
                self.agg_operator_stats[field_name] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_api_call_duration": 0.0,
                    "finish_reasons": {},
                    "prompts": {},
                }

            # update field aggregate statistics
            self.agg_operator_stats[field_name]["total_input_tokens"] += stats["usage"]["prompt_tokens"]
            finish_reason = stats["finish_reason"]
            if finish_reason in self.agg_operator_stats[field_name]["finish_reasons"]:
                self.agg_operator_stats[field_name]["finish_reasons"][finish_reason] += 1
            else:
                self.agg_operator_stats[field_name]["finish_reasons"][finish_reason] = 1

            prompt = stats["prompt"]
            if prompt in self.agg_operator_stats[field_name]["prompts"]:
                self.agg_operator_stats[field_name]["prompts"][prompt] += 1
            else:
                self.agg_operator_stats[field_name]["prompts"][prompt] = 1

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
            "per_record_stats": self.records,
        }

        return full_stats_dict

    def iter_profiler(self, name: str, op_id: str):
        """
        iter_profiler is a decorator factory. This function takes in a `name` argument
        and returns a decorator which will decorate an iterator. In practice, this
        looks almost identical to how you would normally use a decorator. The only
        difference is that now we can use the `name` inside of our decorated function
        to identify the profiling information on a per-iterator basis.

        To use the profiler, simply apply it to an iterator as follows:

        @profiler(name="foo", op_id="some-logical-op")
        def someIterator():
            # do normal iterator things
            yield dr
        """
        def profile_decorator(iterator):
            # return iterator if profiling is not set to True
            if not Profiler.profiling_on():
                return iterator

            @wraps(iterator)
            def timed_iterator():
                t_start = time.time()
                for record in iterator():
                    t_end = time.time()
                    self.agg_operator_stats["total_records"] += 1

                    # for non-LLM workload operators, we need to add the op_id
                    if op_id not in record._stats:
                        record._stats[op_id] = {}

                    # add name and time spent in iteration for operator
                    record._stats[op_id]["name"] = name
                    record._stats[op_id]["iter_time"] = t_end - t_start

                    # add transformation to history of computation if this is an induce
                    if "induce" in name:
                        record._history[op_id] = {
                            "name": name,
                            "record_state": record.asTextJSON(serialize=True),
                        }

                    # add record to set of records computed by this operator
                    self.records.append(record._history)

                    # TODO: make this if-condition less hacky
                    # update aggregate stats for operators with LLM workloads
                    if "filter" in name and "usage" in record._stats[op_id]:
                        stats = dict(record._stats[op_id])
                        self._update_agg_stats(stats)

                    elif "induce" in name and "usage" in record._stats[op_id]:
                        stats = dict(record._stats[op_id])
                        for field_name, field_stats in stats.items():
                            self._update_agg_stats(field_stats, field_name=field_name)

                    # if this is a filter operation and the record did not pass the filter,
                    # then this record is meant to be filtered out, so do not yield it
                    if "filter" in name and record._passed_filter is False:
                        t_start = time.time()
                        self.agg_operator_stats["total_time_in_profiler"] += t_start - t_end
                        continue

                    yield record

                    # start timer for next iteration
                    t_start = time.time()
                    self.agg_operator_stats["total_time_in_profiler"] += t_start - t_end

                # compute total time for iterator to finish
                t_final = time.time()
                self.agg_operator_stats["total_iter_time"] = t_final - t_start

            return timed_iterator
        return profile_decorator
