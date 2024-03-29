from __future__ import annotations

from palimpzest.constants import MODEL_CARDS
from palimpzest.profiler.stats import *

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


    @staticmethod
    def profiling_on() -> bool:
        """
        Returns a boolean indicating whether user has turned on profiling.
        """
        return os.getenv(PZ_PROFILING_ENV_VAR) is not None and os.getenv(PZ_PROFILING_ENV_VAR).lower() == "true"


    def get_data(self) -> Dict[str, Any]:
        """
        Return the aggregate operator statistics as well as the per-record statistics.
        """
        # prepare final dictionary and return
        full_stats_dict = {
            "agg_operator_stats": self.agg_operator_stats.to_dict(),
            "records": self.records,
        }

        return full_stats_dict


    def _update_gen_stats(self, gen_stats: GenerationStats, field_name: str=None, code_gen_step: str=None) -> None:
        """
        This helper function takes a GenerationStats object and updates the Profiler's
        agg_operator_stats field (which is an AggOperatorStats object).

        One important detail is that for conventional queries, we create a GenerationStats
        object per-record and per-field. Thus, for these queries we not only want to keep
        track of the aggregate stats across all generated fields, but we also want to keep
        track of the aggregate stats on a per-field basis. (This is worth tracking because
        certain fields can be significantly more expensive to generate than others).

        Because of this, the AggOperatorStats object has a recursively defined field
        called `per_field_agg_op_stats`, which is a dictionary mapping field names
        to an AggOperatorStats object which tracks the aggregate stats for generating
        that specific field.

        So to summarize:
        - AggOperatorStats has a set of top-level fields which compute aggregates across
            all fields AND all records for the given operation.
            - For all but one operation, these are the only aggregates we compute

        - AggOperatorStats also has a top-level field called `per_field_agg_op_stats` which
            stores aggregates for each generated field, where the aggregation is only across all records.
            - `per_field_agg_op_stats` will only ever be filled for induce operations with a conventional query

        Finally, we have a similar recursive structure for our Code Generation statistics.
        AggOperatorStats has a top-level field called `code_gen_agg_op_stats` which tracks
        aggregate statistics across: initial code generation, advice generation, and advised
        code generation. This is conceptually identical to what we just described with
        per-field aggregates in `per_field_agg_op_stats`, except instead of breaking down
        the statistics per-field we break them down per code generation step.
        """
        # If this method is invoked without a field_name or code_gen_step (i.e. field_name=None
        # and code_gen_step=None), then we simply update the top-level aggregate stats
        # (i.e. the aggregates across all LLM generations)
        agg_op_stats = self.agg_operator_stats

        # If the method is invoked with a field_name, then we update the `per_field_agg_op_stats` for that field
        if field_name is not None:
            agg_op_stats = self.agg_operator_stats.per_field_agg_op_stats[field_name]

        # If the method is invoked with a code_gen_step, then we update the `code_gen_agg_op_stats` for that step
        if code_gen_step is not None:
            agg_op_stats = self.agg_operator_stats.code_gen_agg_op_stats[code_gen_step]

        # update timing and token stats
        agg_op_stats.total_llm_call_duration += gen_stats.llm_call_duration_secs
        agg_op_stats.total_input_tokens += gen_stats.usage["prompt_tokens"]
        agg_op_stats.total_output_tokens += gen_stats.usage["completion_tokens"]

        # compute and update USD cost of generation
        usd_per_input_token = MODEL_CARDS[gen_stats.model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[gen_stats.model_name]["usd_per_output_token"]
        agg_op_stats.total_input_usd += gen_stats.usage["prompt_tokens"] * usd_per_input_token
        agg_op_stats.total_output_usd += gen_stats.usage["prompt_tokens"] * usd_per_output_token
        agg_op_stats.total_usd = agg_op_stats.total_input_usd + agg_op_stats.total_output_usd

        # update distribution of finish reasons
        agg_op_stats.finish_reasons[gen_stats.finish_reason] += 1


    def _update_code_gen_stats(self, full_code_gen_stats: FullCodeGenStats) -> None:
        """
        This helper function takes a FullCodeGenStats object and updates the Profiler's
        agg_operator_stats field (which is an AggOperatorStats object).

        A FullCodeGenStats object is a bit of a beast. It currently contains the following:
        - A CodeGenStepStats object for the initial code generation
        - An AdviceGenStepStats object for the initial advice generation
        - A list of AdvisedCodeGenStepStats objects -- one per piece of advice -- which is used
          to perform a new (and ideally better) code generation

        Each of these stats objects within FullCodeGenStats is a wrapper around a GenerationStats
        object, which we already know how to process (see _update_gen_stats).

        Thus, similar to how we handle conventional queries, this function will update
        the aggregate stats (aggregated across all generation calls) AND it will update
        aggregate stats for:
        - initial code generations
        - advice generations
        - advised code generations
        """
        # get stats for initial code generation step
        gen_stats = full_code_gen_stats.init_code_gen_stats.gen_stats

        # update aggregate operator stats and the stats for the initial code generation step
        self._update_gen_stats(gen_stats)
        self._update_gen_stats(gen_stats, code_gen_step="init_code_gen")

        # get stats for advice generation step
        gen_stats = full_code_gen_stats.advice_gen_stats.gen_stats

        # update aggregate operator stats and the stats for the advice generation step
        self._update_gen_stats(gen_stats)
        self._update_gen_stats(gen_stats, code_gen_step="advice_gen")

        # get stats for each advised code generation (one per-piece of generated advice)
        for advised_code_gen in full_code_gen_stats.advised_code_gen_stats:
            gen_stats = advised_code_gen.gen_stats

            # update aggregate operator stats and the stats for the advised code generation step
            self._update_gen_stats(gen_stats)
            self._update_gen_stats(gen_stats, code_gen_step="advised_code_gen")


    def _update_agg_stats(self, stats: Stats) -> None:
        """
        Given the stats object for a single record which was operated on by this operator,
        update the aggregate stats for this operator.

        `stats` can only be one of:
        - Stats
        - InduceNonLLMStats
        - InduceLLMStats
        - FilterLLMStats

        Stats is only present for non-induce/filter operations, and its only field will
        be the cumulative_iter_time for the record.

        InduceNonLLMStats is either empty or has a single field (api_stats).

        InduceLLMStats is the most complex Stats object, it can contain one or more of
        the following sub-fields:
        - bonded_query_stats
        - conventional_query_stats
        - full_code_gen_stats

        FilterLLMStats has a single field gen_stats which is guaranteed to be filled.
        """
        # every Stats object will have cumulative_iter_time; update it
        self.agg_operator_stats.total_cumulative_iter_time += stats.cumulative_iter_time

        # non-LLM induce objects will have no stats or a single ApiStats object
        if isinstance(stats, InduceNonLLMStats):
            api_stats = stats.api_stats
            if api_stats is not None:
                self.agg_operator_stats.total_api_call_duration += api_stats.api_call_duration_secs

        # LLM induce objects are the most complex; they may have one or more of:
        # - BondedQueryStats
        # - ConventionalQueryStats
        # - FullCodeGenStats
        elif isinstance(stats, InduceLLMStats):
            # process bonded query stats
            bonded_query_stats = stats.bonded_query_stats
            if bonded_query_stats is not None:
                # set input fields and output fields generated by induce operation
                self.agg_operator_stats.input_fields = bonded_query_stats.input_fields
                self.agg_operator_stats.generated_fields = bonded_query_stats.generated_fields

                # update the aggregate operator stats associated with LLM generation
                self._update_gen_stats(bonded_query_stats.gen_stats)

            # process conventional query stats
            conventional_query_stats = stats.conventional_query_stats
            if conventional_query_stats is not None:
                # set input fields and output fields generated by induce operation
                self.agg_operator_stats.input_fields = conventional_query_stats.input_fields
                self.agg_operator_stats.generated_fields = conventional_query_stats.generated_fields

                # update the aggregate (and per-field aggregate) operator stats associated with LLM generation
                for field_query_stats in conventional_query_stats.field_query_stats_lst:
                    field_name = field_query_stats.field_name
                    field_gen_stats = field_query_stats.gen_stats
                    self._update_gen_stats(field_gen_stats)
                    self._update_gen_stats(field_gen_stats, field_name=field_name)

            # process codegen stats
            full_code_gen_stats = stats.full_code_gen_stats
            if full_code_gen_stats is not None:
                self._update_code_gen_stats(full_code_gen_stats)

        # filter llm objects will have a single GenerationStats object
        elif isinstance(stats, FilterLLMStats):
            # update aggregate statistics with filter generation stats
            self._update_gen_stats(stats.gen_stats)


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
                # set operator name in aggregate stats
                self.agg_operator_stats.op_name = name

                # iterate through records and update profiling stats as we go
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
                        "stats": record._stats[self.op_id].to_dict(), # TODO: filter out prompts?
                        "record_state": record.asDict(include_bytes=False),
                    }

                    # add record state to set of records computed by this operator
                    self.records.append(record._state[self.op_id])

                    # update aggregate stats for operators with LLM workloads
                    self._update_agg_stats(record._stats[self.op_id])

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
