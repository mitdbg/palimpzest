from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List


@dataclass
class AggOperatorStats:
    """
    Dataclass for storing aggregate statistics related to an operator.

    Note that not every field will be computed for a given operation. This represents
    the union of all statistics computed across all operators.

    All other Stats classes are concerned with the statistics computed for a single
    induce / filter operation executed on a single input record. This dataclass
    aggregates across all of those statistics (e.g. by capturing the total number
    of output tokens generated across all generations, the total number of records
    processed, etc.)
    """
    ############################
    ##### Universal Fields #####
    ############################
    # the ID of the operation in which these stats were collected
    op_id: str=None
    # [set in iter_profiler] the name of the operation in which these stats were collected
    op_name: str=None
    # [computed in iter_profiler] total number of records returned by the iterator for this operator
    total_records: int=0
    # [computed in _update_agg_stats] total time spent in this iterator; this will include time spent in input operators
    total_cumulative_iter_time: float=0.0
    # total time spent in this operation; does not include time spent waiting on parent/source operators
    total_op_time: float=0.0
    # [computed in iter_profiler] keep track of the total time spent inside of the profiler
    total_time_in_profiler: float=0.0

    ##############################################
    ##### Universal Induce and Filter Fields #####
    ##############################################
    # [computed in _update_agg_stats] usage statistics computed for induce and filter operations
    total_input_tokens: int=0
    total_output_tokens: int=0
    # [computed in _update_agg_stats] dollar cost associated with usage statistics
    total_input_usd: float=0.0
    total_output_usd: float=0.0
    total_usd: float=0.0
    # [computed in _update_agg_stats] time spent waiting for LLM calls to return (in seconds)
    total_llm_call_duration: float=0.0
    # [computed in _update_agg_stats] keep track of finish reasons
    finish_reasons: defaultdict[int]=field(default_factory=lambda: defaultdict(int))
    # [computed in _update_agg_stats] record input fields and the output fields generated in an induce operation
    input_fields: List[str]=field(default_factory=list)
    generated_fields: List[str]=field(default_factory=list)

    #################################################
    ##### Field for Induce w/Conventional Query #####
    #################################################
    # [computed in _update_agg_stats] record aggregated stats on a per-field basis for conventional induce queries
    per_field_agg_op_stats: defaultdict[AggOperatorStats]=field(default_factory=lambda: defaultdict(AggOperatorStats))

    #######################################
    ##### Fields for Induce w/CodeGen #####
    #######################################
    # [computed in _update_agg_stats] record aggregated stats on a per-step basis for codegen queries
    # (here the steps are: initial code generation, advice generation, advised code generation)
    code_gen_agg_op_stats: defaultdict[AggOperatorStats]=field(default_factory=lambda: defaultdict(AggOperatorStats))

    ##########################################
    ##### Optional Non-LLM Induce Fields #####
    ##########################################
    # [computed in _update_agg_stats] time spent waiting for API calls to return (in seconds)
    total_api_call_duration: float=0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Stats:
    """
    Base dataclass for storing statistics captured during the execution of an induce / filter
    operation on a single input record.
    """
    # (set in profiler.py) this is the total time spent waiting for the iterator
    # to yield the data record associated with this Stats object; note that this
    # will capture the total time spent in this operation and all source operations
    # due to the way in which we set timers in profiler.py rather than in the
    # physical operator code
    cumulative_iter_time: float=0.0
    # the time spent by the data record just in this operation; this is computed
    # in the StatsProcessor as (the cumulative_iter_time of this data record in
    # this operation) minus (the cumulative_iter_time of this data record in the
    # source operation)
    op_time: float=0.0

    def to_dict(self):
        return asdict(self)

@dataclass
class ApiStats(Stats):
    """Staistics captured from a (non-LLM) API call."""
    # total time spent waiting for the API to return a response
    api_call_duration_secs: float=0.0

@dataclass
class GenerationStats(Stats):
    """Statistics captured from LLM calls (i.e. the Generator's .generate() functions)."""
    # name of the LLM used to generate the output; should be a key in the MODEL_CARDS
    model_name: str=None
    # total time spent from sending input to LLM to receiving output string
    llm_call_duration_secs: float=0.0
    # the prompt used to generate the output
    prompt: str=None
    # the usage dictionary for the LLM call
    usage: Dict[str, int]=field(default_factory=dict)
    # the reason the LLM stopped generating tokens
    finish_reason: str=None

@dataclass
class BondedQueryStats(Stats):
    """Statistics captured from bonded queries."""
    # the generation statistics from the call to a generator
    gen_stats: GenerationStats=None
    # the set of input fields passed into the generation
    input_fields: List[str]=field(default_factory=list)
    # the set of fields the bonded query was supposed to generate values for
    generated_fields: List[str]=field(default_factory=list)

@dataclass
class FieldQueryStats(Stats):
    """Statistics captured from generating the value for a single field in a conventional query."""
    # the generation statistics from the call to a generator
    gen_stats: GenerationStats=None
    # the field the statistics were generated for
    field_name: str=None

@dataclass
class ConventionalQueryStats(Stats):
    """Statistics captured from conventional queries."""
    # the list of FieldQueryStats for each field in the generated_fields
    field_query_stats_lst: List[FieldQueryStats]=field(default_factory=list)
    # the set of input fields passed into the generation
    input_fields: List[str]=field(default_factory=list)
    # the set of fields the conventional query was supposed to generate values for
    generated_fields: List[str]=field(default_factory=list)

@dataclass
class CodeGenStepStats(Stats):
    """
    Statistics captured from running the initial code generation step.
    In this step, the CODEGEN_PROMPT has formatting applied and then an initial
    piece of code is generated with a call to `llmCodeGen`, which returns the `code`
    and a GenerationStats object.
    """
    # CODEGEN_PROMPT after formatting is applied
    codegen_prompt: str=None
    # initial code generated by call to `llmCodeGen` with formatted CODEGEN_PROMPT
    code: str=None
    # the generation statistics from the call to `llmCodeGen`
    gen_stats: GenerationStats=None
    
@dataclass
class AdviceGenStepStats(Stats):
    """
    Statistics captured from running the advice generation step.
    In this step, the ADVICEGEN_PROMPT has formatting applied and then a set of
    pieces of advice are generated with a call to `llmAdviceGen`, which returns the
    `advices` and a GenerationStats object.
    """
    # ADVICEGEN_PROMPT after formatting is applied
    advicegen_prompt: str=None
    # list of advice strings generated by call to `llmAdviceGen` with formatted ADVICEGEN_PROMPT
    advices: List[str]=field(default_factory=list)
    # the generation statistics from the call to `llmAdviceGen`
    gen_stats: GenerationStats=None

@dataclass
class AdvisedCodeGenStepStats(Stats):
    """
    Statistics captured from running the and advised code generation step.
    In this step, the ADVICED_CODEGEN_PROMPT has formatting applied and then a
    piece of code is generated with a call to `llmCodeGen` for some given piece
    of `advice`. The call to `llmCodeGen` returns the `code` and a GenerationStats object.
    """
    # ADVICED_CODEGEN_PROMPT after formatting is applied
    adviced_codegen_prompt: str=None
    # the piece of advice used to in this advised code generation
    advice: str=None
    # the code generated with this piece of advice
    code: str=None
    # the generation statistics from the call to `llmCodeGen`
    gen_stats: GenerationStats=None

@dataclass
class FullCodeGenStats(Stats):
    """
    The entire set of statistics that can be captured from a call to the `codeGen` function.

    This includes:
    - stats for the initial code generation step
    - stats for the advice generation step
    - a list of stats for each code generation step with a given piece of advice
    """
    # stats from the initial code generation step
    init_code_gen_stats: CodeGenStepStats=None
    # stats from the advice generation step (this is an optional step)
    advice_gen_stats: AdviceGenStepStats=None
    # list of stats from the advised code generation step (one set of stats per-piece of advice)
    advised_code_gen_stats: List[AdvisedCodeGenStepStats]=field(default_factory=list)

@dataclass
class InduceLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from an induce w/LLM operation."""
    # stats from bonded query
    bonded_query_stats: BondedQueryStats=None
    # stats from conventional query
    conventional_query_stats: ConventionalQueryStats=None
    # stats from code generation
    full_code_gen_stats: FullCodeGenStats=None

@dataclass
class InduceNonLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from a hard-coded induce operation."""
    # stats containing time spent calling some external API
    api_stats: ApiStats=None

@dataclass
class FilterLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from a filter operation."""
    # the generation statistics from the call to the filter LLM
    gen_stats: GenerationStats=None


class StatsProcessor:
    """
    This class implements a set of standardized functions for processing profiling statistics
    collected by PZ.
    """
    def __init__(self, stats: Stats) -> None:
        self.stats = stats
