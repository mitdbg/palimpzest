from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple


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
    per_field_agg_op_stats: defaultdict[AggOperatorStats]=field(default_factory=lambda: defaultdict(lambda: AggOperatorStats()))

    #######################################
    ##### Fields for Induce w/CodeGen #####
    #######################################
    # [computed in _update_agg_stats] record aggregated stats on a per-step basis for codegen queries
    # (here the steps are: initial code generation, advice generation, advised code generation)
    code_gen_agg_op_stats: defaultdict[AggOperatorStats]=field(default_factory=lambda: defaultdict(lambda: AggOperatorStats()))

    ##########################################
    ##### Optional Non-LLM Induce Fields #####
    ##########################################
    # [computed in _update_agg_stats] time spent waiting for API calls to return (in seconds)
    total_api_call_duration: float=0.0

    def to_dict(self):
        # convert defaultdict -> dict before calling asdict()
        self.finish_reasons = dict(self.finish_reasons)
        self.per_field_agg_op_stats = dict(self.per_field_agg_op_stats)
        self.code_gen_agg_op_stats = dict(self.code_gen_agg_op_stats)

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
    cumulative_iter_time: float=None
    # the time spent by the data record just in this operation; this is computed
    # in the StatsProcessor as (the cumulative_iter_time of this data record in
    # this operation) minus (the cumulative_iter_time of this data record in the
    # source operation)
    op_time: float=None

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

    TODO
    ----
    8. augment Execution to be able to iteratively draw sample data (instead of just once up front)
    9. implement other methods here to help with understanding profile data
    """
    def __init__(self, profiling_data: Dict[str, Any]) -> None:
        """
        The profiling data dictionary has the following structure:
        
        {
            "agg_operator_stats": {
                #### unique identifier for this instance of this operator
                "op_id": str,
                #### name of this operator
                "op_name": str,
                #### total records processed by op
                "total_records": int,
                #### sum of cumulative_iter_time for records in op (in seconds)
                "total_cumulative_iter_time": float,
                #### Sum of op_time for records in op (in seconds) -- this is computed in StatsProcessor.__init__()
                "total_op_time": float,
                #### total time spent inside of profiler code (in seconds)
                "total_time_in_profiler": float,
                #### total input and output tokens processed in op
                "total_input_tokens": int,
                "total_output_tokens": int,
                #### total dollars spent in op
                "total_input_usd": float,
                "total_output_usd": float,
                "total_usd": float,
                #### total time spent executing LLM calls in this op (in seconds)
                "total_llm_call_duration": float,
                #### distribution of finish reasons for LLM calls in this op
                "finish_reasons": Dict[str, int],
                #### total time spent waiting on non-LLM API calls (in seconds)
                "total_api_call_duration": float,
                #### the input fields for records coming into this op, and the fields this op generated
                "input_fields": List[str],
                "generated_fields": List[str],
                #### ONLY for induce ops with conventional queries -- per-field breakdown of these agg_operator_stats
                "per_field_agg_op_stats": Dict[str, Dict[str, Any]],
                #### ONLY for induce ops with code gen -- per-state breakdown of these agg_operator_stats
                "code_gen_agg_op_stats": Dict[str, Dict[str, Any]],
            },
            "records": [
                {
                    #### name of this operator
                    "name": str,
                    #### unique identifier for this record
                    "uuid": str,
                    #### unique identifier for the parent/source of this record
                    "parent_uuid": str,
                    #### 
                    "stats": {
                        #### Total time in seconds spent waiting for operator to yield this record; includes time spent in source operators
                        "cumulative_iter_time": float,
                        #### Total time in seconds spent by record in this operator -- this is computed in StatsProcessor.__init__()
                        "op_time": 0.0,
                        #### per-record stats; can include zero-or-more of the following fields:
                        ## for induce operations with an LLM
                        "bonded_query_stats": Dict[str, Any],
                        "conventional_query_stats": Dict[str, Any],
                        "full_code_gen_stats": Dict[str, Any],
                        ## for induce operations w/out an LLM
                        "api_stats": Dict[str, Any],
                        ## for filter operations
                        "gen_stats: Dict[str, Any],
                    },
                    #### dictionary representation of the record after being processed by this operator
                    "record_state": {
                        "<field-name-1>": value1,
                        "<field-name-2>": value2,
                        ...
                    }
                },
                ...
            ],
            #### the data structure recurses until the original source operation (e.g. a scan) is reached
            "source": {
                "agg_operator_stats": {...},
                "records": [...],
                "source": {
                    ...
                }
            },
        }
        """
        # compute op_time for each record and the total_op_time for each operator
        self.profiling_data = self._compute_op_time(profiling_data)


    def _compute_op_time(self, profiling_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This helper function computes the time spent by each record in each operation
        (i.e. the record's op_time). It then aggregates the op_times for every record
        in each operation to get a total_op_time.

        Inside the profiler we are only able to track the time it takes for a record
        to be yielded by the operator's iterator method. This time (stored in
        "cumulative_iter_time") is cumulative in the sense that it also captures time
        spent waiting for source/parent operators to yield this record.

        In this function, for each record we effectively compute:

        op_time = (cumulative_iter_time) - (the cumulative_iter_time of this record's parent)

        Once we've computed each record's op_time we finally compute the total_op_time
        for each operator.
        """
        # base case: this is the source operation
        if "source" not in profiling_data:
            # in this case: op_time == cumulative_iter_time
            for record in profiling_data['records']:
                record['stats']['op_time'] = record['stats']['cumulative_iter_time']

            # compute total_op_time
            profiling_data['agg_operator_stats']['total_op_time'] = sum(list(map(lambda record: record['stats']['op_time'], profiling_data['records'])))

            return profiling_data

        # TODO: this is N^2 in # of records; we may want to use a dictionary to speed this up
        # for each record we need to identify its parent to compute the op_time
        for record in profiling_data['records']:
            parent_uuid = record['parent_uuid']
            for source_record in profiling_data['source']['records']:
                if source_record['uuid'] == parent_uuid:
                    record['stats']['op_time'] = record['stats']['cumulative_iter_time'] - source_record['stats']['cumulative_iter_time']

        # compute total_op_time
        profiling_data['agg_operator_stats']['total_op_time'] = sum(list(map(lambda record: record['stats']['op_time'], profiling_data['records'])))

        # recurse
        profiling_data["source"] = self._compute_op_time(profiling_data["source"])

        return profiling_data

    def _est_time_per_record(self, op_data: Dict[str, Any]) -> float:
        """
        Given an operator's profiling data dictionary, estimate the time_per_record
        to be the per-record average op_time.
        """
        return op_data['total_op_time'] / op_data['total_records']

    def _est_num_input_output_tokens(self, op_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Given an operator's profiling data dictionary, estimate the number of input and
        output tokens to be their per-record averages.
        """
        avg_num_input_tokens = op_data['total_input_tokens'] / op_data['total_records']
        avg_num_output_tokens = op_data['total_output_tokens'] / op_data['total_records']

        return avg_num_input_tokens, avg_num_output_tokens

    def _est_usd_per_record(self, op_data: Dict[str, Any]) -> float:
        """
        Given an operator's profiling data dictionary, estimate the usd_per_record
        to be the per-record average usd spent.
        """
        return op_data['total_usd'] / op_data['total_records']

    def _est_selectivity(self, op_data: Dict[str, Any], source_op_data: Dict[str, Any]) -> float:
        """
        Given an operator's profiling data dictionary and the profiling data dictionary
        of its source operator, estimate the selectivity to be the ratio of records
        output by each operation. 
        """
        return op_data['total_records'] / source_op_data['total_records']

    def compute_cost_estimates(self) -> Dict[str, Dict[str, Any]]:
        """
        This function computes a mapping from each induce op_id to a dictionary
        containing the sample estimate(s) of key statistics for that operation.
        """
        op_cost_estimates = {}
        op_data = self.profiling_data['agg_operator_stats']
        source_op_data = (
            self.profiling_data['source']['agg_operator_stats']
            if 'source' in self.profiling_data
            else None
        )
        op_id = op_data['op_id']
        while op_id is not None:
            op_cost_estimates[op_id] = {}

            # compute per-record average time spent in operator
            avg_op_time = self._est_time_per_record(op_data)
            op_cost_estimates[op_id]['time_per_record'] = avg_op_time

            # compute est_num_input_tokens and est_num_output_tokens to be the
            # per-record average number of input and output tokens, respectively;
            avg_num_input_tokens, avg_num_output_tokens = self._est_num_input_output_tokens(op_data)
            op_cost_estimates['est_num_input_tokens'] = avg_num_input_tokens
            op_cost_estimates['est_num_output_tokens'] = avg_num_output_tokens

            # compute _usd_per_record (even though this is just a derivative of
            # avg_num_input/output_tokens) as the per-record average spend
            avg_usd = self._est_usd_per_record(op_data)
            op_cost_estimates['usd_per_record'] = avg_usd

            # NOTE: we estimate selectivity instead of cardinality because, given PZ's current
            #       design, we will run the StatsProcessor on a sample of records to get data
            #       for better cost estimates. Thus, using the `total_records` field as an estimate
            #       for the cardinality of the operation would be really, really bad since it
            #       would just be equal to the sample size. Instead, we estimate the selectivity
            #       and then estimate new cardinalities inside each physical operator by multiplying
            #       its source's cardinality by the selectivity estimate. The ultimate sources
            #       (e.g. the CacheScan / MarshalAndScanDataOp) will give real cardinalities based
            #       on the size of the datasource they are reading from.
            #
            # compute selectivity as (# of records in this op) / (# records in parent op);
            # if this is the source operation then selectivity = 1.0
            selectivity = (
                self._est_selectivity(op_data, source_op_data)
                if source_op_data is not None
                else 1.0
            )
            op_cost_estimates['selectivity'] = selectivity

            # For now, for the reasons outlined in the NOTE above, we do not directly estimate cardinality
            op_cost_estimates['cardinality'] = None

            # TODO: try estimating quality using mean or p90 log probability
            # - first approach: mean output log prob. from generations? (no labels necessary)
            # - if we have labels we can estimate directly (use semantic answer similarity to determine if output is correct)
            op_cost_estimates['quality'] = None

            # 
            op_data = source_op_data
            source_op_data = (
                op_data['source']['agg_operator_stats']
                if 'source' in op_data
                else None
            )
            op_id = op_data['op_id'] if op_data is not None else None

        return op_cost_estimates

    def get_avg_record_stats(self):
        """
        Return a representation of an average trace for a record. E.g., it
        starts in such and such operation and takes blah seconds to load
        on avg., on median, p95, p99, max, etc. Then it goes to induce...
        """
        pass

    def get_operator_aggregate_stats(self):
        """
        Return mapping op_id -> agg. stats. Also include computation tree.
        Also compute mean, median, p95, p99, max stats.
        """
        pass

    def get_output_record_lineages(self):
        """
        Get the lineage of transformations for each record in the final output
        result set. The output is a list of lists, where each element in the outer
        list represents a lineage of computation, and each element in the inner list
        (i.e. the lineage of computation) is the state of each record after each
        physical operation.
        """
        pass

    def get_input_record_lineages(self):
        """
        Get the lineage of transformations for each record in the input set.
        The output is a list of lists, where each element in the outer list
        represents a lineage of computation, and each element in the inner list
        (i.e. the lineage of computation) is the state of each record after each
        physical operation.
        """
        pass
