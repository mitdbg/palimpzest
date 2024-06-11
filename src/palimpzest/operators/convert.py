from __future__ import annotations
from palimpzest.dataclasses import OperatorStats, OperatorCostEstimates
from palimpzest.operators import PhysicalOperator

from palimpzest.constants import *
from palimpzest.corelib import *
from palimpzest.elements import *

from typing import Any, Dict, Optional, Tuple

import math
import concurrent


class ConvertOp(PhysicalOperator):

    inputSchema = Schema
    outputSchema = Schema

    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
        cardinality: Optional[str] = "oneToOne",
        desc: Optional[str] = None,
        targetCacheId: Optional[str] = None,
        shouldProfile: Optional[bool] = False,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            shouldProfile=shouldProfile,
        )
        self.cardinality = cardinality
        self.desc = desc
        self.targetCacheId = targetCacheId

    def physical_op_id(self, plan_position: Optional[int] = None):
        op_dict = {
            "operator": self.op_name(),
            "inputSchema": str(self.inputSchema),
            "outputSchema": str(self.outputSchema),
            "cardinality": self.cardinality,
            "desc": str(self.desc),
        }

        return self._compute_op_id_from_dict(op_dict, plan_position)

    def __str__(self):
        return f"{self.model}_{self.query_strategy}_{self.token_budget}"

    def __call__(self, candidate: DataRecord) -> Tuple[DataRecord, Optional[OperatorStats]]:
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")

    # def __iter__(self) -> IteratorFn:
    #     shouldCache = self.datadir.openCache(self.targetCacheId)

    #     @self.profile(name="convert", shouldProfile=self.shouldProfile)
    #     def iteratorFn():
    #         for nextCandidate in self.source:
    #             resultRecordList = self.__call__(nextCandidate)
    #             if resultRecordList is not None:
    #                 for resultRecord in resultRecordList:
    #                     if resultRecord is not None:
    #                         if shouldCache:
    #                             self.datadir.appendCache(
    #                                 self.targetCacheId, resultRecord
    #                             )
    #                         yield resultRecord
    #         if shouldCache:
    #             self.datadir.closeCache(self.targetCacheId)

    #     return iteratorFn()


class ParallelConvertFromCandidateOp(ConvertOp):
    def __init__(self, streaming, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 32  # TODO hardcoded for now
        self.streaming = streaming

    def __eq__(self, other: PhysicalOperator):
        return super().__eq__(other) and self.streaming == other.streaming

    def copy(self):
        return super().copy(streaming=self.streaming)

    def __iter__(self):
        # This is very crudely implemented right now, since we materialize everything
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_convert", shouldProfile=self.shouldProfile)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks using self.max_workers
            if self.streaming:
                chunksize = self.max_workers
            else:
                chunksize = len(inputs)

            if chunksize == 0:
                return

            for i in range(0, len(inputs), chunksize):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    results = list(
                        executor.map(self.__call__, inputs[i : i + chunksize])
                    )

                    for resultRecordList in results:
                        if resultRecordList is not None:
                            for resultRecord in resultRecordList:
                                if resultRecord is not None:
                                    if shouldCache:
                                        self.datadir.appendCache(
                                            self.targetCacheId, resultRecord
                                        )
                                    yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class LLMConvert(ConvertOp):
    def __init__(
        self,
        model: Optional[Model] = None,
        prompt_strategy: Optional[PromptStrategy] = None,
        query_strategy: Optional[QueryStrategy] = None,
        token_budget: Optional[float] = None,
        image_conversion: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.image_conversion = image_conversion
        self.prompt_strategy = prompt_strategy
        self.query_strategy = query_strategy
        self.token_budget = token_budget

        self.heatmap_json_obj = None
        # use image model if this is an image conversion
        if self.outputSchema == ImageFile and self.inputSchema == File or self.image_conversion:
            # TODO : find a more general way by llm provider
            # TODO : which module is responsible of setting PromptStrategy.IMAGE_TO_TEXT?
            if self.model in [Model.GPT_3_5, Model.GPT_4]:
                self.model = Model.GPT_4V
            if self.model == Model.GEMINI_1:
                self.model = Model.GEMINI_1V
            if self.model in [Model.MIXTRAL, Model.LLAMA2]:
                import random

                self.model = random.choice([Model.GPT_4V, Model.GEMINI_1V])

            # TODO: remove; for evaluations just use GPT_4V
            self.model = Model.GPT_4V
            self.prompt_strategy = PromptStrategy.IMAGE_TO_TEXT
            self.query_strategy = QueryStrategy.BONDED_WITH_FALLBACK
            self.token_budget = 1.0

        # TODO: combine these functions
        # set model to None if this is a simple conversion
        # if self._is_quick_conversion() or self.is_hardcoded():
        # self.model = None
        # self.prompt_strategy = None
        # self.query_strategy = None
        # self.token_budget = 1.0

        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            self.model = None
            self.prompt_strategy = None
            self.token_budget = 1.0

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.cardinality == other.cardinality
            and self.image_conversion == other.image_conversion
            and self.prompt_strategy == other.prompt_strategy
            and self.query_strategy == other.query_strategy
            and self.token_budget == other.token_budget
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
        )

    def __str__(self):
        model = self.model.value if self.model is not None else ""
        ps = self.prompt_strategy.value if self.prompt_strategy is not None else ""
        qs = self.query_strategy.value if self.query_strategy is not None else ""

        return f"{self.__class__.__name__}({str(self.outputSchema):10s}, Model: {model}, Prompt Strategy: {ps}, Query Strategy: {qs}, Token Budget: {str(self.token_budget)})"

    def copy(self):
        return LLMConvert(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            model=self.model,
            cardinality=self.cardinality,
            image_conversion=self.image_conversion,
            prompt_strategy=self.prompt_strategy,
            query_strategy=self.query_strategy,
            token_budget=self.token_budget,
            desc=self.desc,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def physical_op_id(self, plan_position: Optional[int] = None):
        op_dict = {
            "operator": self.op_name(),
            "inputSchema": str(self.inputSchema),
            "outputSchema": str(self.outputSchema),
            "cardinality": self.cardinality,
            "model": self.model.value if self.model is not None else None,
            "prompt_strategy": (
                self.prompt_strategy.value if self.prompt_strategy is not None else None
            ),
            "query_strategy": (
                self.query_strategy.value if self.query_strategy is not None else None
            ),
            "token_budget": self.token_budget,
            "desc": str(self.desc),
        }

        return self._compute_op_id_from_dict(op_dict, plan_position)

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # estimate number of input and output tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

        if self.token_budget is not None:
            est_num_input_tokens = self.token_budget * est_num_input_tokens

        # override for GPT-4V image conversion
        if self.model == Model.GPT_4V:
            # 1024x1024 image is 765 tokens
            # TODO: revert / 10 after running real-estate demo
            est_num_input_tokens = 765 / 10

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        # NOTE: model will only be None for code generation, which uses GPT-3.5 as fallback
        model_name = self.model.value if self.model is not None else Model.GPT_3_5.value
        model_conversion_time_per_record = (
            MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens
        ) / self.max_workers

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[model_name]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # TODO: make this better after arxiv; right now codegen is hard-coded to use GPT-4
        # if we're using code generation, assume that model conversion time and cost are low
        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            model_conversion_time_per_record = 1e-5
            model_conversion_usd_per_record = 1e-4  # amortize code gen cost across records

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality == "oneToOne" else NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["MMLU"] / 100.0) * source_op_cost_estimates.quality

        # TODO: make this better after arxiv; right now codegen is hard-coded to use GPT-4
        # if we're using code generation, assume that quality goes down (or view it as E[Quality] = (p=gpt4[code])*1.0 + (p=0.25)*0.0))
        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            quality = quality * (GPT_4_MODEL_CARD["code"] / 100.0)

        if self.token_budget is not None:
            # for now, assume quality is proportional to sqrt(token_budget)
            quality = quality * math.sqrt(math.sqrt(self.token_budget))  

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    # TODO
    def __call__(self, candidate: DataRecord) -> Tuple[DataRecord, Optional[OperatorStats]]:
        # initialize stats objects
        bonded_query_stats, conventional_query_stats = None, None

        if self.query_strategy == QueryStrategy.CONVENTIONAL:
            # NOTE: runConventionalQuery does exception handling internally
            dr, conventional_query_stats = runConventionalQuery(
                candidate, td, self._verbose
            )

            # if profiling, set record's stats for the given op_id
            # if shouldProfile:
            # dr._stats[td.op_id] = ConvertLLMStats(
            # query_strategy=td.query_strategy.value,
            # token_budget=td.token_budget,
            # conventional_query_stats=conventional_query_stats,
            # )
            return [dr], None

        elif self.query_strategy == QueryStrategy.BONDED:
            drs, new_heatmap_obj, bonded_query_stats, err_msg = runBondedQuery(
                candidate, td, self._verbose
            )

            # if bonded query failed, manually set fields to None
            if err_msg is not None:
                print(f"BondedQuery Error: {err_msg}")
                dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                for field_name in td.outputSchema.fieldNames():
                    setattr(dr, field_name, None)
                drs = [dr]

            # if profiling, set record's stats for the given op_id
            if shouldProfile:
                for dr in drs:
                    dr._stats[td.op_id] = ConvertLLMStats(
                        query_strategy=td.query_strategy.value,
                        token_budget=td.token_budget,
                        bonded_query_stats=bonded_query_stats,
                    )

            return drs, new_heatmap_obj

        elif td.query_strategy == QueryStrategy.BONDED_WITH_FALLBACK:
            drs, new_heatmap_obj, bonded_query_stats, err_msg = runBondedQuery(
                candidate, td, self._verbose
            )

            # if bonded query failed, run conventional query
            if err_msg is not None:
                print(f"BondedQuery Error: {err_msg}")
                print("Falling back to conventional query")
                dr, conventional_query_stats = runConventionalQuery(
                    candidate, td, self._verbose
                )
                drs = [dr] if type(dr) is not list else dr
            # if profiling, set record's stats for the given op_id
            if shouldProfile:
                for dr in drs:
                    # TODO: divide bonded query_stats time, cost, and input/output tokens by len(drs)
                    dr._stats[td.op_id] = ConvertLLMStats(
                        query_strategy=td.query_strategy.value,
                        token_budget=td.token_budget,
                        bonded_query_stats=bonded_query_stats,
                        conventional_query_stats=conventional_query_stats,
                    )

            return drs, new_heatmap_obj

        elif td.query_strategy == QueryStrategy.CODE_GEN:
            dr, full_code_gen_stats = runCodeGenQuery(candidate, td, self._verbose)
            drs = [dr]

            # if profiling, set record's stats for the given op_id
            if shouldProfile:
                for dr in drs:
                    dr._stats[td.op_id] = ConvertLLMStats(
                        query_strategy=td.query_strategy.value,
                        token_budget=td.token_budget,
                        full_code_gen_stats=full_code_gen_stats,
                    )

            return drs, None

        elif td.query_strategy == QueryStrategy.CODE_GEN_WITH_FALLBACK:
            # similar to in _makeLLMTypeConversionFn; maybe we can have one strategy in which we try
            # to use code generation, but if it fails then we fall back to a conventional query strategy?
            dr, full_code_gen_stats, conventional_query_stats = runCodeGenQuery(
                candidate, td, self._verbose
            )
            drs = [dr] if type(dr) is not list else dr
            # # Deleting all failure fields
            # for field_name in td.outputSchema.fieldNames():
            #     if hasattr(new_candidate, field_name) and (getattr(new_candidate, field_name) is None):
            #         delattr(new_candidate, field_name)
            # if td.cardinality == 'oneToMany':
            #     td.cardinality = 'oneToOne'
            # dr, conventional_query_stats = runConventionalQuery(new_candidate, td, self._verbose)
            # drs = [dr] if type(dr) is not list else dr
            for dr in drs:
                dr._parent_uuid = candidate._uuid

            # if profiling, set record's stats for the given op_id
            # if shouldProfile:
            # for dr in drs:
            # TODO: divide bonded query_stats time, cost, and input/output tokens by len(drs)
            # dr._stats[td.op_id] = ConvertLLMStats(
            # query_strategy=td.query_strategy.value,
            # token_budget=td.token_budget,
            # full_code_gen_stats=full_code_gen_stats,
            # conventional_query_stats=conventional_query_stats,
            # )

            return drs, None

        else:
            raise ValueError(f"Unrecognized QueryStrategy: {td.query_strategy.value}")