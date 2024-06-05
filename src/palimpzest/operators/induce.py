"""GV: In my view this whole function should be called Convert. 
And so should all of the methods - ConvertFromCandidate, ParallelConvertFromCandidate, etc.
Keeping it as Induce for legacy compatibility (for now)"""

from __future__ import annotations
from io import BytesIO

from palimpzest.profiler.stats import Stats
from palimpzest.tools.skema_tools import equations_to_latex
import pandas as pd
from .physical import PhysicalOp, MAX_ID_CHARS, IteratorFn

from palimpzest.constants import *
import palimpzest.corelib.schemas as schemas
from palimpzest.elements import *
from palimpzest.solver.task_descriptors import TaskDescriptor
from palimpzest.profiler import Profiler

from typing import Any, Dict, Optional, Tuple

import math
import concurrent
import hashlib


class InduceOp(PhysicalOp):

    inputSchema = Schema
    outputSchema = Schema

    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
        model: Model,
        cardinality: str,
        image_conversion: bool = False,
        prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_QA,
        query_strategy: QueryStrategy = QueryStrategy.BONDED_WITH_FALLBACK,
        token_budget: float = 1.0,
        desc: Optional[str] = None,
        targetCacheId: Optional[str] = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            shouldProfile=shouldProfile,
        )
        self.model = model
        self.cardinality = cardinality
        self.image_conversion = image_conversion
        self.prompt_strategy = prompt_strategy
        self.query_strategy = query_strategy
        self.token_budget = token_budget
        self.desc = desc
        self.targetCacheId = targetCacheId
        self.heatmap_json_obj = None
        # use image model if this is an image conversion
        if outputSchema == ImageFile and inputSchema == File or self.image_conversion:
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

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        # self.profiler = Profiler(op_id=self.opId())
        # self.profile = self.profiler.iter_profiler

        # # construct TaskDescriptor
        # taskDescriptor = self._makeTaskDescriptor()

        # # synthesize task function
        # if not taskDescriptor.op_id in PhysicalOp.synthesizedFns:
        #     PhysicalOp.synthesizedFns[taskDescriptor.op_id] = PhysicalOp.solver.synthesize(taskDescriptor, shouldProfile=self.shouldProfile)

    def copy(self, *args, **kwargs):
        return self.__class__(
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
            *args,
            **kwargs,
        )

    def __eq__(self, other: PhysicalOp):
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
        qs = self.query_strategy.value if self.query_strategy is not None else ""

        return f"{self.__class__.__name__}({str(self.outputSchema):10s}, Model: {model}, Query Strategy: {qs}, Token Budget: {str(self.token_budget)})"

    def __call__(self, candidate: DataRecord) -> Tuple[DataRecord, Optional[Stats]]:
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")

    # def _makeTaskDescriptor(self):
    #     td = TaskDescriptor(
    #         physical_op=self.__class__.__name__,
    #         inputSchema=self.inputSchema,
    #         outputSchema=self.outputSchema,
    #         op_id=self.opId(),
    #         model=self.model,
    #         cardinality=self.cardinality,
    #         image_conversion=self.image_conversion,
    #         prompt_strategy=self.prompt_strategy,
    #         query_strategy=self.query_strategy,
    #         token_budget=self.token_budget,
    #         conversionDesc=self.desc,
    #         pdfprocessor=self.datadir.current_config.get("pdfprocessing"),
    #         plan_idx=self.plan_idx,
    #         heatmap_json_obj=self.heatmap_json_obj,
    #     )
    #     # # This code checks if the function has been synthesized before, and if so, whether it is hardcoded. If so, set model and prompt_strategy to None.
    #     # if td.op_id in PhysicalOp.synthesizedFns:
    #     #     if self.is_hardcoded():
    #     #         td.model = None
    #     #         td.prompt_strategy = None

    #     return td

    # def opId(self):
    #     d = {
    #         "operator": self.__class__.__name__,
    #         "outputSchema": str(self.outputSchema),
    #         "source": self.source.opId(),
    #         "model": self.model.value if self.model is not None else None,
    #         "prompt_strategy": (
    #             self.prompt_strategy.value if self.prompt_strategy is not None else None
    #         ),
    #         "desc": self.desc,
    #         "targetCacheId": self.targetCacheId,
    #     }
    #     ordered = json.dumps(d, sort_keys=True)
    #     return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    # def _attemptMapping(self, candidate: DataRecord):
    # """Attempt to map the candidate to the outputSchema. Return None if it fails."""
    # taskDescriptor = self._makeTaskDescriptor()
    # taskFn = PhysicalOp.solver.synthesize(
    # taskDescriptor, shouldProfile=self.shouldProfile
    # )
    # drs, new_heatmap_json_obj = taskFn(candidate)
    # self.heatmap_json_obj = new_heatmap_json_obj
    # return drs

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        # fetch cost estimates from source operation
        inputEstimates, subPlanCostEst = self.source.estimateCost(cost_est_data)

        # if induce has a quick conversion; set "no-op" cost estimates
        if self._is_quick_conversion() or self.is_hardcoded():
            # we assume time cost of these conversions is negligible
            outputEstimates = {**inputEstimates}
            outputEstimates["timePerElement"] = 0.0
            outputEstimates["usdPerElement"] = 0.0
            return outputEstimates, {
                "cumulative": outputEstimates,
                "thisPlan": {},
                "subPlan": subPlanCostEst,
            }

        # if we have sample estimates, let's use those instead of our prescriptive estimates
        input_fields = self.source.outputSchema.fieldNames()
        generated_fields = [
            field
            for field in self.outputSchema.fieldNames()
            if field not in input_fields
        ]
        generated_fields_str = "-".join(sorted(generated_fields))
        op_filter = f"(generated_fields == '{generated_fields_str}') & (op_name == 'induce' | op_name == 'p_induce')"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            # get estimate data for this physical op's model
            model_name = None if self.model is None else self.model.value
            time_per_record = cost_est_data[op_filter][model_name]["time_per_record"]
            usd_per_record = cost_est_data[op_filter][model_name]["cost_per_record"]
            est_num_input_tokens = cost_est_data[op_filter][model_name][
                "est_num_input_tokens"
            ]
            est_num_output_tokens = cost_est_data[op_filter][model_name][
                "est_num_output_tokens"
            ]
            selectivity = cost_est_data[op_filter][model_name]["selectivity"]
            quality = cost_est_data[op_filter][model_name]["quality"]

            # TODO: if code synth. fails, this will turn into ConventionalQuery calls to GPT-3.5,
            #       which would wildly mess up estimate of time and cost per-record
            # do code synthesis adjustment
            if self.query_strategy in [
                QueryStrategy.CODE_GEN_WITH_FALLBACK,
                QueryStrategy.CODE_GEN,
            ]:
                time_per_record = 1e-5
                usd_per_record = 1e-4
                quality = quality * (GPT_4_MODEL_CARD["code"] / 100.0)

            # token reduction adjustment
            if self.token_budget is not None and self.token_budget < 1.0:
                est_num_input_tokens = self.token_budget * est_num_input_tokens
                usd_per_record = (
                    MODEL_CARDS[self.model.value]["usd_per_input_token"]
                    * est_num_input_tokens
                    + MODEL_CARDS[self.model.value]["usd_per_output_token"]
                    * est_num_output_tokens
                )
                quality = quality * math.sqrt(math.sqrt(self.token_budget))

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates["cardinality"] * selectivity

            # apply quality for this filter to overall quality est.
            quality = inputEstimates["quality"] * quality

            thisCostEst = {
                "time_per_record": time_per_record,
                "usd_per_record": usd_per_record,
                "est_num_output_tokens": est_num_output_tokens,
                "selectivity": selectivity,
                "quality": quality,
            }

            costEst = {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates["cumulativeTimePerElement"]
                + time_per_record,
                "cumulativeUSDPerElement": inputEstimates["cumulativeUSDPerElement"]
                + usd_per_record,
                "totalTime": cardinality * time_per_record
                + inputEstimates["totalTime"],
                "totalUSD": cardinality * usd_per_record + inputEstimates["totalUSD"],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
            }

            return costEst, {
                "cumulative": costEst,
                "thisPlan": thisCostEst,
                "subPlan": subPlanCostEst,
            }

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        if self.token_budget is not None:
            est_num_input_tokens = self.token_budget * est_num_input_tokens

        # estimate number of output tokens as constant multiple of input tokens (for now)
        est_num_output_tokens = OUTPUT_TOKENS_MULTIPLE * est_num_input_tokens

        # override for GPT-4V image conversion
        if self.model == Model.GPT_4V:
            # 1024x1024 image is 765 tokens
            # TODO: revert / 10 after running real-estate demo
            est_num_input_tokens = 765 / 10
            est_num_output_tokens = inputEstimates["estOutputTokensPerElement"]

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
        )

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
            model_conversion_usd_per_record = (
                1e-4  # amortize code gen cost across records
            )

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality != "oneToMany" else 2.0
        cardinality = selectivity * inputEstimates["cardinality"]

        # estimate cumulative time per element
        cumulativeTimePerElement = (
            model_conversion_time_per_record
            + inputEstimates["cumulativeTimePerElement"]
        )
        cumulativeUSDPerElement = (
            model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]
        )

        # NOTE: the following estimate assumes that nested Python generators effectively
        #       execute a single record at a time in sequence. I.e., there is no time
        #       overlap for execution in two different stages of the chain of generators.
        #
        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = (
            model_conversion_time_per_record
            * inputEstimates["cardinality"]
            / self.max_workers
            + inputEstimates["totalTime"]
        )
        totalUSD = (
            model_conversion_usd_per_record * inputEstimates["cardinality"]
            + inputEstimates["totalUSD"]
        )

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["MMLU"] / 100.0) * inputEstimates["quality"]

        # TODO: make this better after arxiv; right now codegen is hard-coded to use GPT-4
        # if we're using code generation, assume that quality goes down (or view it as E[Quality] = (p=gpt4[code])*1.0 + (p=0.25)*0.0))
        if self.query_strategy in [
            QueryStrategy.CODE_GEN_WITH_FALLBACK,
            QueryStrategy.CODE_GEN,
        ]:
            quality = quality * (GPT_4_MODEL_CARD["code"] / 100.0)

        if self.token_budget is not None:
            quality = quality * math.sqrt(
                math.sqrt(self.token_budget)
            )  # now assume quality is proportional to sqrt(token_budget)

        costEst = {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

        return costEst, {"cumulative": costEst, "thisPlan": costEst, "subPlan": None}


class InduceFromCandidateOp(InduceOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self) -> IteratorFn:
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="induce", shouldProfile=self.shouldProfile)
        def iteratorFn():
            for nextCandidate in self.source:
                resultRecordList = self._attemptMapping(nextCandidate)
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


class ParallelInduceFromCandidateOp(InduceOp):
    def __init__(self, streaming, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 32  # TODO hardcoded for now
        self.streaming = streaming

    def __eq__(self, other: PhysicalOp):
        return super().__eq__(other) and self.streaming == other.streaming

    def copy(self):
        return super().copy(streaming=self.streaming)

    def __iter__(self):
        # This is very crudely implemented right now, since we materialize everything
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_induce", shouldProfile=self.shouldProfile)
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
                        executor.map(self._attemptMapping, inputs[i : i + chunksize])
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


class SimpleTypeConvert(InduceOp):
    """This is a very simple function that converts a DataRecord from one Schema to another, when we know they have identical fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.inputSchema == self.outputSchema
        ), "This convert has to be instantiated to convert an input to the same output Schema!"

    def __call__(self, candidate: DataRecord):
        if not candidate.schema == self.inputSchema:
            return None

        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        for field in self.outputSchema.fieldNames():  # type: ignore
            if hasattr(candidate, field):
                setattr(dr, field, getattr(candidate, field))
            elif field.required:
                return None

        # TODO profiling should be done somewhere else
        # if profiling, set record's stats for the given op_id to be an empty Stats object
        # if self.shouldProfile:
        # candidate._stats[td.op_id] = InduceNonLLMStats()

        return [dr], None