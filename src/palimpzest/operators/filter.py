from __future__ import annotations

from palimpzest.generators.generators import DSPyGenerator, ImageTextGenerator
from .physical import PhysicalOperator

from palimpzest.constants import *
from palimpzest.dataclasses import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.elements import DataRecord, DataRecordSet, Filter
from palimpzest.prompts import IMAGE_FILTER_PROMPT

from io import BytesIO
from PIL import Image
from typing import List, Optional

import base64
import time


class FilterOp(PhysicalOperator):
    def __init__(self, filter: Filter, depends_on: Optional[List[str]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.inputSchema == self.outputSchema, "Input and output schemas must match for FilterOp"
        self.filter = filter
        self.depends_on = depends_on if depends_on is None else sorted(depends_on)

    def __str__(self):
        op = super().__str__()
        op += f"    Filter: {str(self.filter)}\n"
        return op

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {"filter": self.filter, "depends_on": self.depends_on, **copy_kwargs}

    def get_op_params(self):
        return {
            "outputSchema": self.outputSchema,
            "filter": self.filter.getFilterStr(),
            "depends_on": self.depends_on,
        }

    def __eq__(self, other: FilterOp):
        return (
            isinstance(other, self.__class__)
            and self.filter == other.filter
            and self.inputSchema == other.inputSchema
            and self.outputSchema == other.outputSchema
        )


class NonLLMFilter(FilterOp):

    def __eq__(self, other: NonLLMFilter):
        return (
            isinstance(other, self.__class__)
            and self.filter == other.filter
            and self.outputSchema == other.outputSchema
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate 1 ms single-threaded execution for filter function
        time_per_record = 0.001 / self.max_workers

        # assume filter fn has perfect quality
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=time_per_record,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        # apply filter to input record
        start_time = time.time()
        try:
            result = self.filter.filterFn(candidate)
        except Exception as e:
            print(f"Error invoking user-defined function for filter: {e}")

        # time spent executing the filter function
        fn_call_duration_secs = time.time() - start_time

        # create copy of candidate and set _passed_operator attribute
        dr = DataRecord.fromParent(candidate.schema, parent_record=candidate)
        dr._passed_operator = result

        # create RecordOpStats object and return
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_id=dr._parent_id,
            record_source_id=dr._source_id,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=fn_call_duration_secs,
            cost_per_record=0.0,
            filter_str=self.filter.getFilterStr(),
            passed_operator=result,
            fn_call_duration_secs=fn_call_duration_secs,
            answer=result,
            op_details={k: str(v) for k, v in self.get_op_params().items()},
        )

        if self.verbose:
            output_str = f"{self.filter.getFilterStr()}:\n{result}"
            print(output_str)

        return DataRecordSet([dr], [record_op_stats])


class LLMFilter(FilterOp):

    def __init__(
        self,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_BOOL,
        image_filter: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.image_filter = image_filter

        doc_schema = str(self.inputSchema)
        doc_type = self.inputSchema.className()
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            if not self.image_filter:
                self.generator = DSPyGenerator(
                    self.model,
                    self.prompt_strategy,
                    doc_schema,
                    doc_type,
                    verbose=self.verbose,
                )
            else:
                self.generator = ImageTextGenerator(self.model, self.verbose)

        else:
            raise Exception(f"Prompt strategy {self.prompt_strategy} not implemented yet")

    def get_copy_kwargs(self):
        copy_kwargs = super().get_copy_kwargs()
        return {
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "image_filter": self.image_filter,
            **copy_kwargs
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {"model": self.model, **op_params}

        return op_params

    def __eq__(self, other: LLMFilter):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.filter == other.filter
            and self.prompt_strategy == other.prompt_strategy
            and self.image_filter == other.image_filter
            and self.outputSchema == other.outputSchema
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates):
        # estimate number of input tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS
        if self.image_filter:
            est_num_input_tokens = 765 / 10 # 1024x1024 image is 765 tokens

        # NOTE: in truth, the DSPy COT output often generates an entire reasoning sentence,
        #       thus the true value may be higher
        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"]
            * est_num_output_tokens
        ) / self.max_workers

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = NAIVE_EST_FILTER_SELECTIVITY
        cardinality = selectivity * source_op_cost_estimates.cardinality

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["overall"] / 100.0) * source_op_cost_estimates.quality

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        start_time = time.time()

        # parse the content from the candidate record
        content = None
        if self.image_filter:
            base64_images = []
            if hasattr(candidate, "contents"):
                # TODO: should address this now; we need a way to infer (or have the programmer declare) what fields contain image content
                base64_images = [
                    base64.b64encode(candidate.contents).decode("utf-8")  
                ]
            elif self.model in [Model.GPT_4o_V, Model.GPT_4o_MINI_V]:
                for image_file in candidate.image_filepaths:  # TODO: (see note above)
                    image = Image.open(image_file)
                    buffered = BytesIO()
                    image.save(buffered, format=image.format)
                    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    base64_images.append(base64_image)

            # for LLAMA vision model, we must concatenate images into a single image
            elif self.model in [Model.LLAMA3_V]:
                # load images, get their dimensions, and create new image to fit them horizontally
                images = [Image.open(image_file) for image_file in candidate.image_filepaths]
                widths, heights = zip(*(img.size for img in images))
                total_width, max_height = sum(widths), max(heights)
                new_image = Image.new(images[0].mode, (total_width, max_height))

                # construct new image by pasting images side-by-side
                x_offset = 0
                for img in images:
                    new_image.paste(img, (x_offset,0))
                    x_offset += img.size[0]

                # encode new image in base64
                buffered = BytesIO()
                new_image.save(buffered, format=images[0].format)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_images.append(base64_image)

            content = base64_images
        else:
            content = candidate._asJSONStr(include_bytes=False, project_cols=self.depends_on)

        # construct the prompt; for image filters we need to wrap the filter condition in an instruction 
        prompt = self.filter.filterCondition
        if self.image_filter:
            prompt = IMAGE_FILTER_PROMPT.format(filter_condition=self.filter.filterCondition)

        # invoke LLM to generate filter decision (True or False)
        response, gen_stats = None, GenerationStats()
        try:
            response, _, gen_stats = self.generator.generate(context=content, question=prompt)
        except Exception as e:
            print(f"Error invoking LLM for filter: {e}")

        # compute whether the record passed the filter or not
        passed_operator = (
            "true" in response.lower()
            if response is not None
            else False
        )

        # create new DataRecord and set _passed_operator attribute
        dr = DataRecord.fromParent(candidate.schema, parent_record=candidate)
        dr._passed_operator = passed_operator

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_id=dr._parent_id,
            record_source_id=dr._source_id,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=gen_stats.cost_per_record,
            model_name=self.model.value,
            filter_str=self.filter.getFilterStr(),
            total_input_tokens=gen_stats.total_input_tokens,
            total_output_tokens=gen_stats.total_output_tokens,
            total_input_cost=gen_stats.total_input_cost,
            total_output_cost=gen_stats.total_output_cost,
            llm_call_duration_secs=gen_stats.llm_call_duration_secs,
            answer=response,
            passed_operator=passed_operator,
            image_operation=self.image_filter,
            op_details={k: str(v) for k, v in self.get_op_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])
