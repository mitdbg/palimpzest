"""This script contains tests for physical operators for filter."""

import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.core.models import GenerationStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineFilter
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsFilter
from palimpzest.query.operators.rag import RAGFilter
from palimpzest.query.operators.split import SplitFilter

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class TextInputSchema(BaseModel):
    text: str = Field(description="Description of an animal")
    age: int = Field(description="The age of the animal in years")

class ImageInputSchema(BaseModel):
    image_file: ImageFilepath = Field(description="File path to an image of an animal")
    height: float = Field(description="The estimated height of the animal in cm")

class AudioInputSchema(BaseModel):
    audio_file: AudioFilepath = Field(description="File path to an audio recording of an animal")
    year: float = Field(description="The year the recording was made")

TextImageInputSchema = union_schemas([TextInputSchema, ImageInputSchema])
TextAudioInputSchema = union_schemas([TextInputSchema, AudioInputSchema])
ImageAudioInputSchema = union_schemas([ImageInputSchema, AudioInputSchema])
TextImageAudioInputSchema = union_schemas([TextInputSchema, ImageInputSchema, AudioInputSchema])

def mock_generator_call(candidate, fields, right_candidate=None, json_output=True, **kwargs):
    field_answers = {"passed_operator": True}
    reasoning = "The input matches that of an elephant."
    generation_stats = GenerationStats(cost_per_record=1.0, time_per_record=1.0, num_input_tokens=10, num_output_tokens=10)
    messages = []
    return field_answers, reasoning, generation_stats, messages


@pytest.mark.parametrize(
    "input_schema",
    [TextInputSchema, ImageInputSchema, AudioInputSchema, TextImageInputSchema, TextAudioInputSchema, ImageAudioInputSchema, TextImageAudioInputSchema],
    ids=["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
)
@pytest.mark.parametrize(
    "physical_op_class",
    [LLMFilter, RAGFilter, SplitFilter, CritiqueAndRefineFilter, MixtureOfAgentsFilter],
    ids=["llm-filter", "rag-filter", "split-filter", "critique-and-refine-filter", "mixture-of-agents-filter"],
)
def test_filter(mocker, input_schema, physical_op_class):
    """Test filter operators on simple input"""
    # RAGFilter and SplitFilter only support text input currently
    if physical_op_class in [RAGFilter, SplitFilter] and input_schema != TextInputSchema:
        pytest.skip(f"{physical_op_class} only supports text input currently")

    if os.getenv("CI") and input_schema in [AudioInputSchema, TextAudioInputSchema, ImageAudioInputSchema, TextImageAudioInputSchema]:
        pytest.skip("Skipping multi-modal audio tests on CI which does not have access to gemini models")

    model = Model.GPT_5_MINI if os.getenv("CI") else Model.GEMINI_2_5_FLASH
    proposer_models = [Model.GPT_5, Model.GPT_5_NANO] if os.getenv("CI") else [Model.GEMINI_2_5_PRO, Model.GEMINI_2_0_FLASH]
    critic_model = Model.GPT_5_NANO if os.getenv("CI") else Model.GEMINI_2_0_FLASH
    refine_model = Model.GPT_5 if os.getenv("CI") else Model.GEMINI_2_5_PRO

    # construct the kwargs for the physical operator
    filter = Filter(filter_condition="The animal is an elephant.")
    physical_op_kwargs = {"input_schema": input_schema, "output_schema": input_schema, "filter": filter, "logical_op_id": "test-filter"}
    if physical_op_class is LLMFilter:
        physical_op_kwargs["model"] = model
    elif physical_op_class is RAGFilter:
        physical_op_kwargs["model"] = model
        physical_op_kwargs["num_chunks_per_field"] = 1
        physical_op_kwargs["chunk_size"] = 1000
    elif physical_op_class is SplitFilter:
        physical_op_kwargs["model"] = model
        physical_op_kwargs["num_chunks"] = 2
        physical_op_kwargs["min_size_to_chunk"] = 1000
    elif physical_op_class is MixtureOfAgentsFilter:
        physical_op_kwargs["proposer_models"] = proposer_models
        physical_op_kwargs["temperatures"] = [0.8, 0.8]
        physical_op_kwargs["aggregator_model"] = model
    elif physical_op_class is CritiqueAndRefineFilter:
        physical_op_kwargs["model"] = model
        physical_op_kwargs["critic_model"] = critic_model
        physical_op_kwargs["refine_model"] = refine_model

    # create filter operator
    filter_op = physical_op_class(**physical_op_kwargs)

    # create input record
    input_record = DataRecord(schema=input_schema, source_indices=[0])
    if all(field in input_schema.model_fields for field in TextInputSchema.model_fields):
        input_record['text'] = "An elephant is a large gray animal with a trunk and big ears."
        input_record['age'] = 3
    if all(field in input_schema.model_fields for field in ImageInputSchema.model_fields):
        input_record.image_file = "tests/pytest/data/elephant.png"
        input_record.height = 304.5
    if all(field in input_schema.model_fields for field in AudioInputSchema.model_fields):
        input_record.audio_file = "tests/pytest/data/elephant.wav"
        input_record.year = 2020

    # only execute LLM calls when running on CI for merge to main
    if not os.getenv("CI"):
        mocker.patch.object(Generator, "__call__", side_effect=mock_generator_call)

    # apply filter operator to the input
    data_record_set = filter_op(input_record)

    # check for single output record with expected fields
    assert len(data_record_set) == 1
    output_record = data_record_set[0]

    assert sorted(output_record._schema.model_fields) == sorted(input_schema.model_fields)
    assert output_record._passed_operator
