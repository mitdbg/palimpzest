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
    argnames=("input_schema", "physical_op_class"),
    argvalues=[
        pytest.param(TextInputSchema, LLMFilter, id="text-only-llm-filter"),
        pytest.param(ImageInputSchema, LLMFilter, id="image-only-llm-filter"),
        pytest.param(AudioInputSchema, LLMFilter, id="audio-only-llm-filter"),
        pytest.param(TextImageInputSchema, LLMFilter, id="text-image-llm-filter"),
        pytest.param(TextAudioInputSchema, LLMFilter, id="text-audio-llm-filter"),
        pytest.param(ImageAudioInputSchema, LLMFilter, id="image-audio-llm-filter"),
        pytest.param(TextImageAudioInputSchema, LLMFilter, id="text-image-audio-llm-filter"),
        pytest.param(TextInputSchema, RAGFilter, id="text-only-rag-filter"),
        pytest.param(TextInputSchema, SplitFilter, id="text-only-split-filter"),
        pytest.param(TextInputSchema, CritiqueAndRefineFilter, id="text-only-critique-and-refine-filter"),
        pytest.param(ImageInputSchema, CritiqueAndRefineFilter, id="image-only-critique-and-refine-filter"),
        pytest.param(AudioInputSchema, CritiqueAndRefineFilter, id="audio-only-critique-and-refine-filter"),
        pytest.param(TextImageInputSchema, CritiqueAndRefineFilter, id="text-image-critique-and-refine-filter"),
        pytest.param(TextAudioInputSchema, CritiqueAndRefineFilter, id="text-audio-critique-and-refine-filter"),
        pytest.param(ImageAudioInputSchema, CritiqueAndRefineFilter, id="image-audio-critique-and-refine-filter"),
        pytest.param(TextImageAudioInputSchema, CritiqueAndRefineFilter, id="text-image-audio-critique-and-refine-filter"),
        pytest.param(TextInputSchema, MixtureOfAgentsFilter, id="text-only-mixture-of-agents-filter"),
        pytest.param(ImageInputSchema, MixtureOfAgentsFilter, id="image-only-mixture-of-agents-filter"),
        pytest.param(AudioInputSchema, MixtureOfAgentsFilter, id="audio-only-mixture-of-agents-filter"),
        pytest.param(TextImageInputSchema, MixtureOfAgentsFilter, id="text-image-mixture-of-agents-filter"),
        pytest.param(TextAudioInputSchema, MixtureOfAgentsFilter, id="text-audio-mixture-of-agents-filter"),
        pytest.param(ImageAudioInputSchema, MixtureOfAgentsFilter, id="image-audio-mixture-of-agents-filter"),
        pytest.param(TextImageAudioInputSchema, MixtureOfAgentsFilter, id="text-image-audio-mixture-of-agents-filter"),
    ]
)
def test_filter(mocker, input_schema, physical_op_class):
    """Test filter operators on simple input"""
    # construct the kwargs for the physical operator
    filter = Filter(filter_condition="The animal is an elephant.")
    physical_op_kwargs = {"input_schema": input_schema, "output_schema": input_schema, "filter": filter, "logical_op_id": "test-filter"}
    if physical_op_class is LLMFilter:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
    elif physical_op_class is RAGFilter:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
        physical_op_kwargs["num_chunks_per_field"] = 1
        physical_op_kwargs["chunk_size"] = 1000
    elif physical_op_class is SplitFilter:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
        physical_op_kwargs["num_chunks"] = 2
        physical_op_kwargs["min_size_to_chunk"] = 1000
    elif physical_op_class is MixtureOfAgentsFilter:
        physical_op_kwargs["proposer_models"] = [Model.GEMINI_2_5_PRO, Model.GEMINI_2_0_FLASH]
        physical_op_kwargs["temperatures"] = [0.8, 0.8]
        physical_op_kwargs["aggregator_model"] = Model.GEMINI_2_5_FLASH
    elif physical_op_class is CritiqueAndRefineFilter:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
        physical_op_kwargs["critic_model"] = Model.GEMINI_2_0_FLASH
        physical_op_kwargs["refine_model"] = Model.GEMINI_2_5_PRO

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
