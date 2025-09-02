"""This script contains tests for physical operators for map."""

import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.core.models import GenerationStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineConvert
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsConvert
from palimpzest.query.operators.rag import RAGConvert
from palimpzest.query.operators.split import SplitConvert

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

class OutputSchema(BaseModel):
    animal: str = Field(description="The animal in the input")

def mock_generator_call(candidate, fields, right_candidate=None, json_output=True, **kwargs):
    field_answers = {"animal": ["elephant"]}
    reasoning = "The input matches that of an elephant."
    generation_stats = GenerationStats(cost_per_record=1.0, time_per_record=1.0, num_input_tokens=10, num_output_tokens=10)
    messages = []
    return field_answers, reasoning, generation_stats, messages


@pytest.mark.parametrize(
    argnames=("input_schema", "output_schema", "physical_op_class"),
    argvalues=[
        pytest.param(TextInputSchema, OutputSchema, LLMConvertBonded, id="text-only-llm-convert-bonded"),
        pytest.param(ImageInputSchema, OutputSchema, LLMConvertBonded, id="image-only-llm-convert-bonded"),
        pytest.param(AudioInputSchema, OutputSchema, LLMConvertBonded, id="audio-only-llm-convert-bonded"),
        pytest.param(TextImageInputSchema, OutputSchema, LLMConvertBonded, id="text-image-llm-convert-bonded"),
        pytest.param(TextAudioInputSchema, OutputSchema, LLMConvertBonded, id="text-audio-llm-convert-bonded"),
        pytest.param(ImageAudioInputSchema, OutputSchema, LLMConvertBonded, id="image-audio-llm-convert-bonded"),
        pytest.param(TextImageAudioInputSchema, OutputSchema, LLMConvertBonded, id="text-image-audio-llm-convert-bonded"),
        pytest.param(TextInputSchema, OutputSchema, RAGConvert, id="text-only-rag-convert"),
        pytest.param(TextInputSchema, OutputSchema, SplitConvert, id="text-only-split-convert"),
        pytest.param(TextInputSchema, OutputSchema, CritiqueAndRefineConvert, id="text-only-critique-and-refine-convert"),
        pytest.param(ImageInputSchema, OutputSchema, CritiqueAndRefineConvert, id="image-only-critique-and-refine-convert"),
        pytest.param(AudioInputSchema, OutputSchema, CritiqueAndRefineConvert, id="audio-only-critique-and-refine-convert"),
        pytest.param(TextImageInputSchema, OutputSchema, CritiqueAndRefineConvert, id="text-image-critique-and-refine-convert"),
        pytest.param(TextAudioInputSchema, OutputSchema, CritiqueAndRefineConvert, id="text-audio-critique-and-refine-convert"),
        pytest.param(ImageAudioInputSchema, OutputSchema, CritiqueAndRefineConvert, id="image-audio-critique-and-refine-convert"),
        pytest.param(TextImageAudioInputSchema, OutputSchema, CritiqueAndRefineConvert, id="text-image-audio-critique-and-refine-convert"),
        pytest.param(TextInputSchema, OutputSchema, MixtureOfAgentsConvert, id="text-only-mixture-of-agents-convert"),
        pytest.param(ImageInputSchema, OutputSchema, MixtureOfAgentsConvert, id="image-only-mixture-of-agents-convert"),
        pytest.param(AudioInputSchema, OutputSchema, MixtureOfAgentsConvert, id="audio-only-mixture-of-agents-convert"),
        pytest.param(TextImageInputSchema, OutputSchema, MixtureOfAgentsConvert, id="text-image-mixture-of-agents-convert"),
        pytest.param(TextAudioInputSchema, OutputSchema, MixtureOfAgentsConvert, id="text-audio-mixture-of-agents-convert"),
        pytest.param(ImageAudioInputSchema, OutputSchema, MixtureOfAgentsConvert, id="image-audio-mixture-of-agents-convert"),
        pytest.param(TextImageAudioInputSchema, OutputSchema, MixtureOfAgentsConvert, id="text-image-audio-mixture-of-agents-convert"),
    ]
)
def test_map(mocker, input_schema, output_schema, physical_op_class):
    """Test map operators on simple input"""
    # construct the kwargs for the physical operator
    physical_op_kwargs = {"input_schema": input_schema, "output_schema": output_schema, "logical_op_id": "test-map"}
    if physical_op_class is LLMConvertBonded:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
    elif physical_op_class is RAGConvert:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
        physical_op_kwargs["num_chunks_per_field"] = 1
        physical_op_kwargs["chunk_size"] = 1000
    elif physical_op_class is SplitConvert:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
        physical_op_kwargs["num_chunks"] = 2
        physical_op_kwargs["min_size_to_chunk"] = 1000
    elif physical_op_class is MixtureOfAgentsConvert:
        physical_op_kwargs["proposer_models"] = [Model.GEMINI_2_5_PRO, Model.GEMINI_2_0_FLASH]
        physical_op_kwargs["temperatures"] = [0.8, 0.8]
        physical_op_kwargs["aggregator_model"] = Model.GEMINI_2_5_FLASH
    elif physical_op_class is CritiqueAndRefineConvert:
        physical_op_kwargs["model"] = Model.GEMINI_2_5_FLASH
        physical_op_kwargs["critic_model"] = Model.GEMINI_2_0_FLASH
        physical_op_kwargs["refine_model"] = Model.GEMINI_2_5_PRO

    # create map operator
    map_op = physical_op_class(**physical_op_kwargs)

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

    # apply map operator to the input
    data_record_set = map_op(input_record)

    # check for single output record with expected fields
    assert len(data_record_set) == 1
    output_record = data_record_set[0]

    assert sorted(output_record._schema.model_fields) == sorted(union_schemas([input_schema, output_schema]).model_fields)
    assert hasattr(output_record, "animal")
    assert output_record.animal == "elephant"
