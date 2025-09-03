"""This script contains tests for physical operators for join."""

import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.core.models import GenerationStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.join import BlockingNestedLoopsJoin, NestedLoopsJoin

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

def create_input_record(schema: type[BaseModel]) -> DataRecord:
    input_record = DataRecord(schema=schema, source_indices=[0])
    if all(field in schema.model_fields for field in TextInputSchema.model_fields):
        input_record['text'] = "An elephant is a large gray animal with a trunk and big ears."
        input_record['age'] = 3
    if all(field in schema.model_fields for field in ImageInputSchema.model_fields):
        input_record.image_file = "tests/pytest/data/elephant.png"
        input_record.height = 304.5
    if all(field in schema.model_fields for field in AudioInputSchema.model_fields):
        input_record.audio_file = "tests/pytest/data/elephant.wav"
        input_record.year = 2020

    return input_record

def mock_generator_call(candidate, fields, right_candidate=None, json_output=True, **kwargs):
    field_answers = {"passed_operator": True}
    reasoning = "The input matches that of an elephant."
    generation_stats = GenerationStats(cost_per_record=1.0, time_per_record=1.0, num_input_tokens=10, num_output_tokens=10)
    messages = []
    return field_answers, reasoning, generation_stats, messages


# TODO: test all joins without CI and with assert False
@pytest.mark.parametrize(
    "left_input_schema",
    [TextInputSchema, ImageInputSchema, AudioInputSchema, TextImageInputSchema, TextAudioInputSchema, ImageAudioInputSchema, TextImageAudioInputSchema],
    ids=["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
)
@pytest.mark.parametrize(
    "right_input_schema",
    [TextInputSchema, ImageInputSchema, AudioInputSchema, TextImageInputSchema, TextAudioInputSchema, ImageAudioInputSchema, TextImageAudioInputSchema],
    ids=["text-only", "image-only", "audio-only", "text-image", "text-audio", "image-audio", "text-image-audio"],
)
@pytest.mark.parametrize(
    "physical_op_class",
    [NestedLoopsJoin, BlockingNestedLoopsJoin],
    ids=["nested-loops-join", "blocking-nested-loops-join"],
)
def test_join(mocker, left_input_schema, right_input_schema, physical_op_class):
    """Test join operators on simple input"""
    # construct the kwargs for the physical operator
    input_schema = union_schemas([left_input_schema, right_input_schema])
    physical_op_kwargs = {
        "input_schema": input_schema,
        "output_schema": input_schema,
        "condition": "Do the two inputs describe the same type of animal?",
        "logical_op_id": "test-join",
        "model": Model.GEMINI_2_5_FLASH,
    }

    # create join operator
    join_op = physical_op_class(**physical_op_kwargs)

    # create left input record
    left_input_record = create_input_record(left_input_schema)
    right_input_record = create_input_record(right_input_schema)

    # only execute LLM calls when running on CI for merge to main
    if not os.getenv("CI"):
        mocker.patch.object(Generator, "__call__", side_effect=mock_generator_call)

    # apply join operator to the inputs
    data_record_set, num_inputs_processed = join_op([left_input_record], [right_input_record])

    # check for single output record with expected fields
    assert len(data_record_set) == 1
    assert num_inputs_processed == 1
    output_record = data_record_set[0]

    assert sorted(output_record._schema.model_fields) == sorted(input_schema.model_fields)
    assert output_record._passed_operator
