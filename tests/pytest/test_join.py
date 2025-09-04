"""This script contains tests for physical operators for join."""

import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.core.models import GenerationStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.join import EmbeddingJoin, NestedLoopsJoin

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
    [NestedLoopsJoin, EmbeddingJoin],
    ids=["nested-loops-join", "embedding-join"],
)
def test_join(mocker, left_input_schema, right_input_schema, physical_op_class):
    """Test join operators on simple input"""
    # RAGConvert and SplitConvert only support text input currently
    left_has_audio = any(field in left_input_schema.model_fields for field in AudioInputSchema.model_fields)
    right_has_audio = any(field in right_input_schema.model_fields for field in AudioInputSchema.model_fields)
    if physical_op_class in [EmbeddingJoin] and (left_has_audio or right_has_audio):
        pytest.skip(f"{physical_op_class} does not support audio input currently")

    # construct the kwargs for the physical operator
    input_schema = union_schemas([left_input_schema, right_input_schema])
    physical_op_kwargs = {
        "input_schema": input_schema,
        "output_schema": input_schema,
        "condition": "Do the two inputs describe the same type of animal?",
        "logical_op_id": "test-join",
        "model": Model.GEMINI_2_5_FLASH,
    }
    if physical_op_class == EmbeddingJoin:
        physical_op_kwargs["num_samples"] = 10

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

def test_embedding_join(mocker):
    """Test EmbeddingJoin operator on simple text input"""
    left_candidates = []
    for left_idx, animal in enumerate(["elephant", "lion", "lion", "bear"]):
        left_input_record = DataRecord(schema=TextInputSchema, source_indices=[left_idx])
        left_input_record['text'] = f"This text describes a {animal}."
        left_input_record['age'] = left_idx + 1
        left_candidates.append(left_input_record)

    right_candidates = []
    for right_idx, animal in enumerate(["elephant", "giraffe", "lion", "zebra"]):
        right_input_record = DataRecord(schema=TextInputSchema, source_indices=[right_idx])
        right_input_record['text'] = f"This text describes a {animal}."
        right_input_record['age'] = right_idx + 2
        right_candidates.append(right_input_record)

    # construct the kwargs for the physical operator
    input_schema = union_schemas([TextInputSchema, TextInputSchema])
    physical_op_kwargs = {
        "input_schema": input_schema,
        "output_schema": input_schema,
        "condition": "Do the two inputs describe the same type of animal?",
        "logical_op_id": "test-join",
        "model": Model.GEMINI_2_5_FLASH,
        "num_samples": 8,
    }

    # create join operator
    join_op = EmbeddingJoin(**physical_op_kwargs)

    # only execute LLM calls when running on CI for merge to main
    if not os.getenv("CI"):
        mock_call = mocker.patch.object(Generator, "__call__", side_effect=mock_generator_call)

    # apply join operator to the inputs
    data_record_set, num_inputs_processed = join_op(left_candidates, right_candidates)

    # check that the mock was called 8 times (num_samples)
    if not os.getenv("CI"):
        assert mock_call.call_count == 8

    # sanity checks on output records and stats
    records = data_record_set.data_records
    record_op_stats_lst = data_record_set.record_op_stats
    assert len(record_op_stats_lst) == 16
    assert num_inputs_processed == 16
    for output_record in records:
        assert sorted(output_record._schema.model_fields) == sorted(input_schema.model_fields)

    # check that all output record stats have embedding stats
    assert all(stats.total_embedding_cost > 0.0 for stats in record_op_stats_lst)
    assert sum(record._passed_operator for record in records) == 3
