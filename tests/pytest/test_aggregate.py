"""This script contains tests for physical operators for semantic aggregation."""

import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import CuratedModel
from palimpzest.utils.model_info import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath, union_schemas
from palimpzest.core.models import GenerationStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.aggregate import SemanticAggregate

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
    num_elephants: int = Field(description="The number of (possibly duplicate) elephants in the input")

def create_input_record(input_schema: type[BaseModel], idx: int) -> DataRecord:
    idx_to_elephant_name = {0: "Dumbo", 1: "Ella", 2: "Babar"}
    idx_to_elephant_height = {0: 250.0, 1: 300.5, 2: 350.2}
    idx_to_elephant_year = {0: 2018, 1: 2019, 2: 2020}
    data_item = {}
    if all(field in input_schema.model_fields for field in TextInputSchema.model_fields):
        data_item['text'] = f"This record contains the age of an elephant named {idx_to_elephant_name[idx]}."
        data_item['age'] = idx + 1
    if all(field in input_schema.model_fields for field in ImageInputSchema.model_fields):
        data_item['image_file'] = "tests/pytest/data/elephant.png"
        data_item['height'] = idx_to_elephant_height[idx]
    if all(field in input_schema.model_fields for field in AudioInputSchema.model_fields):
        data_item['audio_file'] = "tests/pytest/data/elephant.wav"
        data_item['year'] = idx_to_elephant_year[idx]

    return DataRecord(input_schema(**data_item), source_indices=[idx])


def mock_generator_call(candidate, fields, right_candidate=None, json_output=True, **kwargs):
    field_answers = {"num_elephants": [3]}
    reasoning = "The input shows three elephants."
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
    [SemanticAggregate],
    ids=["semantic-aggregate"],
)
def test_aggregate(mocker, input_schema, physical_op_class):
    """Test aggregate operators on simple input"""
    if os.getenv("NO_GEMINI") and input_schema in [AudioInputSchema, TextAudioInputSchema, ImageAudioInputSchema, TextImageAudioInputSchema]:
        pytest.skip("Skipping multi-modal audio tests on CI which does not have access to gemini models")

    model = Model(CuratedModel.GPT_5_MINI) if os.getenv("NO_GEMINI") else Model(CuratedModel.GEMINI_2_5_FLASH)

    # construct the kwargs for the physical operator
    physical_op_kwargs = {
        "input_schema": input_schema,
        "output_schema": OutputSchema,
        "agg_str": "The number of (possibly duplicate) elephants in the input",
        "logical_op_id": "test-aggregate",
        "model": model,
    }

    # create filter operator
    agg_op = physical_op_class(**physical_op_kwargs)

    # create input records
    input_records = [create_input_record(input_schema, idx) for idx in range(3)]

    # only execute LLM calls if specified
    if not os.getenv("RUN_LLM_TESTS"):
        mocker.patch.object(Generator, "__call__", side_effect=mock_generator_call)

    # apply filter operator to the input
    data_record_set = agg_op(input_records)

    # check for single output record with expected fields
    assert len(data_record_set) == 1
    output_record = data_record_set[0]

    assert list(output_record.schema.model_fields) == ["num_elephants"]
    assert output_record.num_elephants == 3
