import os

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.generators.generators import Generator


@pytest.fixture
def question():
    class Question(BaseModel):
        question: str = Field(description="A simple question")
    dr = DataRecord(data_item=Question(question="What color is grass? (one-word answer)"), source_indices=[0])
    return dr

@pytest.fixture
def output_schema():
    class Answer(BaseModel):
        answer: str = Field(description="The one-word answer to the question.")
    return Answer

@pytest.mark.parametrize(
    "model",
    [
        pytest.param(Model.GPT_4o_MINI, marks=pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not present")),
        pytest.param(Model.DEEPSEEK_V3, marks=pytest.mark.skipif(os.getenv("TOGETHER_API_KEY") is None, reason="TOGETHER_API_KEY not present")),
        pytest.param(Model.LLAMA3_2_3B, marks=pytest.mark.skipif(os.getenv("TOGETHER_API_KEY") is None, reason="TOGETHER_API_KEY not present")),
        pytest.param(Model.CLAUDE_3_5_HAIKU, marks=pytest.mark.skipif(os.getenv("ANTHROPIC_API_KEY") is None, reason="ANTHROPIC_API_KEY not present")),
    ]
)
def test_generator(model, question, output_schema):
    generator = Generator(model, PromptStrategy.MAP, None)
    output, _, gen_stats, _ = generator(question, output_schema.model_fields, **{"output_schema": output_schema})
    assert (gen_stats.total_input_tokens + gen_stats.total_cache_read_tokens + gen_stats.total_cache_creation_tokens) > 0
    assert gen_stats.total_output_tokens > 0
    assert output["answer"][0].lower() == "green"
