import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.generators.generators import Generator


@pytest.fixture
def question():
    class Question(BaseModel):
        question: str = Field(description="A simple question")
    dr = DataRecord(schema=Question, source_idx=0)
    dr.question = "What color is grass? (one-word answer)"
    return dr

@pytest.fixture
def output_schema():
    class Answer(BaseModel):
        answer: str = Field(description="The one-word answer to the question.")
    return Answer

@pytest.mark.parametrize("model", [Model.GPT_4o_MINI, Model.DEEPSEEK_V3, Model.LLAMA3_2_3B, Model.MIXTRAL])
def test_generator(model, question, output_schema):
    generator = Generator(model, PromptStrategy.COT_QA)
    output, _, gen_stats, _ = generator(question, output_schema.model_fields, **{"output_schema": output_schema})
    assert gen_stats.total_input_tokens > 0
    assert gen_stats.total_output_tokens > 0
    assert output["answer"][0].lower() == "green"
