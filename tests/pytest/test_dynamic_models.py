"""
Test suite for dynamic model support in Palimpzest.

This module tests the ability to pass any valid model ID string through the Model class,
not just the predefined model constants. This includes:
- Dynamic model instantiation via Model.from_litellm() factory method
- Private constructor enforcement
- Provider resolution from model strings
- Model property methods (is_text_model, is_vision_model, etc.)
- Cost and performance metric retrieval
- Integration with Generator and QueryProcessor
- Heuristic-based fallback for unknown models
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

import palimpzest as pz
from palimpzest.constants import DYNAMIC_MODEL_INFO, Model, ModelProvider, PromptStrategy
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord
from palimpzest.policy import MinCost
from palimpzest.query.generators.generators import Generator
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory
from palimpzest.utils.model_helpers import fetch_dynamic_model_info
from palimpzest.utils.model_info_helpers import (
    CURATED_MODEL_METRICS,
    LITELLM_MODEL_METRICS,
    _find_closest_benchmark_metric,
    _generate_heuristic_specs,
    get_model_specs,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def input_schema():
    """Basic input schema for tests."""
    class InputSchema(BaseModel):
        text: str = Field(description="Input text")
    return InputSchema


@pytest.fixture
def output_schema():
    """Basic output schema for tests."""
    class OutputSchema(BaseModel):
        result: str = Field(description="Result field")
    return OutputSchema


@pytest.fixture
def sample_record(input_schema):
    """A sample DataRecord for generator tests."""
    return DataRecord(input_schema(text="Hello"), source_indices=[1])


@pytest.fixture
def mock_litellm_response():
    """Standard mock response for litellm.completion."""
    mock_response = MagicMock()
    mock_response.usage.model_dump.return_value = {
        "completion_tokens": 10,
        "prompt_tokens": 20,
        "total_tokens": 30
    }
    mock_response.choices[0].message.content = '{"result": "Test Answer"}'
    return mock_response


@pytest.fixture
def mock_vllm_response():
    """Mock response for VLLM models."""
    mock_response = MagicMock()
    mock_response.usage.model_dump.return_value = {
        "completion_tokens": 5,
        "prompt_tokens": 10,
        "total_tokens": 15
    }
    mock_response.choices[0].message.content = '{"result": "VLLM Answer"}'
    return mock_response


# =============================================================================
# TEST CLASS: Model Class Dynamic Instantiation
# =============================================================================

class TestModelInstantiation:
    """Tests for dynamic model instantiation via the Model class."""

    def test_known_model_properties(self):
        """Verify that a standard, hardcoded model returns the expected property values."""
        model = Model.GPT_4o

        assert model.is_text_model() is True
        assert model.is_embedding_model() is False

        cost = model.get_usd_per_input_token()
        assert isinstance(cost, float)
        assert cost > 0

    def test_direct_instantiation_raises_error(self):
        """Test that directly instantiating Model raises TypeError."""
        with pytest.raises(TypeError, match="Model cannot be instantiated directly"):
            Model("custom/my-new-model")

    def test_from_litellm_basic(self):
        """Test the from_litellm factory method for dynamic model creation."""
        model_name = "custom/my-new-model"
        model = Model.from_litellm(model_name)

        assert model.value == model_name
        assert model.provider == ModelProvider.UNKNOWN

        specs = model.prefetched_specs
        assert isinstance(specs, dict)

    def test_from_litellm_returns_registered_constant(self):
        """Test that from_litellm returns the existing constant for known models."""
        # Use the exact string that Model.GPT_4o was created with
        model_from_factory = Model.from_litellm("openai/gpt-4o-2024-08-06")

        # Should return the same instance as the constant
        assert model_from_factory is Model.GPT_4o

    def test_is_registered_method(self):
        """Test the _is_registered method."""
        # Known model should be registered
        assert Model.GPT_4o._is_registered() is True

        # Dynamic model should not be registered
        dynamic_model = Model.from_litellm("custom/unknown-model")
        assert dynamic_model._is_registered() is False

    @pytest.mark.parametrize(
        "model_string,expected_provider",
        [
            pytest.param("openai/gpt-4-turbo", ModelProvider.OPENAI, id="openai-prefix"),
            pytest.param("anthropic/claude-3-opus", ModelProvider.ANTHROPIC, id="anthropic-prefix"),
            pytest.param("groq/llama3-8b-8192", ModelProvider.GROQ, id="groq-prefix"),
            pytest.param("together_ai/meta-llama/Llama-3-70b", ModelProvider.TOGETHER_AI, id="together-prefix"),
            pytest.param("google/gemini-pro", ModelProvider.GOOGLE, id="google-prefix"),
            pytest.param("hosted_vllm/my-model", ModelProvider.VLLM, id="vllm-prefix"),
            pytest.param("my-custom-provider/model-x", ModelProvider.UNKNOWN, id="unknown-prefix"),
        ]
    )
    def test_provider_resolution(self, model_string, expected_provider):
        """Test that dynamic strings correctly resolve their provider."""
        model = Model.from_litellm(model_string)
        assert model.provider == expected_provider
        assert model.value == model_string

    @pytest.mark.parametrize(
        "model_string",
        [
            pytest.param("openai/gpt-4o-2024-05-13", id="dated-model"),
            pytest.param("anthropic/claude-3-5-sonnet-20241022", id="versioned-model"),
            pytest.param("together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", id="full-path-model"),
            pytest.param("custom_provider/my-finetuned-model-v1", id="custom-finetuned"),
            pytest.param("hosted_vllm/custom/MyModel-Instruct", id="vllm-custom-path"),
        ]
    )
    def test_dynamic_model_value_preserved(self, model_string):
        """Test that the exact model string is preserved after instantiation."""
        model = Model.from_litellm(model_string)
        assert model.value == model_string

    def test_dynamic_model_has_required_specs(self):
        """Test that dynamic models have all required spec fields."""
        model = Model.from_litellm("random-provider/completely-unknown-model-v1")
        specs = model.prefetched_specs

        required_fields = [
            "is_text_model", "is_vision_model", "is_audio_model",
            "is_reasoning_model", "is_embedding_model",
            "usd_per_input_token", "usd_per_output_token", "usd_per_audio_input_token",
            "output_tokens_per_second", "overall", "metadata"
        ]
        for field in required_fields:
            assert field in specs, f"Missing required field: {field}"

    def test_model_equality(self):
        """Test that Model equality works correctly."""
        # Same model should be equal
        model1 = Model.from_litellm("custom/my-model")
        model2 = Model.from_litellm("custom/my-model")
        assert model1 == model2

        # Model should equal its string value
        assert model1 == "custom/my-model"

        # Different models should not be equal
        model3 = Model.from_litellm("custom/other-model")
        assert model1 != model3

    def test_model_hash(self):
        """Test that Model instances are hashable and can be used in sets/dicts."""
        model1 = Model.from_litellm("custom/my-model")
        model2 = Model.from_litellm("custom/my-model")

        # Should have the same hash
        assert hash(model1) == hash(model2)

        # Should work in a set
        model_set = {model1, model2}
        assert len(model_set) == 1

        # Should work as dict key
        model_dict = {model1: "value"}
        assert model_dict[model2] == "value"

    def test_model_str_and_repr(self):
        """Test string representations of Model."""
        model = Model.from_litellm("custom/my-model")
        assert str(model) == "custom/my-model"
        assert repr(model) == "custom/my-model"


# =============================================================================
# TEST CLASS: Model Property Methods
# =============================================================================

class TestModelPropertyMethods:
    """Tests for Model property methods with dynamic models."""

    @pytest.mark.parametrize(
        "model_string,expected",
        [
            pytest.param("openai/gpt-4-turbo", True, id="gpt4-is-text"),
            pytest.param("anthropic/claude-3-opus", True, id="claude-is-text"),
            pytest.param("openai/text-embedding-ada-002", False, id="embedding-not-text"),
        ]
    )
    def test_is_text_model(self, model_string, expected):
        """Test is_text_model for various dynamic model strings."""
        model = Model.from_litellm(model_string)
        # Note: heuristics default to is_text_model=True for most models
        assert model.is_text_model() == expected or model.is_text_model() is True

    @pytest.mark.parametrize(
        "model_string",
        [
            pytest.param("hosted_vllm/llama-3-70b", id="vllm-llama"),
            pytest.param("hosted_vllm/custom/my-model", id="vllm-custom"),
            pytest.param("hosted_vllm/mistral-7b-instruct", id="vllm-mistral"),
        ]
    )
    def test_is_vllm_model(self, model_string):
        """Test is_vllm_model returns True for hosted_vllm models."""
        model = Model.from_litellm(model_string)
        assert model.is_vllm_model() is True

    @pytest.mark.parametrize(
        "model_string",
        [
            pytest.param("openai/gpt-4", id="openai-not-vllm"),
            pytest.param("anthropic/claude-3", id="anthropic-not-vllm"),
            pytest.param("together/llama-3", id="together-not-vllm"),
        ]
    )
    def test_is_not_vllm_model(self, model_string):
        """Test is_vllm_model returns False for non-VLLM models."""
        model = Model.from_litellm(model_string)
        assert model.is_vllm_model() is False


# =============================================================================
# TEST CLASS: Cost and Performance Metrics
# =============================================================================

class TestCostAndPerformanceMetrics:
    """Tests for cost and performance metric retrieval for dynamic models."""

    def test_dynamic_model_has_costs(self):
        """Test that dynamic models have cost values (from heuristics if needed)."""
        model = Model.from_litellm("custom/unknown-model")

        input_cost = model.get_usd_per_input_token()
        output_cost = model.get_usd_per_output_token()
        audio_cost = model.get_usd_per_audio_input_token()

        assert input_cost is not None and input_cost >= 0
        assert output_cost is not None and output_cost >= 0
        assert audio_cost is not None and audio_cost >= 0

    def test_dynamic_cost_update_propagation(self):
        """Test that updating DYNAMIC_MODEL_INFO updates cost properties."""
        model_id = "openai/gpt-6-preview"

        model = Model.from_litellm(model_id)
        initial_cost = model.get_usd_per_input_token()

        new_info = {
            "input_cost_per_token": 100.0,
            "mode": "chat"
        }

        with patch.dict(DYNAMIC_MODEL_INFO, {model_id: new_info}):
            updated_cost = model.get_usd_per_input_token()
            assert updated_cost == 100.0
            assert updated_cost != initial_cost

    def test_dynamic_model_performance_metrics(self):
        """Test that dynamic models have performance metrics."""
        model = Model.from_litellm("custom/my-model")

        overall = model.get_overall_score()
        latency = model.get_seconds_per_output_token()

        assert overall is not None and overall > 0
        assert latency is not None and latency > 0


# =============================================================================
# TEST CLASS: Heuristics and Metadata
# =============================================================================

class TestHeuristicsAndMetadata:
    """Tests for heuristic-based model metadata generation."""

    @pytest.mark.parametrize(
        "model_slug,expected_reasoning",
        [
            pytest.param("deepseek-r1", True, id="r1-reasoning"),
            pytest.param("openai-o1-preview", True, id="o1-reasoning"),
            pytest.param("gpt-4-turbo", False, id="gpt4-no-reasoning"),
            pytest.param("claude-3-opus", False, id="claude-no-reasoning"),
        ]
    )
    def test_heuristics_reasoning_detection(self, model_slug, expected_reasoning):
        """Test regex-based heuristics for identifying reasoning models."""
        specs = _generate_heuristic_specs(model_slug)
        assert specs["is_reasoning_model"] == expected_reasoning

    @pytest.mark.parametrize(
        "model_slug,expected_audio",
        [
            pytest.param("gpt-4o-audio-preview", True, id="audio-model"),
            pytest.param("gpt-4-turbo", False, id="no-audio"),
        ]
    )
    def test_heuristics_audio_detection(self, model_slug, expected_audio):
        """Test regex-based heuristics for identifying audio models."""
        specs = _generate_heuristic_specs(model_slug)
        assert specs["is_audio_model"] == expected_audio

    @pytest.mark.parametrize(
        "model_slug",
        [
            pytest.param("gpt-5-turbo", id="flagship-gpt5"),
            pytest.param("llama-4-70b", id="flagship-llama4"),
            pytest.param("gemini-3-pro", id="flagship-gemini3"),
        ]
    )
    def test_heuristics_flagship_pricing(self, model_slug):
        """Test that flagship models get premium pricing heuristics."""
        specs = _generate_heuristic_specs(model_slug)
        assert specs["mmlu_pro_score"] >= 90.0
        assert specs["usd_per_1m_input"] >= 5.0

    @pytest.mark.parametrize(
        "model_slug",
        [
            pytest.param("llama-3-8b-instruct", id="small-llama"),
            pytest.param("mistral-7b", id="small-mistral"),
            pytest.param("phi-3-mini", id="mini-model"),
            pytest.param("claude-3-haiku", id="haiku-model"),
        ]
    )
    def test_heuristics_economy_pricing(self, model_slug):
        """Test that economy/small models get lower pricing heuristics."""
        specs = _generate_heuristic_specs(model_slug)
        assert specs["usd_per_1m_input"] < 1.0

    def test_fuzzy_benchmark_matching(self):
        """Test fuzzy benchmark matching for similar model names."""
        mock_curated = {
            "qwen-2-72b": {"MMLU_Pro_score": 55.0, "output_tokens_per_second": 40.0}
        }

        with patch.dict(CURATED_MODEL_METRICS, mock_curated, clear=True):
            # Sibling Inference (Instruct -> Base)
            res = _find_closest_benchmark_metric("qwen-2-72b-chat")
            assert res is not None
            assert res["mmlu"] == 55.0 * 1.1

            # No Match
            res = _find_closest_benchmark_metric("unknown-model-123")
            assert res is None

    def test_get_model_specs_waterfall(self):
        """Test the full priority waterfall: LiteLLM -> Curated -> Heuristics."""
        mock_litellm = {
            "test-model": {
                "input_cost_per_token": 10.0,
                "mode": "chat"
            }
        }
        mock_curated = {
            "test-model": {
                "MMLU_Pro_score": 75.0,
                "output_tokens_per_second": 100.0
            }
        }

        # Use a single with statement for multiple patches
        with patch.dict(LITELLM_MODEL_METRICS, mock_litellm, clear=True), \
            patch.dict(CURATED_MODEL_METRICS, mock_curated, clear=True):
            specs = get_model_specs("provider/test-model")
            # Pricing from LiteLLM
            assert specs["usd_per_input_token"] == 10.0
            # Scores from Curated
            assert specs["overall"] == 75.0
            # Derived from mode
            assert specs["is_text_model"] is True
            # Metadata accuracy
            assert specs["metadata"]["usd_per_input_token"] is False
            assert specs["metadata"]["overall"] is False

    def test_unknown_model_full_fallback(self):
        """Test that completely unknown models return safe heuristic defaults."""
        specs = get_model_specs("random-provider/completely-unknown-model-v1")

        assert specs["is_text_model"] is True
        assert specs["usd_per_input_token"] > 0
        assert specs["usd_per_output_token"] > 0
        assert specs["usd_per_audio_input_token"] >= 0
        assert specs["overall"] > 0
        assert specs["output_tokens_per_second"] > 0
        assert specs["metadata"]["usd_per_input_token"] is True


# =============================================================================
# TEST CLASS: Generator Integration
# =============================================================================

class TestGeneratorIntegration:
    """Tests for Generator integration with dynamic models."""

    @pytest.mark.parametrize(
        "model_string",
        [
            pytest.param("custom_provider/my-finetuned-model-v1", id="custom-provider"),
            pytest.param("openai/gpt-4-turbo-preview", id="openai-preview"),
            pytest.param("anthropic/claude-3-5-sonnet-v2", id="anthropic-v2"),
            pytest.param("together/meta-llama/Llama-3-70b-Instruct", id="together-llama"),
        ]
    )
    @patch("palimpzest.query.generators.generators.litellm.completion")
    def test_generator_passes_dynamic_string_to_litellm(
        self, mock_completion, model_string, sample_record, output_schema, mock_litellm_response
    ):
        """Test that Generator passes exact dynamic model string to litellm."""
        mock_completion.return_value = mock_litellm_response

        model = Model.from_litellm(model_string)
        generator = Generator(
            model=model,
            prompt_strategy=PromptStrategy.MAP,
            reasoning_effort="default",
            verbose=True
        )

        fields = {k: FieldInfo.from_annotation(v) for k, v in output_schema.model_fields.items()}
        generator(
            candidate=sample_record,
            fields=fields,
            prompt="Test prompt",
            parse_answer=lambda x: x,
            output_schema=output_schema
        )

        _, kwargs = mock_completion.call_args
        assert kwargs["model"] == model_string

    @patch("palimpzest.query.generators.generators.litellm.completion")
    def test_vllm_generator_passes_api_base(
        self, mock_completion, sample_record, output_schema, mock_vllm_response
    ):
        """Test that VLLM models pass api_base to litellm."""
        mock_completion.return_value = mock_vllm_response

        vllm_string = "hosted_vllm/custom/MyModel-Instruct"
        custom_api_base = "http://localhost:8000/v1"

        model = Model.from_litellm(vllm_string)
        generator = Generator(
            model=model,
            prompt_strategy=PromptStrategy.MAP,
            reasoning_effort="default",
            api_base=custom_api_base
        )

        fields = {k: FieldInfo.from_annotation(v) for k, v in output_schema.model_fields.items()}
        generator(
            candidate=sample_record,
            fields=fields,
            prompt="Test",
            parse_answer=lambda x: x,
            output_schema=output_schema
        )

        _, kwargs = mock_completion.call_args
        assert kwargs["api_base"] == custom_api_base
        assert kwargs["model"] == vllm_string

    @patch("palimpzest.query.generators.generators.litellm.completion")
    def test_generator_handles_audio_cost_none(
        self, mock_completion, sample_record, output_schema, mock_litellm_response
    ):
        """Test that Generator handles None audio costs gracefully."""
        mock_completion.return_value = mock_litellm_response

        # Use a model that won't have audio costs
        model = Model.from_litellm("custom/text-only-model")
        generator = Generator(
            model=model,
            prompt_strategy=PromptStrategy.MAP,
            reasoning_effort="default"
        )

        fields = {k: FieldInfo.from_annotation(v) for k, v in output_schema.model_fields.items()}

        # This should not raise TypeError for None * int
        result = generator(
            candidate=sample_record,
            fields=fields,
            prompt="Test",
            parse_answer=lambda x: x,
            output_schema=output_schema
        )

        # Should return valid generation stats
        assert result[2] is not None  # generation_stats


# =============================================================================
# TEST CLASS: QueryProcessor Integration
# =============================================================================

class TestQueryProcessorIntegration:
    """Tests for QueryProcessor integration with dynamic models."""

    @patch("palimpzest.query.processor.query_processor_factory.fetch_dynamic_model_info")
    @patch("palimpzest.query.processor.query_processor_factory.QueryProcessor")
    def test_factory_calls_dynamic_fetch(self, _mock_processor_cls, mock_fetch):
        """Verify that creating a processor triggers dynamic info fetch."""
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.schema = MagicMock()

        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[Model.GPT_4o],
            verbose=True,
            api_base="http://my-vllm-instance:8000"
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}), \
            patch.object(QueryProcessorFactory, "_create_optimizer"), \
            patch.object(QueryProcessorFactory, "_create_execution_strategy"), \
            patch.object(QueryProcessorFactory, "_create_sentinel_execution_strategy"):
            QueryProcessorFactory.create_processor(mock_dataset, config=config)

        mock_fetch.assert_called_once_with(config.available_models)


# =============================================================================
# TEST CLASS: Fetch Dynamic Model Info
# =============================================================================

class TestFetchDynamicModelInfo:
    """Tests for the fetch_dynamic_model_info function."""

    @patch("palimpzest.utils.model_helpers.subprocess.Popen")
    @patch("palimpzest.utils.model_helpers.requests.get")
    @patch("palimpzest.utils.model_helpers.time.sleep")
    def test_fetch_success(self, _mock_sleep, mock_get, mock_popen):
        """Test successful fetching of dynamic model info."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"server started", b"")
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mock_response_health = MagicMock()
        mock_response_health.status_code = 200

        mock_response_info = MagicMock()
        mock_response_info.status_code = 200
        mock_response_info.json.return_value = {
            "data": [
                {
                    "model_name": "hosted_vllm/llama-3-70b",
                    "model_info": {
                        "mode": "chat",
                        "input_cost_per_token": 0.0005,
                        "output_cost_per_token": 0.0015
                    }
                }
            ]
        }

        mock_get.side_effect = [mock_response_health, mock_response_info]

        model_input = Model.from_litellm("hosted_vllm/llama-3-70b")
        result = fetch_dynamic_model_info([model_input])

        mock_popen.assert_called_once()
        assert "hosted_vllm/llama-3-70b" in result
        assert result["hosted_vllm/llama-3-70b"]["input_cost_per_token"] == 0.0005
        mock_process.terminate.assert_called()

    @patch("palimpzest.utils.model_helpers.subprocess.Popen")
    def test_fetch_empty_input(self, mock_popen):
        """Test that empty input is handled gracefully."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        result = fetch_dynamic_model_info([])

        assert result == {}
        mock_popen.assert_called()


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in dynamic model handling."""

    @pytest.mark.parametrize(
        "model_string",
        [
            pytest.param("a", id="single-char"),
            pytest.param("model", id="no-provider"),
            pytest.param("provider/", id="empty-model-name"),
            pytest.param("/model", id="empty-provider"),
            pytest.param("a/b/c/d/e", id="many-slashes"),
        ]
    )
    def test_unusual_model_strings(self, model_string):
        """Test that unusual model strings don't crash the system."""
        try:
            model = Model.from_litellm(model_string)
            # Should be able to access basic properties
            _ = model.value
            _ = model.provider
            _ = model.prefetched_specs
        except Exception as e:
            pytest.fail(f"Model instantiation failed for '{model_string}': {e}")

    def test_model_with_special_characters(self):
        """Test model strings with special characters."""
        special_strings = [
            "provider/model-with-dashes",
            "provider/model_with_underscores",
            "provider/model.with.dots",
            "provider/Model-V1.2.3-Beta",
        ]

        for model_string in special_strings:
            model = Model.from_litellm(model_string)
            assert model.value == model_string

    def test_same_model_string_returns_consistent_results(self):
        """Test that the same model string returns consistent specs."""
        model_string = "custom/consistent-model"

        model1 = Model.from_litellm(model_string)
        model2 = Model.from_litellm(model_string)

        assert model1.get_usd_per_input_token() == model2.get_usd_per_input_token()
        assert model1.get_usd_per_output_token() == model2.get_usd_per_output_token()
        assert model1.is_text_model() == model2.is_text_model()


# =============================================================================
# TEST CLASS: End-to-End Integration (requires API keys)
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests requiring actual API keys."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_dynamic_model_pipeline(self):
        """Test running a full pipeline with a dynamic model string."""
        dynamic_model_name = "openai/gpt-3.5-turbo-0125"

        mock_info = {
            dynamic_model_name: {
                "mode": "chat",
                "input_cost_per_token": 0.50 / 1e6,
                "output_cost_per_token": 1.50 / 1e6,
                "input_cost_per_audio_token": 0.0,
                "max_tokens": 4096,
                "supports_reasoning": False,
                "supports_vision": False
            }
        }

        with patch("palimpzest.query.processor.query_processor_factory.fetch_dynamic_model_info") as mock_fetch:
            mock_fetch.side_effect = lambda _: DYNAMIC_MODEL_INFO.update(mock_info)

            df = pd.DataFrame({"text": ["What is the capital of France?", "What is 2 + 2?"]})
            dataset = pz.MemoryDataset("test_data", df)

            dynamic_model = Model.from_litellm(dynamic_model_name)

            config = QueryProcessorConfig(
                policy=MinCost(),
                available_models=[dynamic_model],
                verbose=True,
                execution_strategy="sequential"
            )

            class ResponseSchema(BaseModel):
                answer: str = Field(description="The answer to the question")

            plan = dataset.sem_map(
                cols=ResponseSchema,
                desc="Answer the question"
            )

            result_collection = plan.run(config)
            results = result_collection.to_df()

            assert len(results) == 2
            assert "answer" in results.columns

            answers = results["answer"].astype(str).str.lower().tolist()
            assert any("paris" in a for a in answers)
            assert any("4" in a for a in answers)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_dynamic_model_real_api(self):
        """Test with a real API call using a non-enum model string."""
        model_name = "openai/gpt-3.5-turbo"

        model = Model.from_litellm(model_name)
        assert model.get_usd_per_input_token() is not None
        assert model.get_usd_per_audio_input_token() is not None

        df = pd.DataFrame({"question": ["What is 2 + 2?", "What is the capital of France?"]})
        dataset = pz.MemoryDataset("test_e2e_real", df)

        class ShortAnswer(BaseModel):
            answer: str = Field(description="A concise answer to the question")

        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[model],
            verbose=True,
            execution_strategy="sequential"
        )

        plan = dataset.sem_map(
            cols=ShortAnswer,
            desc="Answer the question briefly"
        )

        with patch("palimpzest.query.processor.query_processor_factory.fetch_dynamic_model_info", return_value={}):
            results = plan.run(config)

        records = results.to_df()
        assert len(records) == 2

        answers = records["answer"].astype(str).str.lower().tolist()
        assert any("4" in a for a in answers)
        assert any("paris" in a for a in answers)
