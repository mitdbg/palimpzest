"""
Test suite for Model class and model helper functions in Palimpzest.

This module tests:
- Model instantiation with curated model IDs
- Model properties and methods
- Cost and performance metric retrieval
- Model registry and get_all_models()
- Model helper functions (get_models, get_optimal_models, resolve_reasoning_settings)
- Integration with Generator and QueryProcessor
- End-to-end pipeline execution
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

import palimpzest as pz
from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord
from palimpzest.policy import MaxQuality, MinCost, MinCostAtFixedQuality, MinTime
from palimpzest.query.generators.generators import Generator
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory
from palimpzest.utils.model_helpers import (
    get_models,
    get_optimal_models,
)
from palimpzest.utils.model_info_helpers import (
    derive_model_flags,
    fuzzy_match_score,
    predict_local_model_metrics,
    MMLU_PRO_SCORES,
    LATENCY_TPS_DATA,
    DEFAULT_QUALITY_SCORE,
    DEFAULT_SECONDS_PER_OUTPUT_TOKEN,
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

# =============================================================================
# TEST CLASS: Model Class Instantiation
# =============================================================================

class TestModelInstantiation:
    """Tests for Model class instantiation."""

    def test_known_model_instantiation(self):
        """Test that a known model can be instantiated."""
        model = Model.GPT_4o
        assert model is not None
        assert model.value == "openai/gpt-4o-2024-08-06"

    def test_model_instantiation_with_string(self):
        """Test Model instantiation with a valid model string."""
        # This should work if the model exists in the curated JSON
        model = Model("openai/gpt-4o-2024-08-06")
        assert model.value == "openai/gpt-4o-2024-08-06"
        assert model.provider == "openai"

    def test_unknown_model_raises_error(self):
        """Test that unknown model IDs raise ValueError."""
        with pytest.raises(ValueError, match="does not contain information"):
            Model("unknown-provider/nonexistent-model-xyz")

    def test_model_properties_from_specs(self):
        """Test that model properties are correctly loaded from specs."""
        model = Model.GPT_4o

        assert model.is_text_model() is True
        assert model.is_embedding_model() is False
        assert isinstance(model.get_usd_per_input_token(), float)
        assert model.get_usd_per_input_token() > 0

    def test_model_provider_property(self):
        """Test that the provider property returns the correct string."""
        model = Model.GPT_4o
        assert model.provider == "openai"

        model_anthropic = Model.CLAUDE_3_7_SONNET
        assert model_anthropic.provider == "anthropic"

    def test_model_api_base_parameter(self):
        """Test that api_base parameter creates a local/vLLM model."""
        model = Model("hosted_vllm/qwen/Qwen1.5-0.5B-Chat", api_base="http://localhost:8000/v1")
        assert model.value == "hosted_vllm/qwen/Qwen1.5-0.5B-Chat"
        assert model.api_base == "http://localhost:8000/v1"
        assert model.is_vllm_model() is True


# =============================================================================
# TEST CLASS: Model Registry
# =============================================================================

class TestModelRegistry:
    """Tests for Model registry functionality."""

    def test_models_registered_on_creation(self):
        """Test that models are registered in _registry on creation."""
        # The predefined models should be in the registry
        all_models = Model.get_all_models()
        assert len(all_models) > 0

        # Check that GPT_4o is in the registry
        model_values = [m.value for m in all_models]
        assert "openai/gpt-4o-2024-08-06" in model_values

    def test_get_all_models_returns_list(self):
        """Test that get_all_models returns a list of Model instances."""
        all_models = Model.get_all_models()
        assert isinstance(all_models, list)
        assert all(isinstance(m, Model) for m in all_models)

    def test_registry_contains_expected_models(self):
        """Test that the registry contains expected predefined models."""
        all_models = Model.get_all_models()
        model_values = [m.value for m in all_models]

        # Check for some expected models
        expected_models = [
            "openai/gpt-4o-2024-08-06",
            "anthropic/claude-3-7-sonnet-20250219",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ]
        for expected in expected_models:
            assert expected in model_values, f"Expected {expected} in registry"

# =============================================================================
# TEST CLASS: Model Equality and Hashing
# =============================================================================

class TestModelEqualityAndHashing:
    """Tests for Model equality and hashing."""

    def test_model_equality_same_instance(self):
        """Test that the same model instance is equal to itself."""
        model = Model.GPT_4o
        assert model == model

    def test_model_equality_same_value(self):
        """Test that models with the same value are equal."""
        model1 = Model("openai/gpt-4o-2024-08-06")
        model2 = Model("openai/gpt-4o-2024-08-06")
        assert model1 == model2

    def test_model_equality_with_string(self):
        """Test that a model equals its string value."""
        model = Model.GPT_4o
        assert model == "openai/gpt-4o-2024-08-06"

    def test_model_inequality(self):
        """Test that different models are not equal."""
        assert Model.GPT_4o != Model.CLAUDE_3_7_SONNET

    def test_model_hash_consistency(self):
        """Test that model hash is consistent."""
        model1 = Model("openai/gpt-4o-2024-08-06")
        model2 = Model("openai/gpt-4o-2024-08-06")
        assert hash(model1) == hash(model2)

    def test_model_usable_in_set(self):
        """Test that models can be used in sets."""
        model_set = {Model.GPT_4o, Model.GPT_4o, Model.CLAUDE_3_7_SONNET}
        assert len(model_set) == 2

    def test_model_usable_as_dict_key(self):
        """Test that models can be used as dictionary keys."""
        model_dict = {Model.GPT_4o: "gpt4", Model.CLAUDE_3_7_SONNET: "claude"}
        assert model_dict[Model.GPT_4o] == "gpt4"

    def test_model_str_repr(self):
        """Test string representation of Model."""
        model = Model.GPT_4o
        assert str(model) == "openai/gpt-4o-2024-08-06"
        assert repr(model) == "openai/gpt-4o-2024-08-06"

    def test_model_lt_comparison(self):
        """Test less-than comparison for sorting."""
        models = [Model.GPT_4o, Model.CLAUDE_3_7_SONNET, Model.LLAMA3_1_8B]
        sorted_models = sorted(models)
        # Should be sortable without error
        assert len(sorted_models) == 3


# =============================================================================
# TEST CLASS: Model Helper Functions
# =============================================================================

class TestModelHelperFunctions:
    """Tests for model helper functions."""

    def test_get_models_with_openai_key(self):
        """Test get_models returns OpenAI models when key is set."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            models = get_models()
            openai_models = [m for m in models if m.provider == "openai"]
            assert len(openai_models) > 0

    def test_get_models_excludes_embedding_by_default(self):
        """Test that embedding models are excluded by default."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            models = get_models(include_embedding=False)
            embedding_models = [m for m in models if m.is_embedding_model()]
            assert len(embedding_models) == 0

    def test_get_models_includes_embedding_when_requested(self):
        """Test that embedding models are included when requested."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            models = get_models(include_embedding=True)
            embedding_models = [m for m in models if m.is_embedding_model()]
            assert len(embedding_models) > 0

    def test_get_models_empty_without_keys(self):
        """Test that get_models returns empty list without API keys."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "TOGETHER_API_KEY": "",
            "GEMINI_API_KEY": "",
        }, clear=True):
            models = get_models()
            assert len(models) == 0

    def test_get_optimal_models_returns_top_models(self):
        """Test that get_optimal_models returns top models based on policy."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            models = get_optimal_models(policy=MinCost())
            assert len(models) <= 5  # Should return at most 5

    def test_get_optimal_models_respects_policy(self):
        """Test that optimal models selection respects the policy."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "test-key",
        }, clear=False):
            cost_models = get_optimal_models(policy=MinCost())
            quality_models = get_optimal_models(policy=MaxQuality())

            # Both should return models
            assert len(cost_models) > 0
            assert len(quality_models) > 0

    def test_get_optimal_models_never_returns_empty_with_available_models(self):
        """Test that get_optimal_models never returns empty when models are available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            # Use a very high quality constraint that no model can meet (0.95 = 95% MMLU-Pro)
            policy = MinCostAtFixedQuality(min_quality=0.95)
            models = get_optimal_models(policy=policy)

            # Should still return at least one model (the best by primary metric)
            assert len(models) >= 1

    def test_get_optimal_models_fallback_returns_best_by_primary_metric(self):
        """Test that fallback returns best model according to primary metric."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            # MinCostAtFixedQuality has primary_metric="cost"
            # With impossible constraint, should return cheapest model
            policy_cost = MinCostAtFixedQuality(min_quality=0.99)
            cost_models = get_optimal_models(policy=policy_cost)
            assert len(cost_models) >= 1

            # MaxQuality has primary_metric="quality"
            # Even with no constraint issues, verify it returns models
            policy_quality = MaxQuality()
            quality_models = get_optimal_models(policy=policy_quality)
            assert len(quality_models) >= 1

    def test_get_optimal_models_fallback_with_time_policy(self):
        """Test that fallback works with time-based policy."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            # MinTime has primary_metric="time"
            policy = MinTime()
            models = get_optimal_models(policy=policy)

            # Should return models (fastest ones)
            assert len(models) >= 1


# =============================================================================
# TEST CLASS: Generator Integration
# =============================================================================

class TestGeneratorIntegration:
    """Tests for Generator integration with Model class."""

    @patch("palimpzest.query.generators.generators.litellm.completion")
    def test_generator_uses_model_value(
        self, mock_completion, sample_record, output_schema, mock_litellm_response
    ):
        """Test that Generator uses model.value for litellm calls."""
        mock_completion.return_value = mock_litellm_response

        model = Model.GPT_4o
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
        assert kwargs["model"] == "openai/gpt-4o-2024-08-06"

    @patch("palimpzest.query.generators.generators.litellm.completion")
    def test_generator_with_different_providers(
        self, mock_completion, sample_record, output_schema, mock_litellm_response
    ):
        """Test Generator works with models from different providers."""
        mock_completion.return_value = mock_litellm_response

        for model in [Model.GPT_4o, Model.CLAUDE_3_7_SONNET, Model.LLAMA3_3_70B]:
            generator = Generator(
                model=model,
                prompt_strategy=PromptStrategy.MAP,
                reasoning_effort="default"
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
            assert kwargs["model"] == model.value


# =============================================================================
# TEST CLASS: QueryProcessor Integration
# =============================================================================

class TestQueryProcessorIntegration:
    """Tests for QueryProcessor integration."""

    @patch("palimpzest.query.processor.query_processor_factory.QueryProcessor")
    def test_factory_accepts_model_list(self, mock_processor_cls):
        """Test that QueryProcessorFactory accepts available_models."""
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.schema = MagicMock()
        mock_dataset.get_limit.return_value = None

        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[Model.GPT_4o, Model.CLAUDE_3_7_SONNET],
            verbose=True,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "ANTHROPIC_API_KEY": "fake-key"}), \
             patch.object(QueryProcessorFactory, "_create_optimizer"), \
             patch.object(QueryProcessorFactory, "_create_execution_strategy"), \
             patch.object(QueryProcessorFactory, "_create_sentinel_execution_strategy"):
            QueryProcessorFactory.create_processor(mock_dataset, config=config)

        # Verify processor was created
        mock_processor_cls.assert_called_once()

    def test_factory_auto_selects_models_when_none_provided(self):
        """Test that factory calls get_optimal_models when available_models is empty."""
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.schema = MagicMock()
        mock_dataset.get_limit.return_value = None

        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[],  # Empty list
            verbose=True,
        )

        # Mock get_optimal_models to return some models and verify it's called
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}), \
             patch("palimpzest.query.processor.query_processor_factory.get_optimal_models",
                   return_value=[Model.GPT_4o, Model.GPT_4o_MINI]) as mock_get_optimal, \
             patch("palimpzest.query.processor.query_processor_factory.QueryProcessor"), \
             patch.object(QueryProcessorFactory, "_create_optimizer"), \
             patch.object(QueryProcessorFactory, "_create_execution_strategy"), \
             patch.object(QueryProcessorFactory, "_create_sentinel_execution_strategy"):
            QueryProcessorFactory.create_processor(mock_dataset, config=config)
            # Verify get_optimal_models was called with correct policy
            mock_get_optimal.assert_called_once()
            call_kwargs = mock_get_optimal.call_args
            assert call_kwargs[1]["policy"] == config.policy

# =============================================================================
# TEST CLASS: End-to-End Integration
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests for the palimpzest pipeline."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_simple_sem_map_pipeline(self):
        """Test a simple semantic map pipeline end-to-end."""
        # Create a simple dataset
        df = pd.DataFrame({
            "question": ["What is 2 + 2?", "What is the capital of France?"]
        })
        dataset = pz.MemoryDataset("test_e2e", df)

        # Define output schema
        class Answer(BaseModel):
            response: str = Field(description="The answer to the question")

        # Create pipeline
        plan = dataset.sem_map(
            cols=Answer,
            desc="Answer the question concisely"
        )

        # Configure and run
        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[Model.GPT_4o_MINI],
            execution_strategy="sequential",
            progress=False,
            verbose=False,
        )

        # Execute the pipeline
        results = plan.run(config)
        result_df = results.to_df()

        # Verify results
        assert len(result_df) == 2
        assert "response" in result_df.columns

        # Check that we got meaningful answers
        answers = result_df["response"].astype(str).str.lower().tolist()
        assert any("4" in a for a in answers), "Expected answer containing '4'"
        assert any("paris" in a for a in answers), "Expected answer containing 'paris'"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_pipeline_with_filter(self):
        """Test a pipeline with semantic filter end-to-end."""
        # Create dataset with mixed content
        df = pd.DataFrame({
            "text": [
                "The sky is blue.",
                "Python is a programming language.",
                "Water boils at 100 degrees Celsius.",
                "JavaScript runs in browsers.",
            ]
        })
        dataset = pz.MemoryDataset("test_filter", df)

        # Filter for programming-related content
        filtered = dataset.sem_filter("text is about programming")

        # Configure and run
        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[Model.GPT_4o_MINI],
            execution_strategy="sequential",
            progress=False,
            verbose=False,
        )

        results = filtered.run(config)
        result_df = results.to_df()

        # Should have filtered to programming-related rows
        assert len(result_df) >= 1
        assert len(result_df) <= 2  # Should be Python and/or JavaScript rows

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_pipeline_with_auto_model_selection(self):
        """Test that pipeline works with automatic model selection."""
        df = pd.DataFrame({"input": ["Hello, world!"]})
        dataset = pz.MemoryDataset("test_auto", df)

        class Output(BaseModel):
            greeting: str = Field(description="A friendly greeting response")

        plan = dataset.sem_map(cols=Output, desc="Respond with a greeting")

        # Don't specify available_models - let the system auto-select
        config = QueryProcessorConfig(
            policy=MinCost(),
            execution_strategy="sequential",
            progress=False,
            verbose=False,
        )

        results = plan.run(config)
        result_df = results.to_df()

        assert len(result_df) == 1
        assert "greeting" in result_df.columns


# =============================================================================
# TEST CLASS: vLLM / Local Model Support
# =============================================================================

class TestVLLMModelSupport:
    """Tests for local/vLLM model creation, metrics, flags, and validation."""

    # --- Model Creation ---

    def test_vllm_model_creation_with_api_base(self):
        """Test that a vLLM model can be created with api_base."""
        model = Model("hosted_vllm/qwen/Qwen1.5-0.5B-Chat", api_base="http://localhost:8000/v1")
        assert model.value == "hosted_vllm/qwen/Qwen1.5-0.5B-Chat"
        assert model.api_base == "http://localhost:8000/v1"

    def test_vllm_model_stores_extra_kwargs(self):
        """Test that extra kwargs are stored as vllm_kwargs."""
        model = Model("openai/Qwen/Qwen2.5-1.5B-Instruct", api_base="http://localhost:8000/v1", max_tokens=128)
        assert model.vllm_kwargs == {"max_tokens": 128}

    def test_vllm_model_without_api_base_raises(self):
        """Test that a model without api_base and not in curated JSON raises ValueError."""
        with pytest.raises(ValueError, match="does not contain information"):
            Model("hosted_vllm/totally-fake/NonexistentModel-v999")

    # --- Cost is Zero for Local Models ---

    def test_vllm_model_cost_is_zero(self):
        """Test that all cost metrics are 0 for local/vLLM models."""
        model = Model("hosted_vllm/qwen/Qwen1.5-0.5B-Chat", api_base="http://localhost:8000/v1")
        assert model.get_usd_per_input_token() == 0.0
        assert model.get_usd_per_output_token() == 0.0
        assert model.get_usd_per_audio_input_token() == 0.0
        assert model.get_usd_per_cache_read_token() == 0.0
        assert model.get_usd_per_cache_creation_token() == 0.0
        assert model.get_usd_per_audio_cache_creation_token() == 0.0

    # --- Quality and Latency Predictions ---

    def test_predict_local_model_metrics_known_model(self):
        """Test predict_local_model_metrics for a model with known scores."""
        metrics = predict_local_model_metrics("meta-llama/Llama-3.1-8B-Instruct")
        assert metrics["MMLU_Pro_score"] == 44.25
        assert metrics["seconds_per_output_token"] == round(1.0 / 200.0, 6)

    def test_predict_local_model_metrics_unknown_model(self):
        """Test predict_local_model_metrics falls back to defaults for unknown models."""
        metrics = predict_local_model_metrics("some-unknown/model-xyz")
        assert metrics["MMLU_Pro_score"] == DEFAULT_QUALITY_SCORE
        assert metrics["seconds_per_output_token"] == DEFAULT_SECONDS_PER_OUTPUT_TOKEN

    def test_vllm_model_has_quality_score(self):
        """Test that a vLLM model gets a quality score via fuzzy matching."""
        model = Model("openai/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:8000/v1")
        score = model.get_overall_score()
        assert score == 44.25

    def test_vllm_model_has_latency(self):
        """Test that a vLLM model gets latency via fuzzy matching."""
        model = Model("openai/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:8000/v1")
        latency = model.get_seconds_per_output_token()
        assert latency == round(1.0 / 200.0, 6)

    def test_vllm_model_unknown_gets_defaults(self):
        """Test that an unrecognized vLLM model gets default quality and latency."""
        model = Model("openai/some-custom/MyCustomModel-v1", api_base="http://localhost:8000/v1")
        assert model.get_overall_score() == DEFAULT_QUALITY_SCORE
        assert model.get_seconds_per_output_token() == DEFAULT_SECONDS_PER_OUTPUT_TOKEN

    # --- Fuzzy Matching ---

    def test_fuzzy_match_exact_substring(self):
        """Test that fuzzy_match_score finds exact substring matches."""
        score = fuzzy_match_score("meta-llama/Llama-3.3-70B-Instruct-Turbo", MMLU_PRO_SCORES)
        assert score == 69.9

    def test_fuzzy_match_normalized(self):
        """Test that fuzzy_match_score handles normalized matching."""
        score = fuzzy_match_score("deepseek-ai/DeepSeek-V3", MMLU_PRO_SCORES)
        assert score == 73.8

    def test_fuzzy_match_no_match_returns_none(self):
        """Test that fuzzy_match_score returns None for unrecognized models."""
        score = fuzzy_match_score("totally-unknown-model", MMLU_PRO_SCORES)
        assert score is None

    # --- derive_model_flags ---

    def test_derive_model_flags_llama(self):
        """Test that derive_model_flags correctly detects Llama models."""
        flags = derive_model_flags("openai/meta-llama/Llama-3.1-8B-Instruct")
        assert flags.get("is_llama_model") is True

    def test_derive_model_flags_non_llama(self):
        """Test that derive_model_flags does not set is_llama_model for non-Llama."""
        flags = derive_model_flags("openai/Qwen/Qwen2.5-1.5B-Instruct")
        assert "is_llama_model" not in flags

    def test_derive_model_flags_clip(self):
        """Test that derive_model_flags correctly detects CLIP models."""
        flags = derive_model_flags("clip-ViT-B-32")
        assert flags.get("is_clip_model") is True

    def test_derive_model_flags_gpt5(self):
        """Test that derive_model_flags correctly detects GPT-5 models."""
        flags = derive_model_flags("openai/gpt-5-2025-08-07")
        assert flags.get("is_gpt_5_model") is True

    def test_derive_model_flags_o_model(self):
        """Test that derive_model_flags correctly detects O-series models."""
        flags = derive_model_flags("openai/o4-mini-2025-04-16")
        assert flags.get("is_o_model") is True

    # --- is_vllm_model and is_llama_model for local models ---

    def test_vllm_model_is_vllm(self):
        """Test that is_vllm_model returns True for api_base models."""
        model = Model("openai/Qwen/Qwen2.5-1.5B-Instruct", api_base="http://localhost:8000/v1")
        assert model.is_vllm_model() is True

    def test_vllm_llama_model_is_llama(self):
        """Test that a local Llama model correctly reports is_llama_model."""
        model = Model("openai/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:8000/v1")
        assert model.is_llama_model() is True

    def test_vllm_non_llama_is_not_llama(self):
        """Test that a non-Llama local model does not report is_llama_model."""
        model = Model("openai/Qwen/Qwen2.5-1.5B-Instruct", api_base="http://localhost:8000/v1")
        assert model.is_llama_model() is False

    # --- Default capabilities for local models ---

    def test_vllm_model_defaults(self):
        """Test default capabilities for a vLLM model."""
        model = Model("openai/Qwen/Qwen2.5-1.5B-Instruct", api_base="http://localhost:8000/v1")
        assert model.is_text_model() is True
        assert model.is_embedding_model() is False

    # --- QueryProcessor vLLM Validation ---

    def test_factory_rejects_multiple_vllm_models(self):
        """Test that QueryProcessorFactory rejects configs with multiple vLLM models."""
        mock_dataset = MagicMock(spec=Dataset)
        mock_dataset.schema = MagicMock()
        mock_dataset.get_limit.return_value = None

        model1 = Model("openai/model-a", api_base="http://localhost:8000/v1")
        model2 = Model("openai/model-b", api_base="http://localhost:8001/v1")
        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[model1, model2],
        )

        with pytest.raises(ValueError, match="Only one vLLM model"):
            QueryProcessorFactory.create_processor(mock_dataset, config=config)

    # --- Generator vLLM kwargs ---

    @patch("palimpzest.query.generators.generators.litellm.completion")
    def test_generator_passes_vllm_kwargs(self, mock_completion, sample_record, output_schema, mock_litellm_response):
        """Test that Generator passes api_base and vllm_kwargs to litellm."""
        mock_completion.return_value = mock_litellm_response

        model = Model("openai/Qwen/Qwen2.5-1.5B-Instruct", api_base="http://localhost:8000/v1", max_tokens=128)
        generator = Generator(
            model=model,
            prompt_strategy=PromptStrategy.MAP,
            reasoning_effort="default",
        )

        fields = {k: FieldInfo.from_annotation(v) for k, v in output_schema.model_fields.items()}
        generator(candidate=sample_record, fields=fields, prompt="Test", parse_answer=lambda x: x, output_schema=output_schema)

        _, kwargs = mock_completion.call_args
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        assert kwargs["max_tokens"] == 128
        assert "api_key" in kwargs
