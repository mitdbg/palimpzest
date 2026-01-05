import os
import pytest
from unittest.mock import patch, MagicMock

# Corrected imports based on the refactor
from palimpzest.utils.model_helpers import (
    get_model_provider,
    predict_model_specs,
)
from palimpzest.utils.model_info import (
    Model,
    get_optimal_models,
)
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.policy import MaxQuality, MinCost, Policy
from palimpzest.core.models import PlanCost
from palimpzest.constants import CuratedModel

# --- Tests for palimpzest.utils.model_helpers ---

class TestModelHelpers:
    
    @pytest.mark.parametrize("model_name, expected_provider", [
        # Explicit providers
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-3-5-sonnet", "anthropic"),
        ("vertex_ai/gemini-pro", "vertex_ai"),
        ("together_ai/meta-llama/Llama-3-70b", "together_ai"),
        # Known families
        ("gpt-4", "openai"),
        ("o1-preview", "openai"),
        ("claude-3-opus", "anthropic"),
        ("gemini-1.5-pro", "gemini"),
        ("llama-3-70b", "meta"),
        ("mistral-large", "mistral"),
        ("command-r", "cohere"),
        # Edge cases
        ("unknown-provider/my-model", "unknown-provider"),
        ("custom-model-with-openai-in-name", "openai"), 
        (None, "unknown"),
        ("", "unknown"),
    ])
    def test_get_model_provider(self, model_name, expected_provider):
        """Test that the provider is correctly extracted from the model name."""
        assert get_model_provider(model_name) == expected_provider

    def test_predict_model_specs_reasoning(self):
        """Test heuristics for reasoning/thinking models."""
        specs = predict_model_specs("deepseek/deepseek-r1")
        assert specs["tier"] == "Reasoning (Heavy)"
        assert specs["mmlu_pro_score"] >= 80.0
        assert specs["usd_per_1m_input"] >= 10.0
        
        # Test efficient reasoning
        specs_mini = predict_model_specs("openai/o1-mini")
        assert specs_mini["tier"] == "Reasoning (Efficient)"
        assert specs_mini["usd_per_1m_input"] < specs["usd_per_1m_input"]

    def test_predict_model_specs_future_flagship(self):
        """Test heuristics for future flagship models."""
        specs = predict_model_specs("openai/gpt-5")
        assert specs["tier"] == "Next-Gen Flagship"
        assert specs["mmlu_pro_score"] > 90.0

    def test_predict_model_specs_economy(self):
        """Test heuristics for economy models."""
        specs = predict_model_specs("meta/llama-3-8b")
        assert specs["tier"] == "Economy"
        assert specs["usd_per_1m_input"] < 1.0

    def test_predict_model_specs_multimodal(self):
        """Test heuristics for multimodal audio detection."""
        specs = predict_model_specs("openai/gpt-4o-audio-preview")
        assert specs["usd_per_1m_audio_input"] is not None
        assert specs["usd_per_1m_audio_input"] > specs["usd_per_1m_input"]


# --- Tests for palimpzest.utils.model_info ---

class TestModelInfo:
    
    def test_model_class_init(self):
        """Test Model initialization and string inheritance."""
        m = Model("openai/gpt-4o")
        assert m == "openai/gpt-4o"
        assert isinstance(m, str)
        # Check that prediction was run during init
        assert hasattr(m, "prediction")
        assert isinstance(m.prediction, dict)

    def test_model_attributes_fallback(self):
        """Test Model methods fall back to prediction/heuristics when no dynamic info exists."""
        m = Model("openai/gpt-4o")
        # Should be identified as OpenAI based on name
        assert m.is_openai_model() is True
        assert m.is_anthropic_model() is False
        
        # FIX: Use the exact string from CuratedModel so the lookup succeeds
        m_vision = Model(CuratedModel.GPT_4o.value) 
        assert m_vision.is_vision_model() is True

    @patch("palimpzest.utils.model_info.DYNAMIC_MODEL_INFO")
    def test_model_dynamic_info_override(self, mock_dynamic_info):
        """Test that dynamic info overrides static and predicted values."""
        model_id = "test/dynamic-model"
        
        # Mock the dictionary get access
        mock_dynamic_info.get.return_value = {
            "input_cost_per_token": 1.23,
            "supports_vision": True,
            "supports_audio_input": False
        }
        
        m = Model(model_id)
        
        # Verify overridden values
        assert m.get_usd_per_input_token() == 1.23
        assert m.is_vision_model() is True
        assert m.is_audio_model() is False

    @patch("palimpzest.utils.model_info.get_available_model_from_env")
    @patch("palimpzest.utils.model_info.MODEL_CARDS")
    def test_get_optimal_models_max_quality(self, mock_model_cards, mock_get_available):
        """Test optimization logic for MaxQuality."""
        mock_get_available.return_value = ["model_a", "model_b"]
        
        # Mock cards: A is high quality/high cost, B is low quality/low cost
        mock_model_cards.get.side_effect = lambda mid: {
            "model_a": {"overall": 90.0, "usd_per_output_token": 10.0, "seconds_per_output_token": 0.1},
            "model_b": {"overall": 50.0, "usd_per_output_token": 5.0, "seconds_per_output_token": 0.05},
        }.get(mid)

        policy = MaxQuality()
        top_models = get_optimal_models(policy)

        assert len(top_models) > 0
        assert top_models[0] == "model_a"

    @patch("palimpzest.utils.model_info.get_available_model_from_env")
    @patch("palimpzest.utils.model_info.MODEL_CARDS")
    def test_get_optimal_models_min_cost(self, mock_model_cards, mock_get_available):
        """Test optimization logic for MinCost."""
        mock_get_available.return_value = ["model_a", "model_b"]
        
        mock_model_cards.get.side_effect = lambda mid: {
            "model_a": {"overall": 90.0, "usd_per_output_token": 100.0, "seconds_per_output_token": 1.0},
            "model_b": {"overall": 80.0, "usd_per_output_token": 1.0, "seconds_per_output_token": 1.0},
        }.get(mid)

        policy = MinCost()
        top_models = get_optimal_models(policy)

        assert top_models[0] == "model_b"

    @patch("palimpzest.utils.model_info.get_available_model_from_env")
    @patch("palimpzest.utils.model_info.MODEL_CARDS")
    def test_get_optimal_models_constraints(self, mock_model_cards, mock_get_available):
        """Test that models violating constraints are filtered out."""
        mock_get_available.return_value = ["good_model", "bad_model"]
        
        mock_model_cards.get.side_effect = lambda mid: {
            "good_model": {"overall": 90.0},
            "bad_model": {"overall": 10.0}, # Very low quality
        }.get(mid)

        # Policy with strict quality constraint
        class HighQualityPolicy(Policy):
            def constraint(self, plan_cost: PlanCost) -> bool:
                return plan_cost.quality >= 0.8
            def get_dict(self):
                return {"quality": 1.0, "cost": 0.0, "time": 0.0}

        policy = HighQualityPolicy()
        top_models = get_optimal_models(policy)

        assert "good_model" in top_models
        assert "bad_model" not in top_models


# --- Tests for palimpzest.query.processor.config ---

class TestQueryProcessorConfig:

    @patch("palimpzest.query.processor.config.fetch_dynamic_model_info")
    @patch("palimpzest.query.processor.config.get_optimal_models")
    def test_config_automatic_model_selection(self, mock_get_optimal, mock_fetch_dynamic):
        """
        Test that if available_models is None, get_optimal_models is called.
        """
        mock_get_optimal.return_value = [Model("best-model")]
        
        config = QueryProcessorConfig(available_models=None, policy=MaxQuality())
        
        assert len(config.available_models) == 1
        assert config.available_models[0] == "best-model"
        
        mock_get_optimal.assert_called_once()
        mock_fetch_dynamic.assert_not_called()

    @patch("palimpzest.query.processor.config.fetch_dynamic_model_info")
    @patch("palimpzest.query.processor.config.get_optimal_models")
    def test_config_explicit_model_selection(self, mock_get_optimal, mock_fetch_dynamic):
        """
        Test that if available_models is a list, fetch_dynamic_model_info is called.
        """
        models = ["openai/gpt-4o", "meta/llama-3"]
        config = QueryProcessorConfig(available_models=models)
        
        assert len(config.available_models) == 2
        assert isinstance(config.available_models[0], Model)
        assert config.available_models[0] == "openai/gpt-4o"
        
        mock_fetch_dynamic.assert_called_once_with(models)
        mock_get_optimal.assert_not_called()

    def test_config_invalid_model_type(self):
        """Test validation error for invalid model types."""
        with pytest.raises(TypeError):
            QueryProcessorConfig(available_models=[123, 456])

    @patch("palimpzest.query.processor.config.fetch_dynamic_model_info")
    def test_config_policy_defaults(self, mock_fetch_dynamic):
        """Test that policy defaults to MaxQuality if not provided."""
        config = QueryProcessorConfig(available_models=["test/model"])
        assert isinstance(config.policy, MaxQuality)
        mock_fetch_dynamic.assert_called_once()


