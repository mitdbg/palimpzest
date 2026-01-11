import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import palimpzest as pz
from palimpzest.constants import Model, DYNAMIC_MODEL_INFO
from palimpzest.utils.model_helpers import fetch_dynamic_model_info
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.core.data.dataset import Dataset, MemoryDataset
from palimpzest.policy import MinCost

# --- Tests for palimpzest/constants.py and Model Enum ---

def test_model_enum_properties_known_model():
    """
    Verify that a standard, hardcoded model returns the expected property values.
    """
    model = Model.GPT_4o
    
    # Check boolean flags
    assert model.is_text_model() is True
    # GPT-4o is generally not considered an embedding model in this context
    assert model.is_embedding_model() is False
    
    # Check cost retrieval
    cost = model.get_usd_per_input_token()
    assert isinstance(cost, float)
    assert cost > 0

def test_model_enum_dynamic_instantiation():
    """
    Test the _missing_ hook in the Model enum which allows for dynamic model creation.
    """
    model_name = "custom/my-new-model"
    model = Model(model_name)
    
    assert model.value == model_name
    assert model.provider.value == "unknown" 
    
    # Check that it attempts to load specs
    specs = model.prefetched_specs
    assert isinstance(specs, dict)

# --- Tests for palimpzest/utils/model_helpers.py ---

@patch("palimpzest.utils.model_helpers.subprocess.Popen")
@patch("palimpzest.utils.model_helpers.requests.get")
@patch("palimpzest.utils.model_helpers.time.sleep")
def test_fetch_dynamic_model_info_success(mock_sleep, mock_get, mock_popen):
    """
    Test successful fetching of dynamic model info.
    """
    # 1. Setup mock process
    mock_process = MagicMock()
    # communicate() must return a tuple (stdout, stderr)
    mock_process.communicate.return_value = (b"server started", b"")
    # poll() returning None means process is running. 
    # To simulate successful startup we want poll to return None initially 
    mock_process.poll.return_value = None 
    
    mock_popen.return_value = mock_process
    
    # 2. Setup mock requests
    # Sequence: [Health Check OK, Info Endpoint Response]
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
    
    # 3. Execute
    model_input = Model("hosted_vllm/llama-3-70b")
    
    result = fetch_dynamic_model_info([model_input])
    
    # 4. Assertions
    mock_popen.assert_called_once()
    assert "hosted_vllm/llama-3-70b" in result
    assert result["hosted_vllm/llama-3-70b"]["input_cost_per_token"] == 0.0005
    mock_process.terminate.assert_called()

@patch("palimpzest.utils.model_helpers.subprocess.Popen")
def test_fetch_dynamic_model_info_empty_input(mock_popen):
    """
    Test that the function handles empty input gracefully.
    """
    mock_process = MagicMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.poll.return_value = 0 
    mock_popen.return_value = mock_process

    result = fetch_dynamic_model_info([])
    
    assert result == {}
    mock_popen.assert_called()

# --- Tests for palimpzest/query/processor/query_processor_factory.py ---

@patch("palimpzest.query.processor.query_processor_factory.fetch_dynamic_model_info")
@patch("palimpzest.query.processor.query_processor_factory.QueryProcessor")
def test_factory_calls_dynamic_fetch(mock_processor_cls, mock_fetch):
    """
    Verify that creating a processor triggers the dynamic info fetch.
    """
    # Setup
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.schema = MagicMock()
    
    config = QueryProcessorConfig(
        policy=MinCost(),
        available_models=[Model.GPT_4o],
        verbose=True,
        api_base="http://my-vllm-instance:8000"
    )
    
    # Execute with mocked API Key to satisfy validation
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
        with patch.object(QueryProcessorFactory, "_create_optimizer") as mock_opt:
            with patch.object(QueryProcessorFactory, "_create_execution_strategy") as mock_exec:
                 with patch.object(QueryProcessorFactory, "_create_sentinel_execution_strategy") as mock_sent:
                     QueryProcessorFactory.create_processor(mock_dataset, config=config)
    
    # Assert
    mock_fetch.assert_called_once_with(config.available_models)

def test_integration_dynamic_model_usage():
    """
    Unit test for DYNAMIC_MODEL_INFO dictionary updates.
    """
    dynamic_name = "hosted_vllm/special-model"
    
    test_info = {
        "supports_reasoning": True,
        "input_cost_per_token": 0.05,
        "mode": "chat"
    }
    
    with patch.dict(DYNAMIC_MODEL_INFO, {dynamic_name: test_info}):
        model = Model(dynamic_name)
        assert model.is_reasoning_model() is True
        assert model.get_usd_per_input_token() == 0.05

# --- End-to-End Integration Test ---

@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_dynamic_model_end_to_end():
    """
    Test running a full pipeline with a model that is NOT in the standard Model enum.
    We use a specific OpenAI model version (e.g., 'openai/gpt-3.5-turbo-0125') 
    which likely isn't in the hardcoded constants but is valid for the API.
    """
    # 1. Define a dynamic model name (a specific version usually works best as a test case)
    # Using 'gpt-3.5-turbo' is safe; even if it's in constants, the logic for Model(str)
    # handles strings dynamically if they aren't accessed via Model.MEMBER
    dynamic_model_name = "openai/gpt-3.5-turbo-0125"
    
    # 2. Mock the dynamic info fetch so we don't need a local litellm server running
    #    This ensures the optimizer has cost info to proceed.
    mock_info = {
        dynamic_model_name: {
            "mode": "chat",
            "input_cost_per_token": 0.50 / 1e6,
            "output_cost_per_token": 1.50 / 1e6,
            "max_tokens": 4096,
            "supports_reasoning": False,
            "supports_vision": False
        }
    }

    # Patch fetch_dynamic_model_info to populate DYNAMIC_MODEL_INFO without subprocesses
    with patch("palimpzest.query.processor.query_processor_factory.fetch_dynamic_model_info") as mock_fetch:
        mock_fetch.side_effect = lambda models: DYNAMIC_MODEL_INFO.update(mock_info)
        
        # 3. Define the pipeline
        # Create a simple memory dataset
        df = pd.DataFrame({"text": ["What is the capital of France?", "What is 2 + 2?"]})
        dataset = pz.MemoryDataset("test_data", df)
        
        # Create the dynamic model object explicitly
        dynamic_model = Model(dynamic_model_name)
        
        # Configure processor to use ONLY this dynamic model
        config = QueryProcessorConfig(
            policy=MinCost(),
            available_models=[dynamic_model],
            verbose=True,
            execution_strategy="sequential"
        )
        
        # Define output schema
        class ResponseSchema(pz.Schema):
            answer = pz.StringField(desc="The answer to the question")

        # 4. Build the logical plan
        # We explicitly request the dynamic model in the operator to ensure it's used
        plan = dataset.convert(
            output_schema=ResponseSchema,
            desc="Answer the question",
            model=dynamic_model
        )
        
        # 5. Run the pipeline
        # This will trigger the factory, calling our mocked fetch_dynamic_model_info,
        # then the optimizer (using costs from mock_info), and finally execution
        # which sends requests to OpenAI using the real API key.
        result_collection = plan.run(config)
        
        # 6. Verify results
        results = result_collection.to_df()
        
        assert len(results) == 2
        assert "answer" in results.columns
        
        # Basic content validation to ensure the LLM actually ran
        answers = results["answer"].str.lower().tolist()
        assert any("paris" in a for a in answers)
        assert any("4" in a for a in answers)