#!/usr/bin/env python3
"""
Minimal demo for running a vLLM model with Palimpzest.

Prerequisites:
  1. Start a vLLM server serving a small model, e.g.:
     vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
  2. Run this script:
     python demos/vllm-demo.py \
       --api-base http://localhost:8000/v1 \
       --model-id openai/Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import os

from pydantic import BaseModel, Field

import palimpzest as pz


class SentimentResult(BaseModel):
    sentiment: str = Field(description="The sentiment of the text: positive, negative, or neutral")


def main():
    parser = argparse.ArgumentParser(description="Run a minimal vLLM demo")
    parser.add_argument("--api-base", type=str, required=True, help="vLLM server base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID for litellm (e.g. openai/Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens for completion")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # Create the vLLM model with api_base and kwargs on the Model instance
    vllm_model = pz.Model(args.model_id, api_base=args.api_base, max_tokens=args.max_tokens)

    # Load the enron-tiny dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "enron-tiny")
    dataset = pz.TextFileDataset(id="test-sentiment", path=data_path)
    dataset = dataset.sem_map(SentimentResult, desc="Classify the sentiment of the text")

    # Configure with vLLM model
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        available_models=[vllm_model],
        execution_strategy="sequential",
        optimizer_strategy="pareto",
        verbose=args.verbose,
    )

    output = dataset.run(config)
    for record in output:
        print(record)


if __name__ == "__main__":
    main()
