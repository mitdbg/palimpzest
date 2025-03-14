import os
from abc import ABC, abstractmethod
from typing import Any
import requests
from openai import OpenAI
import json

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, 
                 prompt: str, 
                 model: str, 
                 temperature: float = 0.1, 
                 max_tokens: int = 3000, 
                 **kwargs) -> dict[str, Any]:
        """Generate text from the LLM provider."""
        pass

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")

    def generate(self, 
                 prompt: str, 
                 model: str = "gpt-4o-mini", 
                 system_prompt: str | None = None,
                 user_prompt: str | None = None,
                 temperature: float = 0.1,
                 max_tokens: int = 3000, 
                 **kwargs) -> dict[str, Any]:
        """Generate text using OpenAI API."""

        # Handle different model types (completion vs chat)
        if model.startswith("gpt"):
            # Chat completion
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

            if prompt != "":
                messages.append({"role": "user", "content": prompt})

            print("SEND OUT MESSAGES--------------------------------:")
            print(messages)
            response = self.client.chat.completions.create(model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items() if k not in ["messages"]})
            print("RECEIVED RESPONSE--------------------------------:")
            print(response)
            return response
        else:
            # Text completion
            response = self.client.completions.create(model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs)
            return response

class TogetherAIProvider(LLMProvider):
    """Provider for TogetherAI API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TogetherAI API key is required. Set TOGETHER_API_KEY environment variable or pass it directly.")
        self.api_url = "https://api.together.xyz/v1/completions"

    def generate(self, 
                 prompt: str, 
                 model: str, 
                 temperature: float = 0.1, 
                 max_tokens: int = 3000, 
                 **kwargs) -> dict[str, Any]:
        """Generate text using TogetherAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

class LLMClient:
    """Client for interacting with various LLM providers."""

    PROVIDERS = {
        "gpt-4o": OpenAIProvider,
        "gpt-4o-mini": OpenAIProvider,
        "mistralai/Mixtral-8x7B-v0.1": TogetherAIProvider
    }

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        """
        Initialize the LLM client.
        
        Args:
            model: The model to use ("gpt-4o-mini" or "mistralai/Mixtral-8x7B-v0.1")
            api_key: API key for the provider (optional, will use environment variables if not provided)
        """
        if model not in self.PROVIDERS:
            raise ValueError(f"Model {model} not supported. Choose from: {', '.join(self.PROVIDERS.keys())}")

        self.model = model
        self.provider = self.PROVIDERS[model](api_key=api_key)

    def generate(self, 
                 prompt: str, 
                 system_prompt: str | None = None,
                 user_prompt: str | None = None,
                 temperature: float = 0.1, 
                 max_tokens: int = 3000, 
                 **kwargs) -> dict[str, Any]:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: The input text prompt
            model: Model to use (provider-specific)
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response from the LLM provider
        """
        response = self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response

    def get_completion(self, 
                       prompt: str, 
                       system_prompt: str | None = None,
                       user_prompt: str | None = None,
                       temperature: float = 0.1, 
                       max_tokens: int = 3000, 
                       **kwargs) -> str:
        """
        Get just the completion text from the LLM.
        
        Args:
            prompt: The input text prompt
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text as a string
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Extract text based on provider response format
        if isinstance(self.provider, OpenAIProvider):
            # Handle OpenAI's response format
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    # Chat completion
                    return choice.message.content
                elif hasattr(choice, 'text'):
                    # Text completion
                    return choice.text
            raise ValueError(f"Unable to extract content from OpenAI response: {response}")
        elif isinstance(self.provider, TogetherAIProvider) and "choices" in response:
            return response["choices"][0]["text"]
        else:
            raise ValueError(f"Unexpected response format: {response}")
       

# # Example usage
# if __name__ == "__main__":
#     # Using OpenAI
#     openai_client = LLMClient(provider="openai")
#     openai_response = openai_client.get_completion("Tell me a short joke")
#     print(f"OpenAI response: {openai_response}")

#     # Using TogetherAI
#     together_client = LLMClient(provider="togetherai")
#     together_response = together_client.get_completion(
#         "Tell me a short joke", 
#         model="togethercomputer/llama-2-7b"
#     )
#     print(f"TogetherAI response: {together_response}")