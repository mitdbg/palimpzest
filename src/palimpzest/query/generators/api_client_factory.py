from threading import Lock

from openai import OpenAI
from together import Together

from palimpzest.constants import APIClient


class APIClientFactory:
    _instances = {}
    _lock = Lock()

    @classmethod
    def get_client(cls, api_client: APIClient, api_key: str):
        """Get a singleton instance of the requested API client."""
        if api_client not in cls._instances:
            with cls._lock:  # Ensure thread safety
                if api_client not in cls._instances:  # Double-check inside the lock
                    cls._instances[api_client] = cls._create_client(api_client, api_key)
        return cls._instances[api_client]

    @staticmethod
    def _create_client(api_client: APIClient, api_key: str):
        """Create a new client instance based on the api_client name."""
        match api_client:
            case APIClient.OPENAI:
                return OpenAI(api_key=api_key)
            case APIClient.TOGETHER:
                return Together(api_key=api_key)
            case _:
                raise ValueError(f"Unknown api_client: {api_client}")
