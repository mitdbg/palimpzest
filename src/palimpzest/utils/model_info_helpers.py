import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

PZ_MODEL_DATA_URL = "https://palimpzest-research.s3.us-east-1.amazonaws.com/pz_models_information.json"

class ModelMetricsManager:
    """
    Manages fetching and caching of model metrics from an external source.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.data_url = PZ_MODEL_DATA_URL
        self._metrics_cache = None
        self._initialized = True

    def _load_data(self):
        if self._metrics_cache is None:
            logger.info(f"Fetching data from URL: {self.data_url}")
            try:
                self._metrics_cache = requests.get(self.data_url).json()
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                self._metrics_cache = {}

    def get_model_metrics(self, model_name) -> dict[str, Any]:
        self._load_data()
        return self._metrics_cache.get(model_name, {})

    def refresh_data(self) -> None:
        self._metrics_cache = None
        self._load_data()
