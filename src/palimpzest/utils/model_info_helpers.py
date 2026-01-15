import requests


class ModelMetricsManager:
    """
    Manages fetching and caching of model metrics from an external source.
    """
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/mitdbg/palimpzest/GeneralizeLiteLLM-265/src/palimpzest/utils/pz_models_information.json" # TODO: replace with S3 json link (curated_model_info.json)
        self._metrics_cache = None  # Initialize as None (empty)

    def _load_data(self):
        if self._metrics_cache is None:
            print(f"Fetching data from URL: {self.data_url}")
            try:
                self._metrics_cache = requests.get(self.data_url).json()
            except Exception as e:
                print(f"Error fetching data: {e}")
                self._metrics_cache = {}

    def get_model_metrics(self, model_name):
        self._load_data()
        return self._metrics_cache.get(model_name, {})

    def refresh_data(self):
        self._metrics_cache = None
        self._load_data()