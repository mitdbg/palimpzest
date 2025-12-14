from typing import Any, Dict, List
import numpy as np
from palimpzest.graphrag.retrieval import OpenAIEmbeddingModel, SentenceTransformerEmbeddingModel

class Embedder:
    """
    A wrapper around embedding models to be used with Palimpzest's map operator.
    """
    def __init__(self, model_name: str = "openai", config: Any = None):
        self.model_name = model_name
        if model_name == "openai":
            self.model = OpenAIEmbeddingModel(config=config)
        elif model_name == "sentence-transformers":
            self.model = SentenceTransformerEmbeddingModel(config=config)
        else:
            raise ValueError(f"Unknown embedding model: {model_name}")

    def embed(self, record: Dict[str, Any], input_col: str = "text", output_col: str = "embedding") -> Dict[str, Any]:
        """
        Embeds the text in `input_col` and returns a dictionary with the embedding.
        """
        text = record.get(input_col, "")
        if not isinstance(text, str) or not text:
            # Return zero vector or None? 
            # For now, let's return None and handle it downstream or let the model handle empty string
            embedding = []
        else:
            # embed_texts returns a numpy array (N, D)
            embeddings = self.model.embed_texts([text])
            embedding = embeddings[0].tolist()

        return {output_col: embedding}

def get_embedding_udf(
    model_name: str = "openai",
    input_col: str = "text",
    output_col: str = "embedding",
    config: Any = None
) -> callable:
    """
    Returns a UDF compatible with Dataset.map.
    """
    embedder = Embedder(model_name=model_name, config=config)
    
    def udf(record: Dict[str, Any]) -> Dict[str, Any]:
        return embedder.embed(record, input_col=input_col, output_col=output_col)
        
    return udf
