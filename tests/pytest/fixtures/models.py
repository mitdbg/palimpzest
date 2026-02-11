from os import getenv

import pytest

from palimpzest.constants import Model


@pytest.fixture
def embedding_text_only_model():
        return Model.NOMIC_EMBED_TEXT if getenv("TESTS_USE_OLLAMA_FOR_EMBEDDING") else Model.TEXT_EMBEDDING_3_SMALL
