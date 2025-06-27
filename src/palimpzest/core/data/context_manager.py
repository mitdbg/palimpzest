from __future__ import annotations

import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from palimpzest.constants import PZ_DIR
from palimpzest.core.data import context


class ContextManager:
    """
    This class manages the long-term storage of `Contexts`. Each new `Context` is added to
    the `ContextManager` and serialized to disk. `Contexts` are also indexed, which enables
    PZ to search for `Context(s)` which may support `search()` and `compute()` operations.
    """
    def __init__(self):
        # create directory with serialized contexts (if it doesn't already exist)
        self.context_dir = os.path.join(PZ_DIR, "contexts")
        os.makedirs(self.context_dir, exist_ok=True)

        # create vector store (if it doesn't already exist)
        self.chroma_dir = os.path.join(PZ_DIR, "chroma")
        os.makedirs(self.chroma_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(self.chroma_dir)

        # pick embedding function based on presence of API key(s)
        self.emb_fn = None
        if os.getenv("OPENAI_API_KEY"):
            self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )

        self.index = self.chroma_client.get_or_create_collection("contexts", embedding_function=self.emb_fn)

    def add_context(self, context: context.Context) -> None:
        """
        Add the new `Context` to the `ContextManager` by serializing and writing it to disk.

        TODO: track cost
        """
        # write context to dict
        id = context.id
        context_path = os.path.join(self.context_dir, f"{id}.pkl")
        context.to_pkl(context_path)
        # context_json = context.to_json()
        # with open(context_path, "w") as f:
        #     json.dump(context_json, f)

        # add context to vector store
        context_embeddings = self.emb_fn([context._description])
        self.index.upsert(
            ids=[context.id],
            embeddings=context_embeddings,
            metadatas=[{"id": context.id}],
            documents=[context._description],
        )

    def update_context(self, context: context.Context) -> None:
        """
        Update an existing `Context` to reflect changes following a computation.
        """
        self.add_context(context)

    def search_context(self, query: str, k: int = 1) -> list[context.Context]:
        """
        Returns the top-k most relevant `Context(s)` for the given query.

        TODO:
        3) update CostModel to account for benefit of using existing Context(s)
        3.5) try running multiple PZ queries for Kramabench and see if they re-use context(s) effectively
        ---
        4) unit test
        5) track cost
        """
        # embed the search query
        query_embeddings = self.emb_fn([query])

        # look up ids of most similar contexts
        results = self.index.query(
            query_embeddings=query_embeddings,
            n_results=k,
        )
        ids = results["ids"]

        # load and return Context objects
        contexts = []
        for id in ids:
            context_path = os.path.join(self.context_dir, f"{id}.pkl")
            contexts.append(context.Context.from_pkl(context_path))

        return contexts
