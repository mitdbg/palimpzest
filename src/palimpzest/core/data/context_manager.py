from __future__ import annotations

import os
import pickle

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import tiktoken

from palimpzest.constants import PZ_DIR
from palimpzest.core.data import context


class ContextNotFoundError(Exception):
    pass


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

    @staticmethod
    def from_pkl(path: str) -> context.Context:
        """Load a `Context` from its serialized pickle file."""
        with open(path, "rb") as f:
            context = pickle.load(f)

        return context

    @staticmethod
    def to_pkl(context: context.Context, path: str) -> None:
        """Write the given `Context` to a pickle file at the provided `path`."""
        with open(path, "wb") as f:
            pickle.dump(context, f)

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def add_context(self, context: context.Context, update: bool = False) -> None:
        """
        Add the new `Context` to the `ContextManager` by serializing and writing it to disk.

        Args:
            context (`Context`): the context to add to the `ContextManager`
            update (`bool`): whether or not to update an existing context

        TODO: track cost
        """
        # return early if the context already exists and we're not performing an update
        id = context.id
        context_path = os.path.join(self.context_dir, f"{id}.pkl")
        if os.path.exists(context_path) and update is False:
            return

        # write the context to disk
        ContextManager.to_pkl(context, context_path)

        # compute number of tokens in context.description
        description = context.description
        while self.num_tokens_from_string(description, "cl100k_base") > 8192:
            description = description[:int(0.9*len(description))]
 
        # add context to vector store
        context_embeddings = self.emb_fn([description])
        context_payload = {
            "ids": [context.id],
            "embeddings": context_embeddings,
            "metadatas": [{"id": context.id, "materialized": context.materialized}],
            "documents": [context.description],
        }
        if update:
            self.index.update(**context_payload)
        else:
            self.index.add(**context_payload)

    def update_context(self, id: str, description: str, materialized: bool = True) -> None:
        """
        Update an existing `Context` with the given `id` to have the given `description`.
        
        Args:
            id (str): the id of the updated `Context`
            description (str): the update to the description for the specified `Context`
            materialized (bool): boolean to set the materialization status of the `Context`

        Raises:
            ContextNotFoundError: if the given `id` doesn't point to a `Context` in the `ContextManger`.
        """
        context = self.get_context(id)
        new_description = context.description + description  # TODO: should description have RESULT replaced on update? as opposed to appending? should description be some pydantic BaseModel?
        context.set_description(new_description)
        context.set_materialized(materialized)
        self.add_context(context, update=True)

    def get_context(self, id: str) -> context.Context:
        """
        Returns the `Context` specified by the given `id`.

        Args:
            id (str): the id of the retrieved `Context`

        Returns:
            `Context`: the specified `Context`.
        """
        context_path = os.path.join(self.context_dir, f"{id}.pkl")
        try:
            return ContextManager.from_pkl(context_path)
        except FileNotFoundError as err:
            raise ContextNotFoundError from err

    def search_context(self, query: str, k: int = 1, where: dict | None = None) -> list[context.Context]:
        """
        Returns the top-k most relevant `Context(s)` for the given query. If provided,
        the where dictionary will be used to filter the search results.

        TODO:
        3) update CostModel to account for benefit of using existing Context(s)
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
            where=where,
        )
        ids = results["ids"][0]

        # load and return Context objects
        contexts = []
        for id in ids:
            context_path = os.path.join(self.context_dir, f"{id}.pkl")
            contexts.append(ContextManager.from_pkl(context_path))

        return contexts
