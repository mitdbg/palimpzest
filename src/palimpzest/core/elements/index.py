from __future__ import annotations

from abc import ABC, abstractmethod

from chromadb.api.models.Collection import Collection


def index_factory(index: Collection) -> PZIndex:
    """
    Factory function to create a PZ index based on the type of the provided index.

    Args:
        index (Collection): The index provided by the user.

    Returns:
        PZIndex: The PZ wrapped Index.
    """
    if isinstance(index, Collection):
        return ChromaIndex(index)
    else:
        raise TypeError(f"Unsupported index type: {type(index)}\nindex must be a `chromadb.api.models.Collection.Collection`")


class BaseIndex(ABC):

    def __init__(self, index: Collection):
        self.index = index

    def __str__(self):
        """
        Return a string representation of the index.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def search(self, query_embedding: list[float] | list[list[float]], results_per_query: int = 1) -> list | list[list]:
        """
        Query the index with a string or a list of strings.

        Args:
            query (str | list[str]): The query string or list of strings to search for.
            results_per_query (int): The number of top results to retrieve for each query.

        Returns:
            list | list[list]: The top results for the query. If query is a list, the top
                results for each query in the list are returned. Each list will contain the
                raw elements yielded by the index. This way, users can program against the
                results they expect to get from e.g. chromadb or ragatouille.
        """
        pass


class ChromaIndex(BaseIndex):
    def __init__(self, index: Collection):
        assert isinstance(index, Collection), "ChromaIndex input must be a `chromadb.api.models.Collection.Collection`"
        super().__init__(index)


# define type for PZIndex
PZIndex = ChromaIndex
