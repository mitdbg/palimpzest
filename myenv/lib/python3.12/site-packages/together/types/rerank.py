from __future__ import annotations

from typing import List, Literal, Dict, Any

from together.types.abstract import BaseModel
from together.types.common import UsageData


class RerankRequest(BaseModel):
    # model to query
    model: str
    # input or list of inputs
    query: str
    # list of documents
    documents: List[str] | List[Dict[str, Any]]
    # return top_n results
    top_n: int | None = None
    # boolean to return documents
    return_documents: bool = False
    # field selector for documents
    rank_fields: List[str] | None = None


class RerankChoicesData(BaseModel):
    # response index
    index: int
    # object type
    relevance_score: float
    # rerank response
    document: Dict[str, Any] | None = None


class RerankResponse(BaseModel):
    # job id
    id: str | None = None
    # object type
    object: Literal["rerank"] | None = None
    # query model
    model: str | None = None
    # list of reranked results
    results: List[RerankChoicesData] | None = None
    # usage stats
    usage: UsageData | None = None
