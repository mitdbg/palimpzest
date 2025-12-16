from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from palimpzest.utils.hash_helpers import hash_for_serialized_dict


class _TextSplitter(Protocol):
    def split_text(self, text: str) -> list[str]: ...


class _SimpleCharacterSplitter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        out: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            if chunk:
                out.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return out


@dataclass(frozen=True)
class ChunkerConfig:
    kind: str = "recursive_character"
    params: dict[str, Any] | None = None


def _build_splitter(*, chunk_size: int, chunk_overlap: int, kind: str, params: dict[str, Any] | None) -> _TextSplitter:
    """Create a text splitter.

    Defaults to a small built-in splitter if langchain is unavailable.
    """
    raw_kind = (kind or "recursive_character").strip()
    kind_key = raw_kind.lower()
    params = {} if params is None else dict(params)

    try:
        import langchain_text_splitters as lts  # type: ignore
    except Exception as err:
        # Best-effort fallback.
        if kind_key not in {"simple", "character", "recursive_character"}:
            raise ImportError(
                f"langchain_text_splitters is required for chunker_kind={raw_kind!r}. "
                "Install dependencies (e.g. `pip install -e .`) or use chunker_kind='simple'."
            ) from err
        return _SimpleCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Common kinds mapped to known splitters.
    if kind_key in {"recursive_character", "recursive", "recursive-character"}:
        cls = getattr(lts, "RecursiveCharacterTextSplitter", None)
        if cls is None:
            return _SimpleCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **params)

    if kind_key in {"character", "char"}:
        cls = getattr(lts, "CharacterTextSplitter", None)
        if cls is None:
            return _SimpleCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **params)

    if kind_key in {"token", "tokens"}:
        cls = getattr(lts, "TokenTextSplitter", None)
        if cls is None:
            raise ValueError("TokenTextSplitter not available in installed langchain-text-splitters")
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **params)

    if kind_key in {"markdown", "md"}:
        cls = getattr(lts, "MarkdownTextSplitter", None)
        if cls is None:
            raise ValueError("MarkdownTextSplitter not available in installed langchain-text-splitters")
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **params)

    # Allow advanced users to pass a class name, e.g. "MarkdownHeaderTextSplitter".
    candidates = [
        raw_kind,
        raw_kind.replace("-", "_"),
        "".join([p.capitalize() for p in raw_kind.replace("-", "_").split("_")]),
    ]
    cls = None
    for name in candidates:
        if not name:
            continue
        cls = getattr(lts, name, None)
        if cls is not None:
            break
    if cls is None:
        for attr in dir(lts):
            if attr.lower() == kind_key:
                cls = getattr(lts, attr, None)
                break
    if cls is None:
        raise ValueError(f"Unknown chunker_kind: {raw_kind!r}")
    return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **params)


class LangChainChunker:
    """A wrapper around text splitters to be used with Palimpzest's flat_map operator."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        chunker_kind: str = "recursive_character",
        chunker_params: dict[str, Any] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.chunker_kind = (chunker_kind or "recursive_character").strip()
        self.chunker_params: dict[str, Any] = {} if chunker_params is None else dict(chunker_params)

        self.splitter = _build_splitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            kind=self.chunker_kind,
            params=self.chunker_params,
        )

    def chunk(self, record: dict[str, Any], input_col: str = "text", output_col: str = "text") -> list[dict[str, Any]]:
        """
        Chunks the 'text' field of the input record.
        Returns a list of records, each containing a chunk and metadata.
        """
        text = record.get(input_col, "")
        if not isinstance(text, str):
            return []

        # Use create_documents to get metadata handling if needed, 
        # but split_text is simpler for just text.
        # We want to preserve original record metadata potentially?
        # For now, let's just return the chunks and link back to source.
        
        chunks = self.splitter.split_text(text)

        # Preserve original source id if already present (e.g. re-chunking a chunk).
        explicit_source = record.get("source_node_id")
        if isinstance(explicit_source, str) and explicit_source:
            source_node_id = explicit_source
        else:
            rid = record.get("id")
            source_node_id = rid if isinstance(rid, str) and rid else None

        # Always generate deterministic chunk ids. Include the chunking configuration
        # so reruns with different chunk_size/overlap don't collide.
        source_for_id = source_node_id
        if source_for_id is None:
            source_for_id = hash_for_serialized_dict({"record": record, "input_col": input_col})
        chunk_ns = hash_for_serialized_dict(
            {
                "chunker_kind": self.chunker_kind,
                "chunker_params": self.chunker_params,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "input_col": input_col,
                "output_col": output_col,
            }
        )
        
        output_records: list[dict[str, Any]] = []
        prev_chunk_id: str | None = None
        for i, chunk in enumerate(chunks):
            # Use an orderable id so any internal sorting keeps chunk order.
            # Include a config-derived namespace to avoid collisions across chunking configs.
            chunk_id = f"{source_for_id}:chunk:{chunk_ns}:{i:06d}"
            
            # Start with a copy of the original record to preserve metadata
            new_record = record.copy()
            
            # Update fields
            new_record[output_col] = chunk
            new_record["id"] = chunk_id
            new_record["chunk_index"] = i
            new_record["source_node_id"] = source_node_id or source_for_id
            new_record["prev_chunk_id"] = prev_chunk_id

            prev_chunk_id = chunk_id
                
            output_records.append(new_record)
            
        return output_records

def get_chunking_udf(
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    input_col: str = "text",
    output_col: str = "text",
    chunker_kind: str = "recursive_character",
    chunker_params: dict[str, Any] | None = None,
) -> callable:
    """
    Returns a UDF compatible with Dataset.flat_map.
    """
    chunker = LangChainChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunker_kind=chunker_kind,
        chunker_params=chunker_params,
    )
    
    def udf(record: dict[str, Any]) -> list[dict[str, Any]]:
        return chunker.chunk(record, input_col=input_col, output_col=output_col)
        
    return udf
