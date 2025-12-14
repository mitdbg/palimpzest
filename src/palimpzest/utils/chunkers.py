from typing import Any, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LangChainChunker:
    """
    A wrapper around LangChain's text splitters to be used with Palimpzest's flat_map operator.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        separators: List[str] | None = None,
        is_separator_regex: bool = False,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators,
            is_separator_regex=is_separator_regex,
        )

    def chunk(self, record: Dict[str, Any], input_col: str = "text", output_col: str = "text") -> List[Dict[str, Any]]:
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
        source_node_id = record.get("id")
        
        output_records = []
        for i, chunk in enumerate(chunks):
            # Generate a deterministic ID for the chunk if we have a source ID
            chunk_id = f"{source_node_id}_chunk_{i}" if source_node_id else None
            prev_chunk_id = f"{source_node_id}_chunk_{i-1}" if source_node_id and i > 0 else None
            
            # Start with a copy of the original record to preserve metadata
            new_record = record.copy()
            
            # Update fields
            new_record[output_col] = chunk
            new_record["chunk_index"] = i
            new_record["source_node_id"] = source_node_id
            new_record["prev_chunk_id"] = prev_chunk_id
            
            if chunk_id:
                new_record["id"] = chunk_id
                
            output_records.append(new_record)
            
        return output_records

def get_chunking_udf(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    input_col: str = "text",
    output_col: str = "text"
) -> callable:
    """
    Returns a UDF compatible with Dataset.flat_map.
    """
    chunker = LangChainChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def udf(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        return chunker.chunk(record, input_col=input_col, output_col=output_col)
        
    return udf
