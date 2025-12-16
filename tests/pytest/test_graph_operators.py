from unittest.mock import MagicMock

import pandas as pd
import pytest

from palimpzest.constants import Model
from palimpzest.core.data.graph_dataset import GraphDataset, GraphEdge, GraphNode
from palimpzest.core.data.iter_dataset import MemoryDataset
from palimpzest.core.models import GenerationStats
from palimpzest.policy import MaxQuality
from palimpzest.query.operators.aggregate import SemanticAggregate
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.utils.hash_helpers import hash_for_id


# Mock Data
@pytest.fixture
def sample_text_df():
    return pd.DataFrame({
        'id': ['1', '2'],
        'text': [
            'This is the first sentence. This is the second sentence.',
            'Another document here. It has some content.'
        ],
        'title': ['Doc 1', 'Doc 2']
    })

@pytest.fixture
def memory_dataset(sample_text_df):
    return MemoryDataset("test_ds", sample_text_df)

@pytest.fixture
def graph_dataset():
    # Create a simple in-memory graph dataset
    # We need to mock the underlying storage or use a temporary one if possible.
    # For now, let's assume we can instantiate GraphDataset with a mock store or just use it if it supports in-memory.
    # Looking at GraphDataset, it requires a store.
    # Let's mock the store or use a simple implementation if available.
    
    # Since setting up a real GraphDataset might be complex with dependencies, 
    # we will mock the graph object passed to summarize for the linking test.
    graph = MagicMock(spec=GraphDataset)
    graph.graph_id = "test_graph"
    graph.add_edge = MagicMock()
    return graph

@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    return QueryProcessorConfig(
        policy=MaxQuality(), 
        verbose=False,
        available_models=[Model.GPT_4o]
    )

# Tests

def test_chunk_operator(memory_dataset, config):
    """Test the .chunk() operator."""
    # Chunk with small size to force splits
    chunked_ds = memory_dataset.chunk(
        input_col="text",
        output_col="chunk_text",
        chunk_size=20,
        chunk_overlap=0
    )
    
    # Execute
    records = chunked_ds.run(config)
    
    # Verify
    assert len(records) > 2 # Should be more than original 2 records
    for record in records:
        assert hasattr(record, "chunk_text")
        assert hasattr(record, "chunk_index")
        assert hasattr(record, "source_node_id")
        assert hasattr(record, "prev_chunk_id")
        # Check metadata preservation
        assert hasattr(record, "title")
        
    # Check lineage
    # Group by source_node_id
    by_source = {}
    for r in records:
        sid = r.source_node_id
        if sid not in by_source:
            by_source[sid] = []
        by_source[sid].append(r)
        
    for _sid, chunks in by_source.items():
        chunks.sort(key=lambda x: x.chunk_index)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            if i > 0:
                assert chunk.prev_chunk_id == chunks[i-1].id
            else:
                assert chunk.prev_chunk_id is None


def test_chunk_operator_with_graph_edge_policy(config):
    """Chunk a GraphDataset and materialize has_chunk + next_chunk overlay edges."""
    graph = GraphDataset(graph_id="g")
    graph.add_node(
        GraphNode(
            id="n1",
            type="twiki_page",
            label="Doc",
            text="abcdefghij" * 3,  # 30 chars
            attrs={},
        )
    )

    # GraphDataset is a Dataset yielding GraphNode records
    chunked = graph.chunk(
        input_col="text",
        output_col="text",
        chunk_size=10,
        chunk_overlap=0,
        graph=graph,
        edge_policy="has_and_next",
        has_chunk_edge_type="overlay:has_chunk",
        next_chunk_edge_type="overlay:next_chunk",
        chunk_node_type="chunk",
    )

    out = chunked.run(config)
    chunk_records = list(out)
    assert len(chunk_records) == 3
    chunk_records.sort(key=lambda r: r.chunk_index)
    chunk_ids = [r.id for r in chunk_records]

    # Nodes materialized
    for cid in chunk_ids:
        assert graph.has_node(cid)
        node = graph.get_node(cid)
        assert node.type == "chunk"
        assert isinstance(node.text, str)
        assert node.attrs.get("chunk", {}).get("source_node_id") == "n1"

    # has_chunk edges: n1 -> chunk
    for cid in chunk_ids:
        eid = hash_for_id(f"link:overlay:has_chunk:n1->{cid}")
        assert graph.has_edge(eid)
        e = graph.get_edge(eid)
        assert e.src == "n1"
        assert e.dst == cid
        assert e.type == "overlay:has_chunk"

    # next_chunk edges: chunk0 -> chunk1 -> chunk2
    for i in range(1, len(chunk_ids)):
        src = chunk_ids[i - 1]
        dst = chunk_ids[i]
        eid = hash_for_id(f"link:overlay:next_chunk:{src}->{dst}")
        assert graph.has_edge(eid)
        e = graph.get_edge(eid)
        assert e.src == src
        assert e.dst == dst
        assert e.type == "overlay:next_chunk"

def test_embed_operator(memory_dataset, mocker, config):
    """Test the .embed() operator."""
    # Mock the Embedder class in palimpzest.utils.embeddings
    # We need to patch where it is imported/used.
    # The .embed() method imports get_embedding_udf inside the method.
    # We can patch palimpzest.utils.embeddings.Embedder
    
    mock_embedder_cls = mocker.patch("palimpzest.utils.embeddings.Embedder")
    mock_embedder_instance = mock_embedder_cls.return_value
    mock_embedder_instance.embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
    
    embedded_ds = memory_dataset.embed(
        input_col="text",
        output_col="embedding",
        model_name="openai"
    )
    
    records = embedded_ds.run(config)
    
    assert len(records) == 2
    for record in records:
        assert hasattr(record, "embedding")
        assert record.embedding == [0.1, 0.2, 0.3]

def test_summarize_one_to_one(memory_dataset, mocker, config):
    """Test .summarize() in 1-to-1 mode."""
    
    # Mock LLMConvertBonded.convert
    def mock_convert(candidate, fields):
        return {"summary": ["This is a summary."]}, GenerationStats()
        
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=mock_convert)
    
    summary_ds = memory_dataset.summarize(
        prompt="Summarize",
        output_col="summary",
        aggregate=False
    )
    
    records = summary_ds.run(config)
    
    assert len(records) == 2
    for record in records:
        assert hasattr(record, "summary")
        assert record.summary == "This is a summary."
        # Check lineage (parent_ids should be set)
        assert len(record._parent_ids) == 1

def test_summarize_aggregate(memory_dataset, mocker, config):
    """Test .summarize() in N-to-1 mode (Super-Node)."""
    
    # Mock SemanticAggregate.__call__ or generator
    # SemanticAggregate uses self.generator
    # We can patch SemanticAggregate.__call__ directly
    
    def mock_agg_call(self, candidates):
        # Return a DataRecordSet with one record
        from palimpzest.core.elements.records import DataRecord, DataRecordSet, RecordOpStats
        
        data_item = self.output_schema(summary="Super summary")
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)
        stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state={},
            full_op_id="op",
            logical_op_id="log_op",
            op_name="agg",
            time_per_record=0,
            cost_per_record=0
        )
        return DataRecordSet([dr], [stats])

    mocker.patch.object(SemanticAggregate, "__call__", mock_agg_call)
    
    # We need to group first to test aggregation effectively, or just aggregate the whole dataset
    # Let's aggregate the whole dataset
    summary_ds = memory_dataset.summarize(
        prompt="Summarize all",
        output_col="summary",
        aggregate=True
    )
    
    records = summary_ds.run(config)
    
    assert len(records) == 1
    assert records.data_records[0].summary == "Super summary"
    # Check lineage: should have 2 parents (the 2 records in memory_dataset)
    assert len(records.data_records[0]._parent_ids) == 2

def test_summarize_with_linking(memory_dataset, graph_dataset, mocker, config):
    """Test .summarize() with graph linking."""
    
    # Mock LLMConvertBonded for 1-to-1 summary
    def mock_convert(candidate, fields):
        return {"summary": ["Summary linked."]}, GenerationStats()
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=mock_convert)
    
    # Use the mock graph dataset
    summary_ds = memory_dataset.summarize(
        prompt="Summarize",
        output_col="summary",
        aggregate=False,
        graph=graph_dataset,
        edge_type="SUMMARIZES"
    )
    
    records = summary_ds.run(config)
    
    assert len(records) == 2
    
    # Verify edges were added to the graph
    # We expect 2 calls to add_edge (one for each record)
    assert graph_dataset.add_edge.call_count == 2
    
    # Verify edge properties
    calls = graph_dataset.add_edge.call_args_list
    for call in calls:
        edge = call[0][0] # First arg is the edge
        assert isinstance(edge, GraphEdge)
        assert edge.type == "SUMMARIZES"
        # src should be the summary record id
        # dst should be the parent record id
        # We can't easily check exact IDs without more complex setup, but we can check structure
