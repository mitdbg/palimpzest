"""Tests for the physical= query hinting system for physical operator selection."""

import pytest
from pydantic import BaseModel, Field

from palimpzest.constants import Cardinality, Model
from palimpzest.core.elements.filters import Filter
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.logical import ConvertScan, FilteredScan
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsFilter
from palimpzest.query.operators.rag import RAGFilter
from palimpzest.query.optimizer.tasks import _filter_expressions_by_physical


# --- Fixtures ---


@pytest.fixture
def schema():
    class SimpleSchema(BaseModel):
        text: str = Field(description="The text of the document")

    return SimpleSchema


@pytest.fixture
def output_schema():
    class OutputSchema(BaseModel):
        text: str = Field(description="The text of the document")
        summary: str = Field(description="Summary of the text")

    return OutputSchema


# --- Mock physical expressions ---


class _MockOp:
    """A lightweight stand-in for a physical operator."""

    def __init__(self, cls):
        self.__class__ = cls


def _make_mock_expr(cls):
    op = _MockOp(cls)
    expr = type("Expr", (), {"operator": op, "expr_id": f"{cls.__name__}-{id(op)}"})()
    return expr


# --- Tests for _filter_expressions_by_physical ---


class TestFilterExpressionsByPhysical:
    def test_impl_filters_by_exact_class(self):
        exprs = [
            _make_mock_expr(LLMFilter),
            _make_mock_expr(NonLLMFilter),
            _make_mock_expr(MixtureOfAgentsFilter),
        ]
        result = _filter_expressions_by_physical(exprs, {"implementation": LLMFilter})
        assert len(result) == 1
        assert type(result[0].operator) is LLMFilter

    def test_impl_no_subclass_matching(self):
        """Exact type match — LLMFilter should NOT match MixtureOfAgentsFilter subclass."""
        exprs = [
            _make_mock_expr(LLMFilter),
            _make_mock_expr(MixtureOfAgentsFilter),
        ]
        result = _filter_expressions_by_physical(exprs, {"implementation": LLMFilter})
        assert len(result) == 1
        assert type(result[0].operator) is LLMFilter

    def test_no_match_returns_empty(self):
        exprs = [_make_mock_expr(LLMFilter)]
        result = _filter_expressions_by_physical(exprs, {"implementation": RAGFilter})
        assert len(result) == 0

    def test_none_physical_returns_all(self):
        exprs = [_make_mock_expr(LLMFilter), _make_mock_expr(NonLLMFilter)]
        result = _filter_expressions_by_physical(exprs, None)
        assert len(result) == 2

    def test_no_implementation_key_returns_all(self):
        exprs = [_make_mock_expr(LLMFilter), _make_mock_expr(NonLLMFilter)]
        result = _filter_expressions_by_physical(exprs, {"model": Model.GPT_4o})
        assert len(result) == 2

    def test_extra_kwargs_ignored_during_filtering(self):
        """Extra keys beyond implementation don't affect filtering."""
        exprs = [_make_mock_expr(LLMFilter), _make_mock_expr(RAGFilter)]
        result = _filter_expressions_by_physical(
            exprs, {"implementation": RAGFilter, "chunk_size": 2000}
        )
        assert len(result) == 1
        assert type(result[0].operator) is RAGFilter


# --- Tests for physical validation ---


class TestPhysicalValidation:
    def test_valid_physical_dict(self, schema):
        """LLMFilter accepts 'model' via its constructor."""
        f = Filter("text contains 'hello'")
        op = FilteredScan(
            input_schema=schema, output_schema=schema, filter=f,
            physical={"implementation": LLMFilter, "model": Model.GPT_4o},
        )
        assert op.physical["model"] is Model.GPT_4o

    def test_rejects_missing_implementation(self, schema):
        f = Filter("text contains 'hello'")
        with pytest.raises(ValueError, match="implementation"):
            FilteredScan(
                input_schema=schema, output_schema=schema, filter=f,
                physical={"model": Model.GPT_4o},
            )

    def test_rejects_non_class_implementation(self, schema):
        f = Filter("text contains 'hello'")
        with pytest.raises(TypeError, match="must be a class"):
            FilteredScan(
                input_schema=schema, output_schema=schema, filter=f,
                physical={"implementation": "LLMFilter"},
            )

    def test_accepts_valid_rag_kwargs(self, schema):
        """RAGFilter accepts chunk_size, embedding_model, num_chunks_per_field."""
        f = Filter("text contains 'hello'")
        op = FilteredScan(
            input_schema=schema, output_schema=schema, filter=f,
            physical={
                "implementation": RAGFilter,
                "model": Model.GPT_4o,
                "embedding_model": Model.TEXT_EMBEDDING_3_SMALL,
                "chunk_size": 2000,
                "num_chunks_per_field": 4,
            },
        )
        assert op.physical["chunk_size"] == 2000

    def test_no_physical_is_fine(self, schema):
        f = Filter("text contains 'hello'")
        op = FilteredScan(input_schema=schema, output_schema=schema, filter=f)
        assert op.physical is None


# --- Tests for physical propagation through logical operators ---


class TestPhysicalPropagation:
    def test_physical_not_in_id_params(self, schema):
        """physical should NOT affect the logical operator identity."""
        f = Filter("text contains 'hello'")
        op_with = FilteredScan(
            input_schema=schema, output_schema=schema, filter=f,
            physical={"implementation": LLMFilter},
        )
        op_without = FilteredScan(input_schema=schema, output_schema=schema, filter=f)
        assert op_with.get_logical_op_id() == op_without.get_logical_op_id()

    def test_copy_preserves_physical(self, schema):
        f = Filter("text contains 'hello'")
        phys = {"implementation": MixtureOfAgentsFilter}
        op = FilteredScan(input_schema=schema, output_schema=schema, filter=f, physical=phys)
        op_copy = op.copy()
        assert op_copy.physical == phys


# --- Tests for physical in Dataset API ---


class TestDatasetPhysicalAPI:
    def test_sem_filter_accepts_physical(self):
        from palimpzest.core.data.iter_dataset import MemoryDataset

        ds = MemoryDataset(id="test", vals=["hello", "world"])
        phys = {"implementation": LLMFilter}
        result = ds.sem_filter("text contains 'hello'", physical=phys)
        assert result._operator.physical is phys

    def test_sem_map_accepts_physical(self):
        from palimpzest.core.data.iter_dataset import MemoryDataset

        ds = MemoryDataset(id="test", vals=["hello", "world"])
        phys = {"implementation": LLMConvertBonded, "model": Model.GPT_4o}
        result = ds.sem_map(
            [{"name": "summary", "desc": "Summary", "type": str}],
            physical=phys,
        )
        assert result._operator.physical is phys

    def test_sem_flat_map_accepts_physical(self):
        from palimpzest.core.data.iter_dataset import MemoryDataset

        ds = MemoryDataset(id="test", vals=["hello", "world"])
        phys = {"implementation": LLMConvertBonded}
        result = ds.sem_flat_map(
            [{"name": "word", "desc": "A word", "type": str}],
            physical=phys,
        )
        assert result._operator.physical is phys


# --- Test end-to-end usage pattern ---


class TestUsagePattern:
    def test_example_usage_pattern(self):
        from palimpzest.core.data.iter_dataset import MemoryDataset

        ds = MemoryDataset(id="demo", vals=["a", "b"])
        result = ds.sem_filter(
            "text is scientific",
            physical={"implementation": LLMFilter, "model": Model.GPT_4o},
        )
        assert result._operator.physical["implementation"] is LLMFilter
        assert result._operator.physical["model"] is Model.GPT_4o
