import pytest

from palimpzest.utils.chunkers import get_chunking_udf


def test_chunking_udf_assigns_unique_ids_without_source_id() -> None:
    udf = get_chunking_udf(chunk_size=3, chunk_overlap=0, input_col="text", output_col="text")

    out = udf({"text": "abcdef"})
    assert [r["text"] for r in out] == ["abc", "def"]

    ids = [r["id"] for r in out]
    assert len(set(ids)) == len(ids)

    assert out[0]["prev_chunk_id"] is None
    assert out[1]["prev_chunk_id"] == out[0]["id"]


def test_chunk_ids_change_with_chunking_config() -> None:
    rec = {"id": "n1", "text": "abcdefghij"}

    udf_a = get_chunking_udf(chunk_size=5, chunk_overlap=0, input_col="text", output_col="text")
    udf_b = get_chunking_udf(chunk_size=4, chunk_overlap=0, input_col="text", output_col="text")

    out_a = udf_a(rec)
    out_b = udf_b(rec)

    assert out_a[0]["id"] != out_b[0]["id"]


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        get_chunking_udf(chunk_size=10, chunk_overlap=10)
