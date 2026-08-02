"""Tests for chunking. Start here: it's the logic most likely to break."""
import pytest

from rag_assistant.ingestion import chunk_text


def test_chunk_respects_size_and_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_chunk_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []
