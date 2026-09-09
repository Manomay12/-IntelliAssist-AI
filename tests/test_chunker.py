"""
Unit tests for TextChunker service.
Tests recursive text splitting, boundary preservation, overlap, and metadata retention.
"""

# pyrefly: ignore [missing-import]
import pytest
from services.chunker import TextChunker

def test_split_short_text():
    """Test text smaller than chunk size returns single chunk."""
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    text = "Short text for quick chunking verification."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_split_long_text_with_overlap():
    """Test recursive overlapping chunk generation on paragraphs."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "First sentence explaining transformers. "
        "Second sentence discussing attention mechanism. "
        "Third sentence describing multi-head self-attention. "
        "Fourth sentence evaluating cross-attention in encoder-decoder."
    )
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 150  # Boundary check

def test_chunk_document_structure():
    """Test chunk_document retains page number, filename, and file_hash."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    mock_doc = {
        "filename": "ai_research.pdf",
        "file_hash": "a1b2c3d4e5f67890123456789012345678901234567890123456789012345678",
        "total_pages": 2,
        "pages": [
            {
                "page_number": 1,
                "total_pages": 2,
                "text": "Page 1 contains introduction to neural network architecture and optimization."
            },
            {
                "page_number": 2,
                "total_pages": 2,
                "text": "Page 2 presents empirical evaluation benchmarks and cross-validation accuracy."
            }
        ]
    }

    chunks = chunker.chunk_document(mock_doc)
    assert len(chunks) >= 2
    assert chunks[0]["filename"] == "ai_research.pdf"
    assert chunks[0]["file_hash"] == mock_doc["file_hash"]
    assert chunks[0]["page_number"] == 1
    assert any(c["page_number"] == 2 for c in chunks)
    assert all("chunk_id" in c for c in chunks)

def test_chunk_document_empty_fallback():
    """Test guaranteed non-empty chunk fallback for empty documents."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    mock_empty = {
        "filename": "blank.txt",
        "file_hash": "0000000000000000000000000000000000000000000000000000000000000000",
        "pages": [],
        "full_text": ""
    }
    chunks = chunker.chunk_document(mock_empty)
    assert len(chunks) == 1
    assert chunks[0]["filename"] == "blank.txt"
