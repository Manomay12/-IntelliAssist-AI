"""
Unit tests for VectorStore.
Tests document chunk indexing, cosine similarity search, disk persistence, document deletion, and duplicate hash checking.
"""

import pytest
import numpy as np
from services.vector_store import VectorStore

def test_add_and_search_documents(temp_vector_store, sample_chunks):
    """Test indexing chunks and querying top-k results with cosine similarity."""
    added = temp_vector_store.add_documents(sample_chunks)
    assert added == len(sample_chunks)
    assert temp_vector_store.total_chunks == len(sample_chunks)

    results = temp_vector_store.search("transformer attention mechanism", top_k=3)
    assert len(results) > 0
    assert "score" in results[0]
    assert "similarity_percentage" in results[0]
    assert results[0]["score"] >= 0.0
    assert results[0]["score"] <= 1.0

def test_document_scoping_filter(populated_vector_store):
    """Test search strictly filters results when filter_doc is specified."""
    docs = populated_vector_store.get_all_documents()
    assert len(docs) >= 2
    target = docs[0]

    filtered_results = populated_vector_store.search("architecture", top_k=5, filter_doc=target)
    for r in filtered_results:
        assert r["filename"] == target

def test_duplicate_hash_detection(temp_vector_store):
    """Test duplicate detection via SHA-256 file_hash."""
    mock_chunks = [
        {
            "chunk_id": "doc1_p1_c0",
            "filename": "original_doc.pdf",
            "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "page_number": 1,
            "total_pages": 1,
            "text": "Unique research content on deep learning systems."
        }
    ]
    temp_vector_store.add_documents(mock_chunks)

    # Verify hash exists
    assert temp_vector_store.has_document_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") is True
    assert temp_vector_store.has_document_hash("non_existent_hash") is False
    assert temp_vector_store.get_document_by_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") == "original_doc.pdf"

def test_delete_document(populated_vector_store):
    """Test removing a document and re-indexing vector array."""
    docs = populated_vector_store.get_all_documents()
    doc_to_delete = docs[0]
    initial_chunks = populated_vector_store.total_chunks

    removed = populated_vector_store.delete_document(doc_to_delete)
    assert removed > 0
    assert populated_vector_store.total_chunks == initial_chunks - removed
    assert doc_to_delete not in populated_vector_store.get_all_documents()

def test_save_and_load_from_disk(tmp_path, embedding_service, sample_chunks):
    """Test vector database persistence to disk and reload."""
    db_dir = tmp_path / "persistence_db"
    store1 = VectorStore(embedding_service=embedding_service, db_dir=db_dir)
    store1.add_documents(sample_chunks)
    assert store1.total_chunks == len(sample_chunks)

    # Create new store instance pointing to same directory
    store2 = VectorStore(embedding_service=embedding_service, db_dir=db_dir)
    assert store2.total_chunks == len(sample_chunks)
    assert store2.get_all_documents() == store1.get_all_documents()
