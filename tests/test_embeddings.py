"""
Unit tests for EmbeddingService.
Tests 384-dimensional vector output, L2 normalization, query embedding, and active model reporting.
"""

# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
import numpy as np
from services.embeddings import EmbeddingService

def test_embedding_dimensions(embedding_service):
    """Test generated embedding vectors are (N, 384) dimensional float32 arrays."""
    texts = [
        "What is the attention mechanism in transformers?",
        "Medical decision support systems in modern hospitals."
    ]
    vectors = embedding_service.embed_texts(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32

def test_embedding_normalization(embedding_service):
    """Test all generated embedding vectors are L2-normalized (norm == 1.0)."""
    texts = ["Cosine similarity retrieval in vector databases."]
    vectors = embedding_service.embed_texts(texts)
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-4

def test_embed_query(embedding_service):
    """Test single query embedding generation."""
    query = "deep learning neural architectures"
    q_vec = embedding_service.embed_query(query)
    assert isinstance(q_vec, np.ndarray)
    assert q_vec.shape == (384,)
    norm = np.linalg.norm(q_vec)
    assert abs(norm - 1.0) < 1e-4

def test_active_model_name(embedding_service):
    """Test honest reporting of active embedding model name."""
    model_name = embedding_service.get_active_model_name()
    assert isinstance(model_name, str)
    assert len(model_name) > 0
    assert any(term in model_name for term in ["sentence-transformers", "Gemini", "OpenAI", "Projection", "Fallback"])

def test_empty_input(embedding_service):
    """Test empty input handles gracefully without error."""
    empty_vecs = embedding_service.embed_texts([])
    assert empty_vecs.shape == (0, 384)
    
    empty_q = embedding_service.embed_query("")
    assert empty_q.shape == (384,)
