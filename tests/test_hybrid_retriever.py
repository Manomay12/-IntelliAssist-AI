"""
Unit tests for Pure NumPy BM25 Index and Hybrid Retriever with Reciprocal Rank Fusion.
"""

import pytest
import numpy as np
from services.hybrid_retriever import BM25Index, HybridRetriever


def test_bm25_tokenization_and_indexing():
    corpus = [
        "Attention mechanisms in transformer neural networks optimize latency.",
        "Convolutional neural networks are ideal for image classification and computer vision.",
        "Retrieval augmented generation connects dense vectors to large language models."
    ]
    index = BM25Index(corpus=corpus, k1=1.5, b=0.75)
    assert index.doc_count == 3
    assert index.avgdl > 0

    scores = index.score("transformer attention")
    assert len(scores) == 3
    # First document should have highest score for transformer attention
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bm25_empty_query_or_corpus():
    index = BM25Index(corpus=[], k1=1.5, b=0.75)
    assert index.doc_count == 0
    scores = index.score("test query")
    assert len(scores) == 0

    index2 = BM25Index(corpus=["Single document content."])
    scores2 = index2.score("")
    assert np.all(scores2 == 0.0)


def test_hybrid_search_rrf():
    retriever = HybridRetriever(rrf_k=60)
    chunks = [
        {"id": "c1", "filename": "doc1.pdf", "text": "Deep residual networks for image recognition benchmarks."},
        {"id": "c2", "filename": "doc2.pdf", "text": "Language models are few shot learners via prompt engineering."},
        {"id": "c3", "filename": "doc1.pdf", "text": "Transformers with multi-head self-attention mechanisms."}
    ]
    retriever.update_bm25_index(chunks)

    # Simulated dense scores: c3 has highest dense score
    dense_scores = np.array([0.2, 0.4, 0.85], dtype=np.float32)

    results = retriever.hybrid_search(
        query="multi-head self-attention mechanisms",
        chunks=chunks,
        dense_scores=dense_scores,
        top_k=2
    )

    assert len(results) <= 2
    top_result = results[0]
    assert top_result["id"] == "c3"
    assert "match_type" in top_result
    assert "match_explanation" in top_result
    assert top_result["match_type"] in ["Hybrid Match", "Semantic Match", "Keyword Match"]


def test_hybrid_search_filter_doc():
    retriever = HybridRetriever(rrf_k=60)
    chunks = [
        {"id": "c1", "filename": "docA.pdf", "text": "Machine learning in healthcare and clinical diagnosis."},
        {"id": "c2", "filename": "docB.pdf", "text": "Machine learning in finance and algorithmic trading."}
    ]
    retriever.update_bm25_index(chunks)
    dense_scores = np.array([0.8, 0.85], dtype=np.float32)

    results = retriever.hybrid_search(
        query="machine learning",
        chunks=chunks,
        dense_scores=dense_scores,
        top_k=5,
        filter_doc="docA.pdf"
    )

    assert len(results) == 1
    assert results[0]["filename"] == "docA.pdf"
