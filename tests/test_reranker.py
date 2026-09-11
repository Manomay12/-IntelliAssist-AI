"""
Unit tests for NeuralReranker.
"""

import pytest
from services.reranker import NeuralReranker


def test_reranker_sorting():
    reranker = NeuralReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    query = "transformer self-attention computational complexity"
    candidates = [
        {
            "id": "c1",
            "text": "Weather forecast indicates sunny skies across the western region.",
            "score": 0.4
        },
        {
            "id": "c2",
            "text": "Self-attention computational complexity scales quadratically O(N^2) with sequence length in standard transformers.",
            "score": 0.5
        },
        {
            "id": "c3",
            "text": "Cooking recipes for homemade pasta dough with flour and eggs.",
            "score": 0.3
        }
    ]

    reranked = reranker.rerank(query=query, candidates=candidates, top_k=2)
    assert len(reranked) <= 2
    # The relevant candidate about self-attention should be ranked first
    assert reranked[0]["id"] == "c2"
    assert "rerank_score" in reranked[0]


def test_reranker_empty_or_single():
    reranker = NeuralReranker()
    assert reranker.rerank("any query", []) == []

    single = [{"id": "s1", "text": "Only candidate here.", "score": 0.7}]
    res = reranker.rerank("test", single)
    assert len(res) == 1
    assert res[0]["id"] == "s1"
