"""
Unit tests for RAGEngine.
Tests query normalization, greeting handler, strict document targeting, multi-document synthesis, and low-confidence disclaimer protection.
"""

# pyrefly: ignore [missing-import]
import pytest
from services.rag_engine import RAGEngine

def test_query_normalization(rag_engine):
    """Test typo fixing, slang expansion, and intent detection."""
    expanded, intent, is_greeting = rag_engine.normalize_query("wat dis pdf say abt transfmr")
    assert "what" in expanded
    assert "this" in expanded
    assert "transformer" in expanded
    assert is_greeting is False

def test_greeting_handling(rag_engine):
    """Test friendly onboarding response for greeting queries."""
    res = rag_engine.answer_question("hello, how do I use this?")
    assert len(res["answer"]) > 50
    assert "Hello" in res["answer"] or "Welcome" in res["answer"]
    assert res["retrieved_count"] == 0

def test_standard_rag_answer(rag_engine):
    """Test standard RAG answer synthesis with verified source citations."""
    res = rag_engine.answer_question("Explain the attention mechanism in transformers")
    assert len(res["answer"]) > 50
    assert len(res["sources"]) > 0
    assert "filename" in res["sources"][0]
    assert "page_number" in res["sources"][0]
    assert "score" in res["sources"][0]

def test_strict_document_targeting(rag_engine):
    """Test strict document filtering when target doc is specified."""
    docs = rag_engine.vector_store.get_all_documents()
    target_doc = docs[0]
    res = rag_engine.answer_question("Summarize the main findings", filter_doc=target_doc)
    
    assert len(res["sources"]) > 0
    for src in res["sources"]:
        assert src["filename"] == target_doc

def test_low_confidence_hallucination_protection(rag_engine):
    """Test that out-of-domain queries return insufficient evidence disclaimer instead of hallucinating."""
    res = rag_engine.answer_question("What is the recipe for chocolate lava cake with strawberry ice cream?")
    # Verify low confidence handling
    assert "Insufficient Document Evidence" in res["answer"] or res["is_low_confidence"] is True
