"""
Unit tests for DocumentSummarizer and Cross-Document Comparison.
Tests 5 summary modes, topic modeling, entity discovery, and comparative matrix generation.
"""

# pyrefly: ignore [missing-import]
import pytest
from services.summarizer import DocumentSummarizer

SAMPLE_DOC_TEXT = """
Attention Is All You Need introduces the Transformer network architecture based entirely on self-attention mechanisms.
Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable
and requiring significantly less time to train. The model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task,
improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs.
"""

SAMPLE_DOC_B_TEXT = """
Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks combines pre-trained parametric and non-parametric memory.
We build RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index
of Wikipedia, accessed with a neural retriever. We evaluate on a wide range of knowledge-intensive benchmarks, including Natural Questions,
TriviaQA, and CuratedTREC, achieving state-of-the-art results on open-domain question answering.
"""

def test_summary_modes(summarizer):
    """Test all 5 summary modes generate non-empty structured output."""
    modes = ["Executive Summary", "Detailed Summary", "Key Findings", "Bullet Points", "Quick Summary"]
    for m in modes:
        res = summarizer.summarize(SAMPLE_DOC_TEXT, mode=m, doc_name="Transformers_Paper.pdf")
        assert len(res["summary"]) > 20
        assert res["mode"] == m
        assert len(res["topics"]) > 0

def test_entity_and_topic_extraction(summarizer):
    """Test topic extraction and entity discovery."""
    topics = summarizer.extract_key_topics(SAMPLE_DOC_TEXT)
    assert len(topics) > 0

    entities = summarizer.extract_entities(SAMPLE_DOC_TEXT)
    assert isinstance(entities, dict)
    assert "Metrics & Percentages" in entities or "Technologies & Frameworks" in entities

def test_compare_documents(summarizer):
    """Test cross-document comparison matrix and report generation."""
    comp = summarizer.compare_documents(
        doc_a_name="Transformers.pdf",
        doc_a_text=SAMPLE_DOC_TEXT,
        doc_b_name="RAG_Paper.pdf",
        doc_b_text=SAMPLE_DOC_B_TEXT
    )

    assert comp["doc_a"] == "Transformers.pdf"
    assert comp["doc_b"] == "RAG_Paper.pdf"
    assert "| Comparative Dimension |" in comp["comparison_table"]
    assert "Transformers.pdf" in comp["comparison_table"]
    assert "RAG_Paper.pdf" in comp["comparison_table"]
    assert len(comp["executive_synthesis"]) > 50
    assert len(comp["full_report"]) > 100
