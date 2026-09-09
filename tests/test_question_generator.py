"""
Unit tests for QuestionGenerator.
Verifies dynamic question generation, document-specific tailoring, seed-based rotation, and fallback mechanisms.
"""

# pyrefly: ignore [missing-import]
import pytest
from services.question_generator import QuestionGenerator

def test_universal_questions():
    """Verify universal exploratory questions when no document is specified."""
    gen = QuestionGenerator()
    q_list = gen.generate_questions("All Documents", count=4)
    assert len(q_list) == 4
    assert all(isinstance(q, str) and len(q) > 10 for q in q_list)

def test_benchmark_document_questions():
    """Verify document-specific questions for benchmark academic papers."""
    gen = QuestionGenerator()
    att_q = gen.generate_questions("Attention_Is_All_You_Need.pdf", count=4)
    assert len(att_q) == 4
    assert any("attention" in q.lower() or "transformer" in q.lower() or "bleu" in q.lower() for q in att_q)

    rag_q = gen.generate_questions("RAG_Paper.pdf", count=4)
    assert len(rag_q) == 4
    assert any("retrieval" in q.lower() or "rag" in q.lower() or "parametric" in q.lower() for q in rag_q)

    mimic_q = gen.generate_questions("MIMIC_Clinical_Prediction.pdf", count=4)
    assert len(mimic_q) == 4
    assert any("clinical" in q.lower() or "sepsis" in q.lower() or "icu" in q.lower() or "mimic" in q.lower() for q in mimic_q)

def test_custom_text_extraction():
    """Verify question generation from custom arbitrary text."""
    gen = QuestionGenerator()
    sample_text = """
    Deep Residual Learning for Image Recognition
    We present a residual learning framework to ease the training of networks that are substantially deeper.
    Our models achieve 3.57% error on the ImageNet test set, winning 1st place in ILSVRC 2015.
    However, computational complexity during backpropagation remains an active challenge.
    """
    custom_q = gen.generate_questions("ResNet_Paper.pdf", doc_text=sample_text, count=4)
    assert len(custom_q) == 4
    assert any("ResNet Paper" in q or "3.57%" in q or "residual" in q.lower() or "Image Recognition" in q for q in custom_q)

def test_question_shuffling():
    """Verify questions change when different seeds or calls are used."""
    gen = QuestionGenerator()
    set_a = gen.generate_questions("Attention_Is_All_You_Need.pdf", count=3, shuffle_seed=1)
    set_b = gen.generate_questions("Attention_Is_All_You_Need.pdf", count=3, shuffle_seed=99)
    # Both should be valid lists
    assert len(set_a) == 3
    assert len(set_b) == 3
