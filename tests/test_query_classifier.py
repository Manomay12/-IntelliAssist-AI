"""
Unit tests for QueryIntentClassifier.
"""

import pytest
from services.query_classifier import QueryIntentClassifier, INTENT_CLASSES


def test_query_classifier_intents():
    classifier = QueryIntentClassifier()

    # Test question answering
    res_qa = classifier.classify("What is the accuracy of the model on the test benchmark?")
    assert res_qa["intent"] in ["QUESTION_ANSWERING", "ANALYSIS", "SEARCH"]
    assert 0.0 <= res_qa["confidence"] <= 1.0

    # Test summarization
    res_sum = classifier.classify("Can you provide an executive summary and TLDR of this entire paper?")
    assert res_sum["intent"] == "SUMMARIZATION"
    assert res_sum["confidence"] > 0.2

    # Test comparison
    res_comp = classifier.classify("Compare the difference between transformer attention vs RNN memory.")
    assert res_comp["intent"] == "COMPARISON"

    # Test study / pedagogy
    res_study = classifier.classify("Explain this like I am 10 and give me practice quiz questions.")
    assert res_study["intent"] == "STUDY"


def test_query_classifier_fallback_on_empty():
    classifier = QueryIntentClassifier()
    res = classifier.classify("")
    assert res["intent"] == "QUESTION_ANSWERING"
    assert res["confidence"] >= 0.5
