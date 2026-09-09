"""
Unit tests for SentimentIntentAnalyzer.
Tests sentiment polarity classification, user intent detection, and chunk tone trajectory.
"""

# pyrefly: ignore [missing-import]
import pytest
from services.sentiment_analyzer import SentimentIntentAnalyzer

def test_positive_sentiment(sentiment_analyzer):
    """Test positive tone classification."""
    text = "The proposed algorithm achieved state-of-the-art results with remarkable precision and efficiency."
    res = sentiment_analyzer.analyze_sentiment(text)
    assert res["label"] == "Positive"
    assert res["score"] > 0
    assert res["confidence"] >= 50

def test_negative_sentiment(sentiment_analyzer):
    """Test negative / critical tone classification."""
    text = "The model suffered from catastrophic failure, high latency, poor convergence, and frequent errors."
    res = sentiment_analyzer.analyze_sentiment(text)
    assert res["label"] == "Negative"
    assert res["score"] < 0

def test_intent_detection(sentiment_analyzer):
    """Test multi-class intent categorization."""
    question_query = "What are the empirical benchmarks on the GLUE dataset?"
    intent_q = sentiment_analyzer.analyze_intent(question_query)
    assert intent_q["primary_intent"] in ["Question", "Informational"]

    request_query = "Please generate an executive brief for the leadership team."
    intent_r = sentiment_analyzer.analyze_intent(request_query)
    assert intent_r["primary_intent"] in ["Request", "Informational"]

def test_chunk_trends(sentiment_analyzer):
    """Test chunk-by-chunk trajectory analysis."""
    chunks = [
        {"chunk_id": "c1", "text": "Great introductory overview with strong results."},
        {"chunk_id": "c2", "text": "However, limitations and latency bottlenecks were observed."},
        {"chunk_id": "c3", "text": "Overall, the system provides significant advancements."}
    ]
    trends = sentiment_analyzer.analyze_chunk_trends(chunks)
    assert len(trends) == 3
    assert all("chunk_id" in t for t in trends)
    assert all("score" in t for t in trends)
