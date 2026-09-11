"""
Unit tests for DocumentClassifier.
"""

import pytest
from services.doc_classifier import DocumentClassifier, DOC_CLASSES


def test_classify_document_genres():
    classifier = DocumentClassifier()

    research_text = (
        "Abstract: In this paper we introduce an attention mechanism for neural machine translation. "
        "We evaluate on the WMT 2014 benchmark and compare with state-of-the-art baselines. "
        "Our methodology yields a 2.0 BLEU score improvement. References and related work are discussed."
    )
    cat_research = classifier.classify_document(research_text, "attention_paper.pdf")
    assert cat_research["category"] in ["Research Paper", "Academic Material", "Technical Document"]
    assert 0.0 <= cat_research["confidence"] <= 1.0

    financial_text = (
        "Consolidated Financial Statements for Fiscal Year 2025. Total revenue reached $4.2B, representing "
        "a 14% year-over-year growth. Operating margins improved by 230 basis points. EBITDA and balance sheet analysis."
    )
    cat_finance = classifier.classify_document(financial_text, "annual_report_2025.pdf")
    assert cat_finance["category"] in ["Annual Report", "Business Report"]

    policy_text = (
        "Enterprise Data Governance and Privacy Policy. All employees and contractors must adhere to strict "
        "confidentiality and GDPR regulatory compliance standards. Non-compliance results in termination."
    )
    cat_policy = classifier.classify_document(policy_text, "security_policy.pdf")
    assert cat_policy["category"] in ["Policy Document", "Technical Document"]


def test_classify_empty_document():
    classifier = DocumentClassifier()
    res = classifier.classify_document("", "untitled.txt")
    assert res["category"] == "General Document"
