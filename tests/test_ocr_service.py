"""
Unit tests for OCRService.
"""

import pytest
from services.ocr_service import OCRService


def test_ocr_service_scanned_detection():
    # Regular text should not trigger scanned detection
    assert not OCRService.is_scanned_pdf("This document has several hundred characters of readable extracted text content from PDF digital streams.")

    # Scanned image with very little text (< 20 chars) should trigger detection
    assert OCRService.is_scanned_pdf("   \n\t  ")
    assert OCRService.is_scanned_pdf("Page 1")


def test_ocr_fallback_processing():
    ocr = OCRService()
    res = ocr.process_scanned_pdf(b"dummy_bytes", "scanned_invoice.pdf")
    assert res is not None
    assert "pages" in res
    assert "full_text" in res
    assert "Scanned document detected" in res.get("warning", "") or "OCR fallback" in res.get("warning", "")
