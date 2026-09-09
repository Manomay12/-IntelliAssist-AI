"""
Unit tests for DocumentProcessor service.
Tests text extraction, clean text normalization, SHA-256 hashing, corrupt file and empty file handling.
"""

# pyrefly: ignore [missing-import]
import pytest
from services.document_processor import DocumentProcessor

def test_compute_file_hash():
    """Test deterministic SHA-256 hash generation."""
    data1 = b"IntelliAssist AI Document Content"
    data2 = b"IntelliAssist AI Document Content"
    data3 = b"Different Document Content"

    hash1 = DocumentProcessor.compute_file_hash(data1)
    hash2 = DocumentProcessor.compute_file_hash(data2)
    hash3 = DocumentProcessor.compute_file_hash(data3)

    assert len(hash1) == 64
    assert hash1 == hash2
    assert hash1 != hash3

def test_clean_text():
    """Test text normalization, whitespace collapsing, and non-printable stripping."""
    raw = "  Hello \r\n\r\n World! \u00a0\u200b  This  is   a   test. \x00\x01\x02 "
    cleaned = DocumentProcessor.clean_text(raw)
    assert "Hello" in cleaned
    assert "World!" in cleaned
    assert "  " not in cleaned
    assert "\x00" not in cleaned
    assert cleaned.startswith("Hello")

def test_process_text_file():
    """Test processing of standard text files."""
    text_content = "IntelliAssist AI is a major project for document intelligence.\nIt uses vector embeddings and RAG."
    bytes_data = text_content.encode("utf-8")
    
    doc_info = DocumentProcessor.process_file(bytes_data, "research_notes.txt")
    
    assert doc_info["filename"] == "research_notes.txt"
    assert doc_info["file_ext"] == ".txt"
    assert doc_info["file_size"] == len(bytes_data)
    assert doc_info["total_words"] > 5
    assert len(doc_info["pages"]) >= 1
    assert "IntelliAssist AI" in doc_info["full_text"]
    assert doc_info["is_valid"] is True

def test_process_empty_file():
    """Test handling of 0-byte empty files."""
    doc_info = DocumentProcessor.process_file(b"", "empty_doc.txt")
    assert doc_info["file_size"] == 0
    assert doc_info["warning"] is not None

def test_process_corrupt_docx():
    """Test graceful handling of corrupt DOCX binaries."""
    corrupt_bytes = b"PK\x03\x04NOT_A_VALID_DOCX_STREAM_CORRUPTED"
    doc_info = DocumentProcessor.process_file(corrupt_bytes, "corrupted.docx")
    assert doc_info["filename"] == "corrupted.docx"
    assert doc_info["warning"] is not None

def test_process_sample_documents(sample_documents):
    """Test processing of real PDF, DOCX, and TXT files."""
    for sf in sample_documents:
        with open(sf, "rb") as f:
            bytes_data = f.read()
        doc_info = DocumentProcessor.process_file(bytes_data, sf.name)
        assert doc_info["total_chars"] > 100
        assert doc_info["total_words"] > 20
        assert doc_info["total_pages"] >= 1
        assert len(doc_info["file_hash"]) == 64
