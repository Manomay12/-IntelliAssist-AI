"""
Comprehensive test script for IntelliAssist AI services.
Tests document processing, chunking, embeddings, vector store, RAG answering,
summarizer, sentiment analyzer, and conversation persistence.
"""

import sys
import os
from pathlib import Path

# Force UTF-8 on Windows stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils.sample_docs import generate_all_samples
from services.document_processor import DocumentProcessor
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.rag_engine import RAGEngine
from services.summarizer import DocumentSummarizer
from services.sentiment_analyzer import SentimentIntentAnalyzer
from services.conversation_manager import ConversationManager

def run_all_tests():
    print("==================================================")
    print("[*] Running IntelliAssist AI Verification Test Suite")
    print("==================================================")

    # 1. Test sample docs generation
    print("\n[1/8] Generating sample academic documents...")
    sample_files = generate_all_samples()
    for sf in sample_files:
        assert sf.exists(), f"Sample file {sf} was not created."
        print(f"  + Created: {sf.name} ({sf.stat().st_size} bytes)")

    # 2. Test document extraction & cleaning
    print("\n[2/8] Testing DocumentProcessor on PDF, DOCX, TXT...")
    all_chunks = []
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)

    for sf in sample_files:
        with open(sf, "rb") as f:
            bytes_data = f.read()
        doc_info = DocumentProcessor.process_file(bytes_data, sf.name)
        print(f"  + Processed '{sf.name}': {doc_info['total_pages']} pages, {doc_info['total_chars']} chars, {doc_info['total_words']} words.")
        
        chunks = chunker.chunk_document(doc_info)
        all_chunks.extend(chunks)
        print(f"  + Generated {len(chunks)} chunks for '{sf.name}'.")

    assert len(all_chunks) > 0, "No chunks were generated."

    # 3. Test Embedding Service
    print("\n[3/8] Testing EmbeddingService...")
    embed_service = EmbeddingService()
    test_texts = ["What is the attention mechanism?", "Clinical decision support in healthcare"]
    vectors = embed_service.embed_texts(test_texts)
    assert vectors.shape == (2, 384), f"Unexpected embedding shape: {vectors.shape}"
    print(f"  + Successfully generated normalized vectors with shape: {vectors.shape}")

    # 4. Test Vector Database & Cosine Index
    print("\n[4/8] Testing VectorStore indexing & similarity search...")
    vdb = VectorStore(embedding_service=embed_service)
    vdb.clear()
    indexed_count = vdb.add_documents(all_chunks)
    assert indexed_count == len(all_chunks), "Chunk indexing count mismatch."
    print(f"  + Indexed {indexed_count} chunks into Vector DB.")

    search_res = vdb.search(query="attention mechanism and transformer architecture", top_k=3)
    assert len(search_res) > 0, "Vector search returned no results."
    print(f"  + Vector search returned {len(search_res)} top chunks. Top score: {search_res[0]['similarity_percentage']}% from {search_res[0]['filename']}")

    # 5. Test LLM Service & RAG Engine (with forgiving improper query handling)
    print("\n[5/8] Testing RAG Engine & Forgiving Query Handling...")
    llm_service = LLMService(provider="Demo Mode (Smart AI)")
    rag = RAGEngine(vector_store=vdb, llm_service=llm_service)
    
    # 5a. Standard query
    rag_ans = rag.answer_question("Explain how the attention mechanism works in transformers.")
    assert len(rag_ans["answer"]) > 50, "RAG answer is too short."
    assert len(rag_ans["sources"]) > 0, "RAG answer has no source citations."

    # 5b. Informal / Typo query ("wat dis pdf say abt transfmr")
    typo_ans = rag.answer_question("wat dis pdf say abt transfmr")
    assert len(typo_ans["answer"]) > 50, "Failed to answer informal/typo query."
    assert len(typo_ans["sources"]) > 0, "No sources retrieved for informal query."

    # 5c. Greeting query ("hi, help me understand")
    greeting_ans = rag.answer_question("hi")
    assert "Hello" in greeting_ans["answer"] or "Welcome" in greeting_ans["answer"]

    print(f"  + RAG Answer generated ({rag_ans['latency_sec']}s, {len(rag_ans['sources'])} sources cited).")
    print("  + Handled greetings and typo-ridden informal queries seamlessly.")

    # 6. Test Document Summarizer
    print("\n[6/8] Testing DocumentSummarizer...")
    summarizer = DocumentSummarizer(llm_service=llm_service)
    sample_text = all_chunks[0]["text"] + "\n\n" + all_chunks[1]["text"]
    sum_res = summarizer.summarize(sample_text, mode="Executive Summary", doc_name="Test_Doc.pdf")
    assert len(sum_res["summary"]) > 20, "Summary output is empty."
    print(f"  + Summarizer generated Executive Summary with {len(sum_res['topics'])} topics and {len(sum_res['takeaways'])} takeaways.")

    # 7. Test Sentiment & Intent Analyzer
    print("\n[7/8] Testing SentimentIntentAnalyzer...")
    sentiment_analyzer = SentimentIntentAnalyzer()
    sent_res = sentiment_analyzer.analyze_sentiment("The model achieved outstanding accuracy with exceptional performance.")
    assert sent_res["label"] == "Positive", f"Expected Positive sentiment, got {sent_res['label']}"
    
    intent_res = sentiment_analyzer.analyze_intent("Can you please summarize the key findings of this report?")
    assert intent_res["primary_intent"] in ["Question", "Request"], f"Unexpected intent: {intent_res['primary_intent']}"
    print(f"  + Sentiment analysis: {sent_res['label']} ({sent_res['confidence']}%) | Primary Intent: {intent_res['primary_intent']}")

    # 8. Test Conversation Manager
    print("\n[8/8] Testing ConversationManager persistence, clear chat, and deletion...")
    conv_mgr = ConversationManager()
    s_id = conv_mgr.new_session("Evaluation Test Session")
    conv_mgr.add_message("user", "Hello IntelliAssist!")
    conv_mgr.add_message("assistant", "Hello! How can I help with your documents today?")
    assert len(conv_mgr.messages) == 2
    md_export = conv_mgr.export_as_markdown()
    assert "# Evaluation Test Session" in md_export
    
    # Test clear current chat
    conv_mgr.clear_current_chat()
    assert len(conv_mgr.messages) == 0

    # Test delete session
    conv_mgr.new_session("Temp Session For Deletion")
    conv_mgr.add_message("user", "Temp message")
    conv_mgr.save_session()
    del_res = conv_mgr.delete_session(conv_mgr.current_session_id)
    assert del_res is True

    print("  + Conversation session saved, exported, cleared, and deleted successfully.")

    print("\n==================================================")
    print("[SUCCESS] ALL 8 TESTS PASSED! 100% OPERATIONAL")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
