"""
Verification test for strict per-document targeting, multi-document synthesis, and query auto-routing.
"""

import sys
from pathlib import Path

# Setup path and encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.rag_engine import RAGEngine

def main():
    print("=" * 60)
    print("[*] Testing Multi-Document Scoping & Strict Targeting")
    print("=" * 60)

    vdb = VectorStore()
    llm = LLMService(provider="Demo Mode (Smart AI)")
    rag = RAGEngine(vector_store=vdb, llm_service=llm)

    docs = vdb.get_all_documents()
    print(f"[*] Available Documents in Vector DB ({len(docs)}):")
    for d in docs:
        print(f"  - {d}")

    assert len(docs) >= 3, "Expected at least 3 sample documents in Vector DB."

    # Test 1: Strict Target to Transformers PDF
    doc_1 = "AI_Research_Paper_Transformers_and_RAG.pdf"
    print(f"\n[Test 1] Querying with target scope: {doc_1}...")
    res_1 = rag.answer_question("Summarize the main findings", filter_doc=doc_1)
    sources_1 = [s["filename"] for s in res_1.get("sources", [])]
    print(f"  + Sources: {set(sources_1)}")
    print(f"  + Answer Header: {res_1['answer'].splitlines()[0] if res_1['answer'].splitlines() else ''}")
    assert all(s == doc_1 for s in sources_1), f"Expected all sources to be {doc_1}, got {sources_1}"
    assert doc_1 in res_1["answer"], f"Expected {doc_1} mentioned in answer title."

    # Test 2: Strict Target to Project Report DOCX
    doc_2 = "Project_Report_IntelliAssist_AI.docx"
    print(f"\n[Test 2] Querying with target scope: {doc_2}...")
    res_2 = rag.answer_question("What are the key technical concepts and methodology?", filter_doc=doc_2)
    sources_2 = [s["filename"] for s in res_2.get("sources", [])]
    print(f"  + Sources: {set(sources_2)}")
    print(f"  + Answer Header: {res_2['answer'].splitlines()[0] if res_2['answer'].splitlines() else ''}")
    assert all(s == doc_2 for s in sources_2), f"Expected all sources to be {doc_2}, got {sources_2}"
    assert doc_2 in res_2["answer"], f"Expected {doc_2} mentioned in answer title."

    # Test 3: Strict Target to Healthcare TXT
    doc_3 = "Machine_Learning_Healthcare_Overview.txt"
    print(f"\n[Test 3] Querying with target scope: {doc_3}...")
    res_3 = rag.answer_question("Explain the empirical benchmarks and evidence", filter_doc=doc_3)
    sources_3 = [s["filename"] for s in res_3.get("sources", [])]
    print(f"  + Sources: {set(sources_3)}")
    print(f"  + Answer Header: {res_3['answer'].splitlines()[0] if res_3['answer'].splitlines() else ''}")
    assert all(s == doc_3 for s in sources_3), f"Expected all sources to be {doc_3}, got {sources_3}"
    assert doc_3 in res_3["answer"], f"Expected {doc_3} mentioned in answer title."

    # Test 4: Strict Target to User Uploaded Report PDF
    if "2026-08-19_report_4_9478B.pdf" in docs:
        doc_4 = "2026-08-19_report_4_9478B.pdf"
        print(f"\n[Test 4] Querying with target scope: {doc_4}...")
        res_4 = rag.answer_question("tell me about what is in this pdf", filter_doc=doc_4)
        sources_4 = [s["filename"] for s in res_4.get("sources", [])]
        print(f"  + Sources: {set(sources_4)}")
        print(f"  + Answer Header: {res_4['answer'].splitlines()[0] if res_4['answer'].splitlines() else ''}")
        assert all(s == doc_4 for s in sources_4), f"Expected all sources to be {doc_4}, got {sources_4}"
        assert doc_4 in res_4["answer"], f"Expected {doc_4} mentioned in answer title."

    # Test 5: Auto document mention detection without filter_doc
    print(f"\n[Test 5] Query mentioning 'healthcare' without explicit filter...")
    res_5 = rag.answer_question("What are the findings in the healthcare document?")
    sources_5 = [s["filename"] for s in res_5.get("sources", [])]
    print(f"  + Detected Sources: {set(sources_5)}")
    assert any("Healthcare" in s for s in sources_5), f"Expected Healthcare sources, got {sources_5}"

    # Test 6: Multi-Document Scope (All Documents)
    print(f"\n[Test 6] Querying across All Documents (Multi-Doc Synthesis)...")
    res_6 = rag.answer_question("Compare all documents and summarize their findings", filter_doc=None)
    sources_6 = set(s["filename"] for s in res_6.get("sources", []))
    print(f"  + Multi-Doc Sources Covered: {sources_6}")
    print(f"  + Answer Header: {res_6['answer'].splitlines()[0] if res_6['answer'].splitlines() else ''}")
    assert len(sources_6) > 1, f"Expected multiple document sources in multi-doc scope, got {sources_6}"

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL DOCUMENT TARGETING TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
