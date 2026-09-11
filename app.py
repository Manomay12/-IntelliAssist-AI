"""
Main Application Entry Point for IntelliAssist AI — Smart Document AI Assistant.
A production-ready, academic Major Project AI document intelligence dashboard.
"""

import sys
import time
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
# pyrefly: ignore [missing-import]
import streamlit as st

# Auto-launch with streamlit run if executed directly via `python app.py`
if __name__ == "__main__" and not st.runtime.exists():
    # pyrefly: ignore [missing-import]
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())

# Streamlit Page Configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="IntelliAssist AI — Smart Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System and Components
import importlib
import components.styles
import components.sidebar
import components.chat_ui
import services.conversation_manager
import services.rag_engine
import services.llm_service
import pages_views.chat_view
import pages_views.history_view

# Development hot-reload (disabled in production for high performance)
if os.getenv("DEBUG_RELOAD") == "1":
    import importlib
    importlib.reload(components.styles)
    importlib.reload(components.sidebar)
    importlib.reload(components.chat_ui)
    importlib.reload(services.conversation_manager)
    importlib.reload(services.rag_engine)
    importlib.reload(services.llm_service)
    importlib.reload(pages_views.chat_view)
    importlib.reload(pages_views.history_view)

from components.styles import inject_custom_styles
from components.sidebar import render_sidebar
from utils.config import (
    APP_NAME, APP_TAGLINE, APP_VERSION, APP_SUBTITLE,
    DEFAULT_LLM_PROVIDER, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, UPLOAD_DIR, DATA_DIR,
    GEMINI_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY
)
from utils.sample_docs import generate_all_samples

# Services
from services.document_processor import DocumentProcessor
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.rag_engine import RAGEngine
from services.summarizer import DocumentSummarizer
from services.sentiment_analyzer import SentimentIntentAnalyzer
from services.conversation_manager import ConversationManager
from services.question_generator import QuestionGenerator
from services.doc_classifier import DocumentClassifier

# Page Views
from pages_views.dashboard_view import render_dashboard
from pages_views.documents_view import render_documents_page
from pages_views.chat_view import render_chat_page
from pages_views.search_view import render_search_page
from pages_views.summarizer_view import render_summarizer_page
from pages_views.sentiment_view import render_sentiment_page
from pages_views.analytics_view import render_analytics_page
from pages_views.history_view import render_history_page
from pages_views.settings_view import render_settings_page

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("intelliassist_ai")

# Inject CSS
inject_custom_styles()

DOCUMENTS_METADATA_FILE = DATA_DIR / "documents_metadata.json"

def save_document_registry(registry: dict):
    """Persist document metadata and extracted text to disk."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DOCUMENTS_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save document registry: %s", e)

def load_document_registry() -> dict:
    """Load document metadata from disk if available."""
    if DOCUMENTS_METADATA_FILE.exists():
        try:
            with open(DOCUMENTS_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def resolve_api_key_for_provider(provider: str, settings: dict) -> str:
    """Safely map provider name to its corresponding API key."""
    if "NVIDIA" in provider:
        return settings.get("nvidia_api_key") or os.getenv("NVIDIA_API_KEY", "")
    elif "OpenAI" in provider:
        return settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    elif "Gemini" in provider:
        return settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
    return ""

def initialize_state():
    """Initialize persistent session states and singleton services."""
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Dashboard"

    if "settings" not in st.session_state:
        if "NVIDIA" in DEFAULT_LLM_PROVIDER:
            default_model = "meta/llama-3.2-11b-vision-instruct"
        elif "OpenAI" in DEFAULT_LLM_PROVIDER:
            default_model = "gpt-4o-mini"
        else:
            default_model = "gemini-flash-latest"

        st.session_state.settings = {
            "provider": DEFAULT_LLM_PROVIDER,
            "model_name": default_model,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "gemini_api_key": GEMINI_API_KEY,
            "nvidia_api_key": NVIDIA_API_KEY,
            "openai_api_key": OPENAI_API_KEY
        }

    # Core AI singletons
    if "embedding_service" not in st.session_state:
        st.session_state.embedding_service = EmbeddingService()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(embedding_service=st.session_state.embedding_service)

    if "llm_service" not in st.session_state:
        s = st.session_state.settings
        chosen_key = resolve_api_key_for_provider(s["provider"], s)
        st.session_state.llm_service = LLMService(
            provider=s["provider"],
            model_name=s["model_name"],
            temperature=s["temperature"],
            max_tokens=s["max_tokens"],
            api_key=chosen_key
        )

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine(
            vector_store=st.session_state.vector_store,
            llm_service=st.session_state.llm_service
        )

    if "summarizer" not in st.session_state:
        st.session_state.summarizer = DocumentSummarizer(llm_service=st.session_state.llm_service)

    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = SentimentIntentAnalyzer()

    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    if "question_generator" not in st.session_state:
        st.session_state.question_generator = QuestionGenerator()

    if "doc_classifier" not in st.session_state:
        st.session_state.doc_classifier = DocumentClassifier(embedding_service=st.session_state.embedding_service)

    if "document_registry" not in st.session_state:
        st.session_state.document_registry = load_document_registry()

    # Ensure all registered documents have a valid semantic category
    if st.session_state.document_registry:
        reg_updated = False
        for fname, dinfo in st.session_state.document_registry.items():
            if not dinfo.get("category"):
                cat_eval = st.session_state.doc_classifier.classify_document(dinfo.get("full_text", ""), fname)
                dinfo["category"] = cat_eval.get("category", "General Document") if isinstance(cat_eval, dict) else str(cat_eval)
                reg_updated = True
        if reg_updated:
            save_document_registry(st.session_state.document_registry)

    # Synchronize registry documents with vector store on startup only if vector store is unpopulated
    vdb_docs = set(st.session_state.vector_store.get_all_documents())
    if len(vdb_docs) == 0 and st.session_state.document_registry:
        sync_needed = False
        for fname, dinfo in st.session_state.document_registry.items():
            if fname not in vdb_docs and dinfo.get("full_text"):
                try:
                    chk = TextChunker().chunk_document({
                        "filename": fname,
                        "pages": [{"page_number": 1, "total_pages": dinfo.get("total_pages", 1), "text": dinfo.get("full_text")}]
                    })
                    if chk:
                        st.session_state.vector_store.add_documents(chk, persist=False)
                        sync_needed = True
                except Exception:
                    pass
        if sync_needed:
            st.session_state.vector_store.save_to_disk()

    if "activity_logs" not in st.session_state:
        st.session_state.activity_logs = [
            {"icon": "Init", "title": "System Initialized", "time": "Just now", "desc": "Vector database loaded with cosine index and hybrid BM25."}
        ]

    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0

    if "total_summaries" not in st.session_state:
        st.session_state.total_summaries = 0

    if "total_searches" not in st.session_state:
        st.session_state.total_searches = 0

    if "latencies" not in st.session_state:
        st.session_state.latencies = [0.25, 0.30, 0.28]

    if "current_summary_data" not in st.session_state:
        st.session_state.current_summary_data = None

    if "chat_doc_filter" not in st.session_state:
        st.session_state.chat_doc_filter = "All Documents"

    if "selected_doc_preview" not in st.session_state:
        st.session_state.selected_doc_preview = None

    if "target_summary_doc" not in st.session_state:
        st.session_state.target_summary_doc = None

    if "target_sentiment_doc" not in st.session_state:
        st.session_state.target_sentiment_doc = None

initialize_state()

def record_activity(icon: str, title: str, desc: str = ""):
    """Add a new log item to recent activity feed."""
    st.session_state.activity_logs.insert(0, {
        "icon": icon,
        "title": title,
        "time": time.strftime("%H:%M:%S"),
        "desc": desc
    })
    if len(st.session_state.activity_logs) > 20:
        st.session_state.activity_logs.pop()

def process_and_index_files(files_or_paths):
    """Process a list of files or file paths through the single-pass batch indexing pipeline."""
    progress_bar = st.progress(0, text="Initializing processing pipeline...")
    step_container = st.empty()

    chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    total_files = len(files_or_paths)
    all_new_chunks = []

    for f_idx, file_obj in enumerate(files_or_paths):
        if isinstance(file_obj, (str, Path)):
            path = Path(file_obj)
            filename = path.name
            with open(path, "rb") as f:
                file_bytes = f.read()
            file_size = len(file_bytes)
        else:
            filename = file_obj.name
            file_bytes = file_obj.getvalue()
            file_size = len(file_bytes)

        pct = int(10 + (f_idx / max(1, total_files)) * 60)
        progress_bar.progress(pct, text=f"Processing '{filename}'...")
        step_container.markdown(f"""
        <div class="step-item step-active">Extracting & analyzing: <b>{filename}</b></div>
        """, unsafe_allow_html=True)

        file_hash = DocumentProcessor.compute_file_hash(file_bytes)
        processed_doc = DocumentProcessor.process_file(file_bytes, filename)
        if processed_doc.get("warning"):
            st.warning(processed_doc["warning"])

        full_text = processed_doc.get("full_text", "")
        # Semantic Document Classification
        cat_res = st.session_state.doc_classifier.classify_document(full_text, filename)
        category = cat_res.get("category", "General Document") if isinstance(cat_res, dict) else str(cat_res)

        chunks = chunker.chunk_document(processed_doc)
        for c in chunks:
            c["file_hash"] = file_hash
            c["category"] = category

        # Remove previous chunks for this file in-memory
        st.session_state.vector_store.delete_document(filename, persist=False)
        # Add new chunks in-memory
        st.session_state.vector_store.add_documents(chunks, persist=False)
        all_new_chunks.extend(chunks)

        # Record in registry
        st.session_state.document_registry[filename] = {
            "filename": filename,
            "file_hash": file_hash,
            "file_ext": processed_doc.get("file_ext", ".txt"),
            "file_size": file_size,
            "category": category,
            "total_pages": processed_doc.get("total_pages", 1),
            "total_chars": processed_doc.get("total_chars", 0),
            "total_words": processed_doc.get("total_words", 0),
            "chunk_count": len(chunks),
            "full_text": full_text,
            "upload_date": time.strftime("%b %d, %H:%M"),
            "status": "Ready"
        }

    # Batch save once to disk
    progress_bar.progress(90, text="Synchronizing vector index & persistence layers...")
    save_document_registry(st.session_state.document_registry)
    st.session_state.vector_store.save_to_disk()

    progress_bar.progress(100, text="Indexing complete")
    step_container.markdown(f"""
    <div class="step-item step-done"><b>{len(files_or_paths)} file(s) indexed ({len(all_new_chunks)} chunks ready)</b></div>
    """, unsafe_allow_html=True)

    record_activity("Doc", f"Indexed {len(files_or_paths)} file(s)", f"Added {len(all_new_chunks)} chunks to vector store.")
    st.toast(f"Successfully indexed {len(files_or_paths)} document(s)!")

def load_sample_documents_action():
    """Load and index the 3 bundled academic sample documents."""
    sample_files = generate_all_samples()
    with st.spinner("Loading and indexing sample documents..."):
        process_and_index_files(sample_files)
    st.rerun()

# Auto-seed sample documents if both registry and vector database are completely empty
if len(st.session_state.document_registry) == 0 and len(st.session_state.vector_store.chunks) == 0:
    sample_files = generate_all_samples()
    process_and_index_files(sample_files)

# Session actions helper callbacks
all_sessions = st.session_state.conversation_manager.list_all_sessions()
current_sid = st.session_state.conversation_manager.current_session_id

def handle_load_session(s_id: str):
    if st.session_state.conversation_manager.load_session(s_id):
        st.session_state.nav_page = "AI Chat"
        st.toast("Loaded conversation session!", icon="📂")
        st.rerun()

def handle_delete_session(s_id: str):
    st.session_state.conversation_manager.delete_session(s_id)
    st.toast("Deleted conversation session.", icon="🗑️")
    st.rerun()

def handle_clear_current_chat():
    st.session_state.conversation_manager.clear_current_chat()
    st.toast("Active chat messages cleared!", icon="🧹")
    st.rerun()

def handle_new_chat():
    st.session_state.conversation_manager.new_session()
    st.session_state.chat_doc_filter = "All Documents"
    st.session_state.nav_page = "AI Chat"
    st.toast("Started a brand new chat session!", icon="✨")
    st.rerun()

def handle_delete_all_sessions():
    cnt = st.session_state.conversation_manager.delete_all_sessions()
    st.toast(f"Deleted all {cnt} saved conversation sessions.", icon="🗑️")
    st.rerun()

# Sidebar Rendering
is_api_conn = bool(st.session_state.settings.get("gemini_api_key") or st.session_state.settings.get("nvidia_api_key") or st.session_state.settings.get("openai_api_key"))
selected_nav = render_sidebar(
    active_nav=st.session_state.nav_page,
    total_docs=len(st.session_state.document_registry),
    total_chunks=st.session_state.vector_store.total_chunks,
    provider_name=st.session_state.settings.get("provider", "Demo AI"),
    is_api_connected=is_api_conn,
    recent_sessions=all_sessions,
    current_session_id=current_sid,
    on_load_session=handle_load_session,
    on_delete_session=handle_delete_session,
    on_new_chat=handle_new_chat,
    on_clear_active_chat=handle_clear_current_chat,
    on_navigate=lambda p: (setattr(st.session_state, "nav_page", p), st.rerun())
)

if selected_nav != st.session_state.nav_page:
    st.session_state.nav_page = selected_nav
    if selected_nav == "Summarizer":
        st.session_state.target_summary_doc = None
        st.session_state.current_summary_data = None
    st.rerun()

# ----------------- Navigation Router -----------------

doc_list = sorted(list(st.session_state.document_registry.values()), key=lambda x: x["filename"])
all_doc_names = [d["filename"] for d in doc_list]

def handle_quick_ask(question: str):
    st.session_state.nav_page = "AI Chat"
    handle_chat_message(question, "All Documents")
    st.rerun()

def handle_open_doc(filename: str):
    st.session_state.selected_doc_preview = st.session_state.document_registry.get(filename)
    st.session_state.nav_page = "Documents"
    st.rerun()

def handle_summarize_doc(filename: str):
    st.session_state.target_summary_doc = filename
    st.session_state.nav_page = "Summarizer"
    st.rerun()

def handle_chat_doc(filename: str):
    st.session_state.chat_doc_filter = filename
    st.session_state.nav_page = "AI Chat"
    st.rerun()

def handle_delete_doc(filename: str):
    st.session_state.vector_store.delete_document(filename)
    if filename in st.session_state.document_registry:
        del st.session_state.document_registry[filename]
        save_document_registry(st.session_state.document_registry)
    record_activity("Delete", f"Deleted {filename}", "Removed chunks from vector store.")
    st.toast(f"Deleted {filename} and updated vector index.")
    st.rerun()

def handle_chat_message(prompt: str, doc_filter: str):
    """Process a user question through the RAG pipeline."""
    # 1. Add user message
    st.session_state.conversation_manager.add_message(
        role="user",
        content=prompt
    )
    
    # 2. Query RAG Engine
    with st.spinner("Retrieving document context and generating answer..."):
        filter_param = None if doc_filter == "All Documents" else doc_filter
        rag_res = st.session_state.rag_engine.answer_question(
            question=prompt,
            top_k=6,
            filter_doc=filter_param
        )

    # 3. Add AI answer
    st.session_state.conversation_manager.add_message(
        role="assistant",
        content=rag_res.get("answer", ""),
        sources=rag_res.get("sources", []),
        model_info={
            "provider": rag_res.get("provider", "Demo AI"),
            "model": rag_res.get("model", ""),
            "latency_sec": rag_res.get("latency_sec", 0.35),
            "is_demo": rag_res.get("is_demo", False)
        }
    )

    # Telemetry
    st.session_state.total_questions += 1
    short_q = (prompt[:22] + "...") if len(prompt) > 25 else prompt
    record_activity("Chat", f"Q&A: {short_q}", f"Retrieved {len(rag_res.get('sources', []))} passages ({rag_res.get('latency_sec', 0.35):.2f}s)")

def upload_and_route_to_chat(files):
    """Index newly uploaded files and set active chat scope to the uploaded document."""
    process_and_index_files(files)
    if files:
        first_file = files[0]
        fname = first_file.name if hasattr(first_file, "name") else str(first_file)
        st.session_state.chat_doc_filter = fname
    st.rerun()

# Page: Dashboard
if st.session_state.nav_page == "Dashboard":
    render_dashboard(
        doc_infos=doc_list,
        total_chunks=st.session_state.vector_store.total_chunks,
        total_questions=st.session_state.total_questions,
        total_convs=len(all_sessions),
        recent_activities=st.session_state.activity_logs,
        on_quick_ask=handle_quick_ask,
        on_navigate=lambda p: (setattr(st.session_state, "nav_page", p), st.rerun()),
        on_open_doc=handle_open_doc,
        on_summarize_doc=handle_summarize_doc,
        on_chat_doc=handle_chat_doc,
        on_delete_doc=handle_delete_doc,
        on_load_samples=load_sample_documents_action,
        on_upload_files=lambda files: (process_and_index_files(files), st.rerun())
    )

# Page: Documents
elif st.session_state.nav_page == "Documents":
    render_documents_page(
        doc_infos=doc_list,
        on_upload_files=lambda files: (process_and_index_files(files), st.rerun()),
        on_load_samples=load_sample_documents_action,
        on_open_doc=handle_open_doc,
        on_summarize_doc=handle_summarize_doc,
        on_chat_doc=handle_chat_doc,
        on_delete_doc=handle_delete_doc,
        selected_doc_preview=st.session_state.selected_doc_preview
    )

# Page: AI Chat
elif st.session_state.nav_page == "AI Chat":
    render_chat_page(
        messages=st.session_state.conversation_manager.messages,
        all_documents=all_doc_names,
        selected_doc_filter=st.session_state.get("chat_doc_filter", "All Documents"),
        on_send_message=lambda q, df: (handle_chat_message(q, df), st.rerun()),
        on_feedback=lambda mid, fb: st.session_state.conversation_manager.set_feedback(mid, fb),
        on_new_session=handle_new_chat,
        on_clear_chat=handle_clear_current_chat,
        on_export_markdown=lambda: st.session_state.conversation_manager.export_as_markdown(),
        session_title=st.session_state.conversation_manager.title,
        saved_sessions=all_sessions,
        current_session_id=current_sid,
        on_load_session=handle_load_session,
        on_delete_session=handle_delete_session,
        doc_registry=st.session_state.document_registry,
        on_upload_files=upload_and_route_to_chat,
        question_generator=st.session_state.question_generator
    )

# Page: Semantic Search
elif st.session_state.nav_page == "Semantic Search":
    def do_semantic_search(query: str, top_k: int, threshold: float, doc_filter: str):
        st.session_state.total_searches += 1
        short_q = (query[:22] + "...") if len(query) > 25 else query
        record_activity("Search", f"Search: {short_q}", f"Top-{top_k} matches retrieved")
        return st.session_state.vector_store.search(
            query=query,
            top_k=top_k,
            threshold=threshold,
            filter_doc=None if doc_filter == "All Documents" else doc_filter
        )

    render_search_page(
        all_documents=all_doc_names,
        on_search=do_semantic_search,
        on_ask_about_result=lambda q: (setattr(st.session_state, "nav_page", "AI Chat"), handle_chat_message(q, "All Documents"), st.rerun()),
        doc_registry=st.session_state.document_registry,
        question_generator=st.session_state.question_generator
    )

# Page: Summarizer
elif st.session_state.nav_page == "Summarizer":
    def do_summarize(doc_name: str, mode: str):
        doc_info = st.session_state.document_registry.get(doc_name, {})
        full_text = doc_info.get("full_text", "")
        summary_result = st.session_state.summarizer.summarize(text=full_text, mode=mode, doc_name=doc_name)
        st.session_state.current_summary_data = summary_result
        st.session_state.total_summaries += 1
        short_d = (doc_name[:22] + "...") if len(doc_name) > 25 else doc_name
        record_activity("Summary", f"Summary: {short_d}", f"Mode: {mode}")
        return summary_result

    def do_compare(doc_a_name: str, doc_b_name: str):
        doc_a_info = st.session_state.document_registry.get(doc_a_name, {})
        doc_b_info = st.session_state.document_registry.get(doc_b_name, {})
        text_a = doc_a_info.get("full_text", "")
        text_b = doc_b_info.get("full_text", "")
        record_activity("Compare", f"Compare: {doc_a_name[:12]} vs {doc_b_name[:12]}", "Cross-document matrix generated")
        return st.session_state.summarizer.compare_documents(doc_a_name, text_a, doc_b_name, text_b)

    render_summarizer_page(
        all_documents=all_doc_names,
        on_summarize=do_summarize,
        on_compare=do_compare,
        current_summary_data=st.session_state.current_summary_data,
        default_doc=st.session_state.target_summary_doc,
        doc_registry=st.session_state.document_registry,
        summarizer_service=st.session_state.summarizer
    )

# Page: 🧠 Sentiment & Intent
elif st.session_state.nav_page == "Sentiment & Intent":
    def analyze_doc_sentiment(doc_name: str):
        doc_info = st.session_state.document_registry.get(doc_name, {})
        full_text = doc_info.get("full_text", "")
        chunks = st.session_state.vector_store.get_document_chunks(doc_name)
        
        sent = st.session_state.sentiment_analyzer.analyze_sentiment(full_text)
        intent = st.session_state.sentiment_analyzer.analyze_intent(full_text)
        trends = st.session_state.sentiment_analyzer.analyze_chunk_trends(chunks)
        short_d = (doc_name[:22] + "...") if len(doc_name) > 25 else doc_name
        record_activity("Tone", f"Tone: {short_d}", f"{sent['label']} ({sent['confidence']}%)")
        return {"sentiment": sent, "intent": intent, "trends": trends}

    def analyze_custom(text: str):
        sent = st.session_state.sentiment_analyzer.analyze_sentiment(text)
        intent = st.session_state.sentiment_analyzer.analyze_intent(text)
        return {"sentiment": sent, "intent": intent}

    render_sentiment_page(
        all_documents=all_doc_names,
        on_analyze_document=analyze_doc_sentiment,
        on_analyze_custom_text=analyze_custom,
        default_doc=st.session_state.target_sentiment_doc
    )

# Page: Analytics
elif st.session_state.nav_page == "Analytics":
    avg_lat = sum(st.session_state.latencies) / max(1, len(st.session_state.latencies))
    render_analytics_page(
        doc_infos=doc_list,
        total_chunks=st.session_state.vector_store.total_chunks,
        total_questions=st.session_state.total_questions,
        total_summaries=st.session_state.total_summaries,
        total_searches=st.session_state.total_searches,
        avg_latency=avg_lat
    )

# Page: History
elif st.session_state.nav_page == "History":
    def handle_rename_session(s_id: str, new_name: str):
        st.session_state.conversation_manager.rename_session(s_id, new_name)
        st.toast("Renamed conversation.")

    render_history_page(
        sessions=all_sessions,
        current_session_id=current_sid,
        on_load_session=handle_load_session,
        on_rename_session=handle_rename_session,
        on_delete_session=handle_delete_session,
        on_delete_all_sessions=handle_delete_all_sessions,
        on_new_chat=handle_new_chat
    )

# Page: Settings
elif st.session_state.nav_page == "Settings":
    def handle_save_settings(new_settings: Dict[str, Any]):
        st.session_state.settings.update(new_settings)
        chosen_key = resolve_api_key_for_provider(new_settings["provider"], new_settings)
        st.session_state.llm_service = LLMService(
            provider=new_settings["provider"],
            model_name=new_settings["model_name"],
            temperature=new_settings["temperature"],
            max_tokens=new_settings["max_tokens"],
            api_key=chosen_key
        )
        st.session_state.rag_engine.llm_service = st.session_state.llm_service
        st.session_state.summarizer.llm_service = st.session_state.llm_service

    def handle_clear_history():
        st.session_state.conversation_manager.new_session()
        for p in st.session_state.conversation_manager.storage_dir.glob("session_*.json"):
            try:
                p.unlink()
            except Exception:
                pass

    def handle_rebuild_index():
        st.session_state.vector_store.clear()
        for doc_info in st.session_state.document_registry.values():
            chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
            processed_doc = {
                "filename": doc_info["filename"],
                "pages": [{"page_number": 1, "text": doc_info["full_text"], "total_pages": doc_info.get("total_pages", 1)}]
            }
            chunks = chunker.chunk_document(processed_doc)
            st.session_state.vector_store.add_documents(chunks)

    def handle_clear_all():
        st.session_state.vector_store.clear()
        st.session_state.document_registry = {}
        if DOCUMENTS_METADATA_FILE.exists():
            DOCUMENTS_METADATA_FILE.unlink()
        handle_clear_history()

    render_settings_page(
        current_settings=st.session_state.settings,
        on_save_settings=handle_save_settings,
        on_clear_history=handle_clear_history,
        on_rebuild_index=handle_rebuild_index,
        on_clear_all_data=handle_clear_all
    )

# Footer
st.markdown("""
<div class="app-footer">
    <b>IntelliAssist AI</b> • Smart Document Intelligence Platform<br>
    Built with Python • NLP • RAG • Embeddings • Vector Search • Generative AI
</div>
""", unsafe_allow_html=True)
