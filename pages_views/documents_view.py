"""
Documents Page View for IntelliAssist AI.
Manages multi-format document ingestion, real-time extraction pipeline,
document classification profiling, and text inspection.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st
from components.document_card import render_document_card
from utils.helpers import format_file_size, get_file_icon

def render_documents_page(
    doc_infos: Optional[List[Dict[str, Any]]] = None,
    on_upload_files: Optional[Callable[[List[Any]], None]] = None,
    on_load_samples: Optional[Callable[[], None]] = None,
    on_open_doc: Optional[Callable[[str], None]] = None,
    on_summarize_doc: Optional[Callable[[str], None]] = None,
    on_chat_doc: Optional[Callable[[str], None]] = None,
    on_delete_doc: Optional[Callable[[str], None]] = None,
    selected_doc_preview: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs
):
    """Render the Document Library and ingestion interface."""
    doc_infos = doc_infos if doc_infos is not None else kwargs.get("doc_infos", [])
    on_upload_files = on_upload_files or kwargs.get("on_upload_files", lambda f: None)
    on_load_samples = on_load_samples or kwargs.get("on_load_samples", lambda: None)
    on_open_doc = on_open_doc or kwargs.get("on_open_doc", lambda d: None)
    on_summarize_doc = on_summarize_doc or kwargs.get("on_summarize_doc", lambda d: None)
    on_chat_doc = on_chat_doc or kwargs.get("on_chat_doc", lambda d: None)
    on_delete_doc = on_delete_doc or kwargs.get("on_delete_doc", lambda d: None)
    st.markdown("""
    <div style="margin-bottom:22px;">
        <h2 style="font-weight:800; color:#ffffff; margin:0 0 6px 0; letter-spacing:-0.03em; font-size:1.8rem;">
            Document Management & Ingestion
        </h2>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0;">
            Ingest PDF, DOCX, or TXT documents into the vectorized knowledge base. Text is cleaned, segmented into semantic chunks, and embedded into 384-dimensional dense vectors alongside a lexical BM25 index.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Dropzone
    st.markdown("""
    <div class="modern-card" style="border: 2px dashed rgba(56, 189, 248, 0.35); text-align:center; padding:32px 24px;">
        <h3 style="margin:0 0 6px 0; font-size:1.2rem; font-weight:700; color:#ffffff;">Upload and Index Documents</h3>
        <p style="color:#94a3b8; font-size:0.88rem; margin-bottom:18px;">
            Drop PDF, DOCX, or TXT files here to index into the RAG semantic memory.
        </p>
        <div style="display:inline-flex; gap:12px; font-size:0.78rem; font-weight:600; color:#38bdf8; background:rgba(6,182,212,0.1); padding:6px 18px; border-radius:9999px; border:1px solid rgba(6,182,212,0.25);">
            <span>PDF</span> • <span>DOCX</span> • <span>TXT</span> • <span>MD</span> • <span>Max 25MB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="doc_file_uploader"
    )

    col_proc, col_samp = st.columns([3, 2])
    with col_proc:
        if uploaded_files:
            if st.button(f"Process & Index {len(uploaded_files)} File(s)", key="btn_process_uploads", type="primary", use_container_width=True):
                on_upload_files(uploaded_files)

    with col_samp:
        if st.button("Load Academic Benchmark Samples (3 Papers)", key="btn_load_samples_doc_page", use_container_width=True):
            on_load_samples()

    # Document Preview Panel if requested
    if selected_doc_preview:
        with st.expander(f"Viewing Document: {selected_doc_preview.get('filename')}", expanded=True):
            st.markdown(f"""
            <div style="display:flex; flex-wrap:wrap; gap:14px; margin-bottom:14px; font-size:0.82rem; color:#94a3b8;">
                <span style="background:rgba(6,182,212,0.12); color:#38bdf8; padding:3px 10px; border-radius:6px; font-weight:600;">Category: {selected_doc_preview.get('category', 'General Document')}</span>
                <span style="background:rgba(255,255,255,0.04); padding:3px 10px; border-radius:6px;">Total Pages: <b>{selected_doc_preview.get('total_pages', 1)}</b></span>
                <span style="background:rgba(255,255,255,0.04); padding:3px 10px; border-radius:6px;">Total Chunks: <b>{selected_doc_preview.get('chunk_count', 0)}</b></span>
                <span style="background:rgba(255,255,255,0.04); padding:3px 10px; border-radius:6px;">Words: <b>{selected_doc_preview.get('total_words', 0):,}</b></span>
                <span style="background:rgba(255,255,255,0.04); padding:3px 10px; border-radius:6px;">Characters: <b>{selected_doc_preview.get('total_chars', 0):,}</b></span>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area(
                "Document Extracted Text",
                selected_doc_preview.get("full_text", "No text content available."),
                height=300,
                disabled=True
            )

    st.markdown("<div style='margin:28px 0;'></div>", unsafe_allow_html=True)

    # Document Library Section
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <h3 style="margin:0; font-size:1.2rem; font-weight:700; color:#f8fafc;">
            Indexed Document Library ({len(doc_infos)} files)
        </h3>
    </div>
    """, unsafe_allow_html=True)

    if not doc_infos:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:36px 20px;">
            <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:6px;">Your document library is empty</div>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:420px; margin:0 auto 16px auto;">
                Upload documents above or click 'Load Academic Benchmark Samples' to populate sample research papers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for doc in doc_infos:
            render_document_card(
                doc,
                on_open=on_open_doc,
                on_summarize=on_summarize_doc,
                on_chat=on_chat_doc,
                on_delete=on_delete_doc,
                key_prefix="lib"
            )
