"""
Documents Page View for IntelliAssist AI.
Manages multi-file uploads, 7-step animated indexing pipeline, document library, and text inspection.
"""

from typing import Dict, Any, List, Callable, Optional
import time
import streamlit as st
from components.document_card import render_document_card
from utils.helpers import format_file_size, get_file_icon

def render_documents_page(
    doc_infos: List[Dict[str, Any]],
    on_upload_files: Callable[[List[Any]], None],
    on_load_samples: Callable[[], None],
    on_open_doc: Callable[[str], None],
    on_summarize_doc: Callable[[str], None],
    on_chat_doc: Callable[[str], None],
    on_delete_doc: Callable[[str], None],
    selected_doc_preview: Optional[Dict[str, Any]] = None
):
    """Render the Documents management and upload workspace."""
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-weight:800; color:#ffffff; margin:0 0 6px 0; letter-spacing:-0.02em;">📄 Document Management & Ingestion</h2>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0;">
            Upload PDF, DOCX, or TXT documents. Our automated pipeline extracts text, cleans artifacts, chunks content, and indexes semantic vectors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Central Upload Card
    st.markdown("""
    <div class="modern-card" style="border: 2px dashed rgba(99, 102, 241, 0.4); text-align:center; padding:28px 24px;">
        <div style="font-size:2.4rem; margin-bottom:8px;">📤</div>
        <h3 style="margin:0 0 4px 0; font-size:1.25rem; font-weight:700; color:#ffffff;">Upload your documents</h3>
        <p style="color:#94a3b8; font-size:0.88rem; margin-bottom:16px;">
            Drop PDF, DOCX, or TXT files here and let IntelliAssist understand them.
        </p>
        <div style="display:inline-flex; gap:12px; font-size:0.78rem; font-weight:600; color:#818cf8; background:rgba(99,102,241,0.1); padding:6px 16px; border-radius:9999px; border:1px solid rgba(99,102,241,0.25);">
            <span>📕 PDF</span> • <span>📘 DOCX</span> • <span>📄 TXT</span> • <span>Max 25MB</span>
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
            if st.button(f"⚡ Process & Index {len(uploaded_files)} File(s)", key="btn_process_uploads", type="primary", use_container_width=True):
                on_upload_files(uploaded_files)

    with col_samp:
        if st.button("📦 Load Instant Academic Samples (3 Files)", key="btn_load_samples_doc_page", use_container_width=True):
            on_load_samples()

    # Document Preview Modal / Expander if requested
    if selected_doc_preview:
        with st.expander(f"👁️ Viewing Document: {selected_doc_preview.get('filename')}", expanded=True):
            st.markdown(f"""
            <div style="display:flex; gap:16px; margin-bottom:14px; font-size:0.82rem; color:#94a3b8;">
                <span>📄 Total Pages: <b>{selected_doc_preview.get('total_pages', 1)}</b></span>
                <span>🧩 Total Chunks: <b>{selected_doc_preview.get('chunk_count', 0)}</b></span>
                <span>🔤 Characters: <b>{selected_doc_preview.get('total_chars', 0):,}</b></span>
                <span>📝 Words: <b>{selected_doc_preview.get('total_words', 0):,}</b></span>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area(
                "Document Extracted Text",
                selected_doc_preview.get("full_text", "No text content available."),
                height=280,
                disabled=True
            )

    st.markdown("<div style='margin:28px 0;'></div>", unsafe_allow_html=True)

    # Document Library Section
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <h3 style="margin:0; font-size:1.2rem; font-weight:700; color:#f8fafc;">
            📚 Document Library ({len(doc_infos)} files indexed)
        </h3>
    </div>
    """, unsafe_allow_html=True)

    if not doc_infos:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:36px 20px;">
            <div style="font-size:2.8rem; margin-bottom:12px;">📄</div>
            <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:6px;">Your document workspace is empty</div>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:400px; margin:0 auto 16px auto;">
                Upload your first document or click 'Load Instant Academic Samples' above to explore RAG, search, and summarization.
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
