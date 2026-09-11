"""
Dashboard Page View for IntelliAssist AI.
Redesigned with the hero section, drag-and-drop ingestion hub,
intelligence pipeline visualizer, recently opened documents, and telemetry activity feed.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st
from components.metrics import render_metric_grid
from components.document_card import render_document_card

def render_dashboard(
    doc_infos: Optional[List[Dict[str, Any]]] = None,
    total_chunks: int = 0,
    total_questions: int = 0,
    total_convs: int = 0,
    recent_activities: Optional[List[Dict[str, Any]]] = None,
    on_quick_ask: Optional[Callable[[str], None]] = None,
    on_navigate: Optional[Callable[[str], None]] = None,
    on_open_doc: Optional[Callable[[str], None]] = None,
    on_summarize_doc: Optional[Callable[[str], None]] = None,
    on_chat_doc: Optional[Callable[[str], None]] = None,
    on_delete_doc: Optional[Callable[[str], None]] = None,
    on_load_samples: Optional[Callable[[], None]] = None,
    on_upload_files: Optional[Callable[[List[Any]], None]] = None,
    *args,
    **kwargs
):
    """Render the modernized hero dashboard and document intelligence workspace."""
    # Defensive fallbacks for kwargs or omitted parameters
    if doc_infos is None:
        doc_infos = kwargs.get("doc_infos") or []
    if recent_activities is None:
        recent_activities = kwargs.get("recent_activities") or []
    if on_quick_ask is None:
        on_quick_ask = kwargs.get("on_quick_ask") or (lambda q: None)
    if on_navigate is None:
        on_navigate = kwargs.get("on_navigate") or (lambda p: None)
    if on_open_doc is None:
        on_open_doc = kwargs.get("on_open_doc") or (lambda d: None)
    if on_summarize_doc is None:
        on_summarize_doc = kwargs.get("on_summarize_doc") or (lambda d: None)
    if on_chat_doc is None:
        on_chat_doc = kwargs.get("on_chat_doc") or (lambda d: None)
    if on_delete_doc is None:
        on_delete_doc = kwargs.get("on_delete_doc") or (lambda d: None)
    if on_load_samples is None:
        on_load_samples = kwargs.get("on_load_samples") or (lambda: None)
    if on_upload_files is None:
        on_upload_files = kwargs.get("on_upload_files")
    # 1. Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">Next-Generation Document Intelligence</div>
        <h1 class="hero-title">
            Your Documents. <span class="hero-gradient-text">Understood.</span>
        </h1>
        <p class="hero-subtitle">
            Read. Search. Learn. Analyze. Upload complex PDFs, reports, and academic research papers. 
            Extract multi-page insights, run hybrid semantic search, and verify evidence with exact citations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Hero Central Upload Drop-Zone Hub
    col_l, col_center, col_r = st.columns([1, 8, 1])
    with col_center:
        st.markdown("""
        <div class="upload-dropzone">
            <div style="font-size:2rem; color:#38bdf8; margin-bottom:8px;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
            </div>
            <div style="font-weight:700; font-size:1.15rem; color:#ffffff; margin-bottom:4px;">
                Drop your document here or browse files
            </div>
            <div style="font-size:0.84rem; color:#94a3b8; margin-bottom:14px;">
                Supported: PDF • DOCX • TXT • MD
            </div>
        </div>
        """, unsafe_allow_html=True)

        dash_upload = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "doc", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="dash_hero_uploader"
        )
        if dash_upload and on_upload_files:
            if st.button(f"Process & Index {len(dash_upload)} Document(s)", key="dash_hero_index_btn", type="primary", use_container_width=True):
                on_upload_files(dash_upload)

    st.markdown("<div style='margin:28px 0;'></div>", unsafe_allow_html=True)

    # 3. Telemetry Metrics Grid
    render_metric_grid(len(doc_infos), total_chunks, total_questions, total_convs)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # 4. Intelligence Pipeline Visualizer
    st.markdown("""
    <div style="margin-bottom:8px; font-size:0.82rem; font-weight:700; color:#38bdf8; letter-spacing:0.06em; text-transform:uppercase;">
        INTELLIGENCE PIPELINE
    </div>
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="pipeline-step-node">01</div>
            <div class="pipeline-step-title">Upload</div>
            <div class="pipeline-step-desc">PDF • DOCX • TXT</div>
        </div>
        <div class="pipeline-connector"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-node">02</div>
            <div class="pipeline-step-title">Extract</div>
            <div class="pipeline-step-desc">Structure & OCR</div>
        </div>
        <div class="pipeline-connector"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-node">03</div>
            <div class="pipeline-step-title">Understand</div>
            <div class="pipeline-step-desc">Intent & Semantics</div>
        </div>
        <div class="pipeline-connector"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-node">04</div>
            <div class="pipeline-step-title">Semantic Index</div>
            <div class="pipeline-step-desc">384-d Dense + BM25</div>
        </div>
        <div class="pipeline-connector"></div>
        <div class="pipeline-step">
            <div class="pipeline-step-node">05</div>
            <div class="pipeline-step-title">Ask AI</div>
            <div class="pipeline-step-desc">Grounded RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # 5. Quick Ask AI Widget
    st.markdown("""
    <div class="modern-card" style="border-color:rgba(56, 189, 248, 0.3);">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="background:rgba(6,182,212,0.12); padding:6px; border-radius:8px; border:1px solid rgba(6,182,212,0.25);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
            </div>
            <h3 style="margin:0; font-size:1.1rem; font-weight:700; color:#f8fafc;">Search & Query Across All Documents</h3>
        </div>
        <p style="color:#94a3b8; font-size:0.86rem; margin-bottom:14px;">
            Ask questions to retrieve hybrid semantic context with verified source citations and confidence scores.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_btn = st.columns([5.2, 1.2])
    with col_input:
        quick_query = st.text_input(
            "Quick question input",
            placeholder="Ask anything about your documents (e.g., 'What are the main findings of the research paper?')",
            label_visibility="collapsed",
            key="dash_quick_ask_input"
        )
    with col_btn:
        if st.button("Ask AI", key="dash_quick_ask_btn", type="primary", use_container_width=True):
            if quick_query.strip():
                on_quick_ask(quick_query)

    st.markdown("<div style='margin:28px 0;'></div>", unsafe_allow_html=True)

    # 6. Two Column Layout: Recent Documents & Recent Activity
    col_docs, col_act = st.columns([1.1, 1])

    with col_docs:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f8fafc;">Recently Opened Documents</h3>
        </div>
        """, unsafe_allow_html=True)

        if not doc_infos:
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding:32px 20px;">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom:6px;">Your workspace is empty</div>
                <p style="font-size:0.85rem; color:#94a3b8; max-width:340px; margin:0 auto 16px auto;">
                    Upload documents or load pre-packaged benchmark samples for instant evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Upload Documents", key="dash_empty_upload", use_container_width=True):
                    on_navigate("Documents")
            with c2:
                if st.button("Load Sample Papers", key="dash_empty_samples", use_container_width=True):
                    on_load_samples()
        else:
            for doc in doc_infos[:3]:
                render_document_card(
                    doc,
                    on_open=on_open_doc,
                    on_summarize=on_summarize_doc,
                    on_chat=on_chat_doc,
                    on_delete=on_delete_doc,
                    key_prefix="dash"
                )
            if len(doc_infos) > 3:
                st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                if st.button(f"View All ({len(doc_infos)}) Documents in Library", key="dash_view_all_docs", use_container_width=True):
                    on_navigate("Documents")

    with col_act:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f8fafc;">Intelligence Activity Feed</h3>
        </div>
        """, unsafe_allow_html=True)

        if not recent_activities:
            st.markdown("""
            <div class="activity-item" style="padding:28px 18px; text-align:center;">
                <div style="font-weight:600; font-size:0.9rem; color:#cbd5e1;">No recent activity logged</div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">Ask questions or run semantic searches to see real-time activity logs here.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for act in recent_activities[:4]:
                title = act.get("title", "AI Query")
                time_str = act.get("time", "Just now")
                desc = act.get("desc", "")
                
                st.markdown(f"""
                <div class="activity-item">
                    <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
                        <div style="font-weight:600; font-size:0.86rem; color:#f8fafc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{title}">
                            {title}
                        </div>
                        <span style="font-size:0.72rem; color:#38bdf8; background:rgba(6,182,212,0.1); padding:2px 8px; border-radius:9999px; border:1px solid rgba(6,182,212,0.25); flex-shrink:0;">
                            {time_str}
                        </span>
                    </div>
                    {f'<div style="font-size:0.78rem; color:#94a3b8; margin-top:6px; line-height:1.4;">{desc}</div>' if desc else ''}
                </div>
                """, unsafe_allow_html=True)
