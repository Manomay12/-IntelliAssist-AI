"""
Dashboard Page View for IntelliAssist AI.
Displays key metrics, Quick Ask AI card, Recent Documents with actions, and AI activity feed.
"""

from typing import Dict, Any, List, Callable
import streamlit as st
from components.metrics import render_metric_grid
from components.document_card import render_document_card
from utils.helpers import get_file_icon

def render_dashboard(
    doc_infos: List[Dict[str, Any]],
    total_chunks: int,
    total_questions: int,
    total_convs: int,
    recent_activities: List[Dict[str, Any]],
    on_quick_ask: Callable[[str], None],
    on_navigate: Callable[[str], None],
    on_open_doc: Callable[[str], None],
    on_summarize_doc: Callable[[str], None],
    on_chat_doc: Callable[[str], None],
    on_delete_doc: Callable[[str], None],
    on_load_samples: Callable[[], None]
):
    """Render the main dashboard."""
    # Top Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1 style="margin:0 0 8px 0; font-size:1.85rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">
                    Good day! 👋 <span style="background:linear-gradient(135deg, #a5b4fc 0%, #818cf8 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Welcome to IntelliAssist AI</span>
                </h1>
                <p style="margin:0; color:#94a3b8; font-size:0.95rem; max-width:680px;">
                    Upload documents, extract deep insights with RAG, perform semantic search, and synthesize multi-page summaries with exact citations.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Metrics Grid
    render_metric_grid(len(doc_infos), total_chunks, total_questions, total_convs)

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # Quick Ask AI Widget
    st.markdown("""
    <div class="modern-card" style="background:linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%); border-color:rgba(99, 102, 241, 0.3);">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <span style="font-size:1.3rem;">⚡</span>
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f8fafc;">Quick Ask AI Across All Documents</h3>
        </div>
        <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:16px;">
            Type any question to retrieve context-aware answers with source citations using our RAG pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1.2])
    with col_input:
        quick_query = st.text_input(
            "Quick question input",
            placeholder="Ask anything about your documents (e.g., 'What are the main findings of the research paper?')",
            label_visibility="collapsed",
            key="dash_quick_ask_input"
        )
    with col_btn:
        if st.button("🚀 Ask AI", key="dash_quick_ask_btn", use_container_width=True):
            if quick_query.strip():
                on_quick_ask(quick_query)

    st.markdown("<div style='margin:30px 0;'></div>", unsafe_allow_html=True)

    # Two Column Layout: Recent Documents & Recent Activity
    col_docs, col_act = st.columns([1.1, 1])

    with col_docs:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f8fafc;">📄 Recent Documents</h3>
        </div>
        """, unsafe_allow_html=True)

        if not doc_infos:
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding:32px 20px;">
                <div style="font-size:2.5rem; margin-bottom:10px;">📂</div>
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom:6px;">Your workspace is empty</div>
                <p style="font-size:0.85rem; color:#94a3b8; max-width:340px; margin:0 auto 16px auto;">
                    Upload your own PDF, DOCX, or TXT documents, or load the pre-packaged sample documents for instant testing.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ Upload Documents", key="dash_empty_upload", use_container_width=True):
                    on_navigate("Documents")
            with c2:
                if st.button("📦 Load Sample Documents", key="dash_empty_samples", use_container_width=True):
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
                if st.button(f"View All ({len(doc_infos)}) Documents →", key="dash_view_all_docs"):
                    on_navigate("Documents")

    with col_act:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f8fafc;">⚡ Recent AI Activity</h3>
        </div>
        """, unsafe_allow_html=True)

        if not recent_activities:
            st.markdown("""
            <div class="modern-card" style="padding:24px 18px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:8px;">🕘</div>
                <div style="font-weight:600; font-size:0.9rem; color:#cbd5e1;">No recent activity yet</div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">Ask questions or run semantic searches to see activity logs here.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for act in recent_activities[:5]:
                icon = act.get("icon", "💬")
                title = act.get("title", "AI Query")
                time_str = act.get("time", "Just now")
                desc = act.get("desc", "")
                
                st.markdown(f"""
                <div class="modern-card" style="padding:12px 16px; margin-bottom:8px;">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="font-size:1.1rem;">{icon}</span>
                            <span style="font-weight:600; font-size:0.85rem; color:#f8fafc;">{title}</span>
                        </div>
                        <span style="font-size:0.72rem; color:#64748b;">{time_str}</span>
                    </div>
                    {f'<div style="font-size:0.78rem; color:#94a3b8; margin-top:4px; line-height:1.4;">{desc}</div>' if desc else ''}
                </div>
                """, unsafe_allow_html=True)
