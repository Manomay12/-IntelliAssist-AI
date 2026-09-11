"""
AI Chat Page View for IntelliAssist AI.
Grounded conversational RAG interface with ML query understanding,
streaming generation, source citations, document filtering, and prompt suggestion chips.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st
from components.chat_ui import render_chat_message, render_suggested_questions
from utils.helpers import get_file_icon

def render_chat_page(
    messages: List[Dict[str, Any]],
    all_documents: List[str],
    selected_doc_filter: str,
    on_send_message: Callable[[str, str], None],
    on_feedback: Callable[[str, str], None],
    on_new_session: Callable[[], None],
    on_clear_chat: Callable[[], None],
    on_export_markdown: Callable[[], str],
    session_title: str = "New Discussion",
    saved_sessions: Optional[List[Dict[str, Any]]] = None,
    current_session_id: Optional[str] = None,
    on_load_session: Optional[Callable[[str], None]] = None,
    on_delete_session: Optional[Callable[[str], None]] = None,
    doc_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    on_upload_files: Optional[Callable[[List[Any]], None]] = None,
    question_generator: Optional[Any] = None,
    *args,
    **kwargs
):
    """Render the AI Chat Page."""
    registry = doc_registry or kwargs.get("doc_registry", {})
    total_docs = len(all_documents)

    # Top Header Banner
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
        <div>
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="font-weight:800; color:#ffffff; margin:0; letter-spacing:-0.03em; font-size:1.6rem;">IntelliAssist AI Chat</h2>
                <span style="font-size:0.75rem; background:rgba(6,182,212,0.12); color:#38bdf8; padding:3px 12px; border-radius:9999px; border:1px solid rgba(6,182,212,0.25); font-weight:700;">
                    Grounded RAG Pipeline
                </span>
            </div>
            <p style="color:#94a3b8; font-size:0.86rem; margin:4px 0 0 0;">
                Discussion: <b style="color:#f8fafc;">{session_title}</b> • <span style="color:#38bdf8; font-weight:600;">{len(messages)} messages</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Document Scope & Action Toolbar
    filter_options = ["All Documents"] + all_documents
    active_scope = st.session_state.get("chat_doc_filter", selected_doc_filter)
    if active_scope not in filter_options:
        active_scope = "All Documents"
        st.session_state.chat_doc_filter = "All Documents"

    default_filter_idx = filter_options.index(active_scope)

    col_scope, col_upload, col_new, col_clear, col_hist, col_export = st.columns([2.5, 1.2, 1.0, 1.0, 1.2, 1.1])

    with col_scope:
        doc_filter = st.selectbox(
            f"Target Scope ({total_docs} Files)",
            filter_options,
            index=default_filter_idx,
            key=f"chat_scope_selector_{active_scope}",
            help="Choose whether to query across all documents or target a specific file"
        )
        st.session_state.chat_doc_filter = doc_filter

    with col_upload:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        with st.popover("Upload", use_container_width=True, help="Upload and index new documents into chat"):
            st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#ffffff; margin-bottom:6px;'>Upload to Active Scope</div>", unsafe_allow_html=True)
            chat_files = st.file_uploader(
                "Upload files",
                type=["pdf", "docx", "doc", "txt", "md"],
                accept_multiple_files=True,
                key="chat_direct_file_uploader",
                label_visibility="collapsed"
            )
            if chat_files and on_upload_files:
                if st.button(f"Index {len(chat_files)} File(s)", key="chat_index_btn", type="primary", use_container_width=True):
                    on_upload_files(chat_files)
    
    with col_new:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("New", key="chat_new_session_btn", type="primary", help="Start fresh chat session", use_container_width=True):
            on_new_session()
            
    with col_clear:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        with st.popover("Clear", use_container_width=True, help="Clear active conversation messages"):
            st.markdown("<p style='font-size:0.82rem; color:#f8fafc; margin-bottom:8px;'>Clear all messages in active session?</p>", unsafe_allow_html=True)
            if st.button("Confirm Clear", key="chat_confirm_clear_btn", type="primary", use_container_width=True):
                on_clear_chat()
                
    with col_hist:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        sessions_list = saved_sessions or []
        hist_label = f"History ({len(sessions_list)})"
        with st.popover(hist_label, use_container_width=True, help="Switch between saved discussions"):
            st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#ffffff; margin-bottom:8px;'>Saved Discussions</div>", unsafe_allow_html=True)
            if not sessions_list:
                st.caption("No saved chats yet.")
            else:
                for s in sessions_list[:10]:
                    s_id = s.get("session_id", "")
                    s_title = s.get("title", "Conversation")
                    if len(s_title) > 24:
                        s_title = s_title[:22] + "..."
                    m_cnt = s.get("message_count", 0)
                    is_cur = (s_id == current_session_id)
                    prefix = "• " if is_cur else ""
                    
                    col_pick, col_del = st.columns([4, 1])
                    with col_pick:
                        if st.button(f"{prefix}{s_title} ({m_cnt})", key=f"quick_pick_{s_id}", use_container_width=True):
                            if on_load_session:
                                on_load_session(s_id)
                    with col_del:
                        if st.button("✕", key=f"quick_del_{s_id}", help="Delete this discussion"):
                            if on_delete_session:
                                on_delete_session(s_id)
                                
    with col_export:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        md_content = on_export_markdown()
        st.download_button(
            "Export",
            data=md_content,
            file_name=f"chat_{session_title.lower().replace(' ', '_')[:20]}.md",
            mime="text/markdown",
            key="chat_export_btn",
            help="Export conversation as Markdown transcript",
            use_container_width=True
        )

    # Active Single-Document Scope Indicator
    if doc_filter != "All Documents" and doc_filter in registry:
        active_doc = registry[doc_filter]
        category = active_doc.get("category", "General Document")
        st.markdown(f"""
        <div style="background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.3); border-radius:12px; padding:10px 16px; margin:6px 0 14px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.1rem; color:#38bdf8;">•</span>
                <div>
                    <div style="font-weight:700; font-size:0.92rem; color:#ffffff;">Targeted Scope: <span style="color:#38bdf8;">{doc_filter}</span></div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:1px;">{category} • {active_doc.get('total_pages', 1)} pages • {active_doc.get('chunk_count', 0)} chunks indexed</div>
                </div>
            </div>
            <span style="font-size:0.72rem; background:rgba(16,185,129,0.15); color:#34d399; padding:3px 10px; border-radius:9999px; border:1px solid rgba(16,185,129,0.3); font-weight:700;">
                Single Document Focus
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

    # Empty State or Message Stream
    if not messages:
        target_name = f"'{doc_filter}'" if doc_filter != "All Documents" else "all indexed documents"
        st.markdown(f"""
        <div class="modern-card" style="text-align:center; padding:42px 20px; margin:24px 0;">
            <h3 style="font-weight:700; font-size:1.2rem; color:#f8fafc; margin:0 0 6px 0;">
                Query IntelliAssist AI on {target_name}
            </h3>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:540px; margin:0 auto 20px auto;">
                Ask technical questions, request multi-section breakdowns, or explore empirical outcomes.
                Every claim is grounded in retrieved passages with verified citations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in messages:
            render_chat_message(
                msg,
                on_feedback=on_feedback
            )

    # Suggested Prompts Row
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    target_prompt_scope = doc_filter
    render_suggested_questions(
        on_select=lambda q: on_send_message(q, target_prompt_scope),
        active_doc=target_prompt_scope,
        doc_registry=registry,
        question_generator=question_generator
    )

    # Bottom Chat Input
    st.markdown("<div style='margin:14px 0;'></div>", unsafe_allow_html=True)
    current_scope = st.session_state.get("chat_doc_filter", doc_filter)
    user_prompt = st.chat_input(
        placeholder=f"Ask anything about {'all documents' if current_scope == 'All Documents' else current_scope}...",
        key="main_chat_input"
    )
    if user_prompt:
        on_send_message(user_prompt, current_scope)
