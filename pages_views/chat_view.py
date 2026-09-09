"""
AI Chat Page View for IntelliAssist AI.
The primary conversational RAG interface with in-chat direct document uploading, multi-document intelligence inspector,
file profile cards, exhaustive explanation generator, streaming answers, source citations, feedback, prompt suggestions, and high-visibility action controls.
"""

from typing import Dict, Any, List, Callable, Optional
# pyrefly: ignore [missing-import]
import streamlit as st
from components.chat_ui import render_chat_message, render_suggested_questions

def render_chat_page(
    messages: List[Dict[str, Any]],
    all_documents: List[str],
    selected_doc_filter: str,
    on_send_message: Callable[[str, str], None],
    on_feedback: Callable[[str, str], None],
    on_new_session: Callable[[], None],
    on_clear_chat: Callable[[], None],
    on_export_markdown: Callable[[], str],
    session_title: str = "New Conversation",
    saved_sessions: Optional[List[Dict[str, Any]]] = None,
    current_session_id: Optional[str] = None,
    on_load_session: Optional[Callable[[str], None]] = None,
    on_delete_session: Optional[Callable[[str], None]] = None,
    doc_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    on_upload_files: Optional[Callable[[List[Any]], None]] = None,
    question_generator: Optional[Any] = None
):
    """Render the high-visibility AI Chat Page with in-chat uploading, file intelligence details, and deep explanations."""
    registry = doc_registry or {}
    total_docs = len(all_documents)
    total_pages = sum(d.get("total_pages", 1) for d in registry.values())
    total_words = sum(d.get("total_words", 0) for d in registry.values())
    total_chunks = sum(d.get("chunk_count", 0) for d in registry.values())

    # Top Header Banner
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:10px;">
        <div>
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="font-weight:800; color:#ffffff; margin:0; letter-spacing:-0.02em; font-size:1.6rem;">💬 IntelliAssist AI Chat</h2>
                <span style="font-size:0.75rem; background:rgba(99,102,241,0.2); color:#c7d2fe; padding:4px 12px; border-radius:9999px; border:1px solid rgba(99,102,241,0.4); font-weight:600;">
                    🟢 RAG Grounded & Deep Explanation
                </span>
            </div>
            <p style="color:#94a3b8; font-size:0.88rem; margin:4px 0 0 0;">
                Active Discussion: <b style="color:#f8fafc;">{session_title}</b> • <span style="color:#a5b4fc; font-weight:600;">{len(messages)} messages exchanged</span>
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

    col_scope, col_upload, col_new, col_clear, col_hist, col_export = st.columns([2.4, 1.3, 1.1, 1.1, 1.2, 1.1])

    with col_scope:
        doc_filter = st.selectbox(
            f"📄 Target Scope ({total_docs} Files)",
            filter_options,
            index=default_filter_idx,
            key=f"chat_scope_selector_{active_scope}",
            help="Choose whether to query all indexed documents or target search strictly to a single file"
        )
        st.session_state.chat_doc_filter = doc_filter

    with col_upload:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        with st.popover("📤 Upload", use_container_width=True, help="Upload and index new PDF, DOCX, TXT, MD, CSV files directly into AI Chat"):
            st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#ffffff; margin-bottom:6px;'>📤 Upload to AI Chat</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem; color:#cbd5e1; margin-bottom:8px;'>Upload any PDF, Word (.docx), Text, Markdown, or CSV file. It will be indexed immediately for chat!</p>", unsafe_allow_html=True)
            chat_files = st.file_uploader(
                "Upload files",
                type=["pdf", "docx", "doc", "txt", "md", "csv", "json", "log"],
                accept_multiple_files=True,
                key="chat_direct_file_uploader",
                label_visibility="collapsed"
            )
            if chat_files and on_upload_files:
                if st.button(f"⚡ Index {len(chat_files)} File(s) Now", key="chat_index_btn", type="primary", use_container_width=True):
                    on_upload_files(chat_files)
    
    with col_new:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ New", key="chat_new_session_btn", type="primary", help="Start a new chat session", use_container_width=True):
            on_new_session()
            
    with col_clear:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        with st.popover("🗑️ Clear", use_container_width=True, help="Clear active conversation messages"):
            st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#ffffff; margin-bottom:6px;'>🧹 Clear Chat Messages</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.82rem; color:#cbd5e1; margin-bottom:10px;'>Wipe all messages from this active session? This action cannot be undone.</p>", unsafe_allow_html=True)
            if st.button("Yes, Clear Chat", key="chat_confirm_clear_btn", type="primary", use_container_width=True):
                on_clear_chat()
                
    with col_hist:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        sessions_list = saved_sessions or []
        hist_label = f"📜 History ({len(sessions_list)})"
        with st.popover(hist_label, use_container_width=True, help="Switch between saved discussions"):
            st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#ffffff; margin-bottom:8px;'>Saved Conversations</div>", unsafe_allow_html=True)
            if not sessions_list:
                st.caption("No saved chats yet.")
            else:
                for s in sessions_list[:10]:
                    s_id = s.get("session_id", "")
                    s_title = s.get("title", "Conversation")
                    if len(s_title) > 26:
                        s_title = s_title[:24] + "..."
                    m_cnt = s.get("message_count", 0)
                    is_cur = (s_id == current_session_id)
                    prefix = "👉 " if is_cur else "💬 "
                    
                    col_pick, col_del = st.columns([4, 1])
                    with col_pick:
                        if st.button(f"{prefix}{s_title} ({m_cnt})", key=f"quick_pick_{s_id}", use_container_width=True):
                            if on_load_session:
                                on_load_session(s_id)
                    with col_del:
                        if st.button("🗑️", key=f"quick_del_{s_id}", help="Delete this chat"):
                            if on_delete_session:
                                on_delete_session(s_id)
                                
    with col_export:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        md_content = on_export_markdown()
        st.download_button(
            "📥 Export",
            data=md_content,
            file_name=f"chat_{session_title.lower().replace(' ', '_')[:20]}.md",
            mime="text/markdown",
            key="chat_export_btn",
            help="Export conversation as Markdown transcript",
            use_container_width=True
        )

    # -------------------------------------------------------------
    # 📑 FILE INTELLIGENCE INSPECTOR & UPLOADED FILE PROFILES
    # -------------------------------------------------------------
    with st.expander(f"📑 Document Intelligence Inspector & Uploaded Files ({total_docs} Files)", expanded=False):
        # Summary KPI Badges
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin-bottom:14px;">
            <div style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3); border-radius:10px; padding:10px 14px; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#c7d2fe;">{total_docs}</div>
                <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Uploaded Files</div>
            </div>
            <div style="background:rgba(14,165,233,0.12); border:1px solid rgba(14,165,233,0.3); border-radius:10px; padding:10px 14px; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#7dd3fc;">{total_pages}</div>
                <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Total Pages</div>
            </div>
            <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:10px 14px; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#6ee7b7;">{total_words:,}</div>
                <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Total Words</div>
            </div>
            <div style="background:rgba(139,92,246,0.12); border:1px solid rgba(139,92,246,0.3); border-radius:10px; padding:10px 14px; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#d8b4fe;">{total_chunks}</div>
                <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Semantic Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not registry:
            st.info("No documents uploaded yet. Use the '📤 Upload' button in the toolbar above to add your PDF, DOCX, or TXT files!")
        else:
            for fname, doc_info in registry.items():
                ext = doc_info.get("file_ext", ".txt").lower()
                icon = "📕" if ext == ".pdf" else "📘" if ext in [".docx", ".doc"] else "📄"
                fsize_kb = round(doc_info.get("file_size", 0) / 1024, 1)
                fsize_str = f"{fsize_kb} KB" if fsize_kb < 1024 else f"{round(fsize_kb/1024, 2)} MB"
                p_cnt = doc_info.get("total_pages", 1)
                w_cnt = doc_info.get("total_words", 0)
                c_cnt = doc_info.get("chunk_count", 0)
                u_date = doc_info.get("upload_date", "Recently")
                raw_text = doc_info.get("full_text", "")
                snippet = (raw_text[:220] + "...") if len(raw_text) > 220 else raw_text

                is_selected = (doc_filter == fname)
                border_color = "rgba(99,102,241,0.8)" if is_selected else "rgba(255,255,255,0.08)"
                bg_color = "rgba(99,102,241,0.15)" if is_selected else "rgba(15,23,42,0.6)"

                st.markdown(f"""
                <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:12px; padding:14px 16px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                        <div>
                            <div style="font-weight:700; font-size:0.95rem; color:#ffffff; display:flex; align-items:center; gap:8px;">
                                <span>{icon}</span> <span>{fname}</span>
                                {'<span style="font-size:0.7rem; background:#6366f1; color:#fff; padding:1px 8px; border-radius:9999px;">Active Target</span>' if is_selected else ''}
                            </div>
                            <div style="display:flex; gap:12px; font-size:0.78rem; color:#94a3b8; margin-top:4px; flex-wrap:wrap;">
                                <span>📦 Size: <b>{fsize_str}</b></span>
                                <span>📄 Pages: <b>{p_cnt}</b></span>
                                <span>🔤 Words: <b>{w_cnt:,}</b></span>
                                <span>🧩 Chunks: <b>{c_cnt} vectors</b></span>
                                <span>🕒 Uploaded: <b>{u_date}</b></span>
                            </div>
                        </div>
                    </div>
                    <div style="font-size:0.8rem; color:#cbd5e1; margin-top:8px; line-height:1.45; background:rgba(0,0,0,0.25); padding:8px 10px; border-radius:6px; border-left:3px solid #6366f1;">
                        <b>Content Preview:</b> {snippet}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if not is_selected:
                    col_btn1, col_btn2 = st.columns([1.5, 4])
                    with col_btn1:
                        if st.button(f"🎯 Target this file", key=f"target_btn_{fname}", use_container_width=True):
                            st.session_state.chat_doc_filter = fname
                            st.rerun()

    # Active Single-Document Insight Card (when a single document is selected in scope)
    if doc_filter != "All Documents" and doc_filter in registry:
        active_doc = registry[doc_filter]
        f_ext = active_doc.get("file_ext", ".txt").lower()
        f_icon = "📕" if f_ext == ".pdf" else "📘" if f_ext in [".docx", ".doc"] else "📄"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(139,92,246,0.15) 100%); border:1px solid rgba(99,102,241,0.5); border-radius:12px; padding:12px 16px; margin:4px 0 14px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.4rem;">{f_icon}</span>
                <div>
                    <div style="font-weight:700; font-size:0.95rem; color:#ffffff;">Currently Answering From: <span style="color:#a5b4fc;">{doc_filter}</span></div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">{active_doc.get('total_pages', 1)} pages • {active_doc.get('total_words', 0):,} words • {active_doc.get('chunk_count', 0)} chunks indexed • All answers will be 100% focused on this document</div>
                </div>
            </div>
            <span style="font-size:0.75rem; background:rgba(16,185,129,0.2); color:#6ee7b7; padding:4px 12px; border-radius:9999px; border:1px solid rgba(16,185,129,0.4); font-weight:600;">
                🎯 Strict Single-Document Scope Active
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif doc_filter == "All Documents":
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:8px 14px; margin:4px 0 12px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <span style="font-size:0.82rem; color:#94a3b8;">
                🌐 <b>Multi-Document Scope Active:</b> Searching across all {total_docs} uploaded documents ({total_chunks} chunks). To focus on a specific file, choose it from the Target Scope dropdown above.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0; border-bottom:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

    # Empty State or Message Stream
    if not messages:
        target_name = "your selected file" if doc_filter != "All Documents" else "all uploaded documents"
        st.markdown(f"""
        <div class="modern-card" style="text-align:center; padding:36px 20px; margin:20px 0;">
            <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
            <h3 style="font-weight:700; font-size:1.15rem; color:#f8fafc; margin:0 0 6px 0;">
                Ask IntelliAssist AI for Full Explanations on {target_name}
            </h3>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:520px; margin:0 auto 20px auto;">
                Ask complex questions, request complete technical breakdowns, or explore findings. Answers are factual, structured, and cite exact source pages.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in messages:
            render_chat_message(
                msg,
                on_feedback=on_feedback
            )

    # Suggested Prompts
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
        placeholder=f"Ask anything about {'all documents' if current_scope == 'All Documents' else current_scope} for a full explanation...",
        key="main_chat_input"
    )
    if user_prompt:
        on_send_message(user_prompt, current_scope)
