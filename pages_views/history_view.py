"""
Conversation History Page View for IntelliAssist AI.
Manages saved chat sessions, search, renaming, individual deletion, and bulk history deletion.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st


def render_history_page(
    sessions: List[Dict[str, Any]],
    current_session_id: str,
    on_load_session: Callable[[str], None],
    on_rename_session: Callable[[str, str], None],
    on_delete_session: Callable[[str], None],
    on_delete_all_sessions: Callable[[], None],
    on_new_chat: Callable[[], None],
    *args,
    **kwargs
):
    """Render the enhanced Conversation History manager with bulk delete options."""
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:9999px; background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.25); margin-bottom:8px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#22d3ee; letter-spacing:0.04em; text-transform:uppercase;">Session Archives</span>
        </div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin:0 0 6px 0;">
            Conversation History
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0; max-width:760px; line-height:1.5;">
            Review past Q&A sessions, reopen previous discussions, export transcripts, or manage chat records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top Metrics & Actions Bar
    total_messages = sum(s.get("message_count", 0) for s in sessions)
    
    m_c1, m_c2, m_c3, a_c1, a_c2 = st.columns([1.2, 1.2, 1.2, 1.4, 1.6])
    with m_c1:
        st.markdown(f"""
        <div class="metric-card" style="padding:10px 14px;">
            <div class="metric-value" style="font-size:1.3rem;">{len(sessions)}</div>
            <div class="metric-label" style="font-size:0.75rem;">Total Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c2:
        st.markdown(f"""
        <div class="metric-card" style="padding:10px 14px;">
            <div class="metric-value" style="font-size:1.3rem;">{total_messages}</div>
            <div class="metric-label" style="font-size:0.75rem;">Total Messages</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c3:
        st.markdown(f"""
        <div class="metric-card" style="padding:10px 14px;">
            <div class="metric-value" style="font-size:1.1rem; font-family:monospace; color:#38bdf8;">{current_session_id[:6]}...</div>
            <div class="metric-label" style="font-size:0.75rem;">Active Session</div>
        </div>
        """, unsafe_allow_html=True)
    with a_c1:
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        if st.button("New Session", key="btn_hist_new_chat", type="primary", use_container_width=True):
            on_new_chat()
    with a_c2:
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        with st.popover("Clear All History", use_container_width=True, help="Permanently delete all saved conversations"):
            st.markdown("<p style='font-size:0.85rem; color:#ef4444; font-weight:700; margin:0 0 6px 0;'>Permanent Deletion</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem; color:#cbd5e1; margin:0 0 10px 0;'>This will delete all saved conversation logs. This action cannot be undone.</p>", unsafe_allow_html=True)
            if st.button("Yes, Clear Everything", key="btn_confirm_del_all_hist", type="primary", use_container_width=True):
                on_delete_all_sessions()

    st.markdown("<div style='margin:16px 0 10px 0;'></div>", unsafe_allow_html=True)

    # Search Filter Bar
    search_kw = st.text_input(
        "Filter History",
        placeholder="Filter conversations by title, topic, or document...",
        label_visibility="collapsed",
        key="history_search_input"
    )

    st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)

    # Filter sessions
    filtered_sessions = sessions
    if search_kw.strip():
        filtered_sessions = [s for s in sessions if search_kw.lower() in s.get("title", "").lower()]

    if not filtered_sessions:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:48px 24px;">
            <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:6px;">No saved conversations found</div>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:400px; margin:0 auto 16px auto; line-height:1.6;">
                Ask questions in the AI Chat to automatically archive conversation sessions here.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for s in filtered_sessions:
            s_id = s.get("session_id", "")
            title = s.get("title", "Conversation")
            date_str = s.get("updated_at", s.get("created_at", "Recently"))
            msg_count = s.get("message_count", 0)
            docs = s.get("associated_docs", [])
            is_active = (s_id == current_session_id)

            badge_html = "<span style='font-size:0.72rem; background:rgba(16,185,129,0.15); color:#6ee7b7; padding:2px 8px; border-radius:9999px; border:1px solid rgba(16,185,129,0.3); margin-left:8px; font-weight:600;'>Active</span>" if is_active else ""

            with st.container():
                st.markdown(f"""
                <div class="modern-card" style="margin-bottom:12px; padding:16px 20px; border-left:3px solid {'#06b6d4' if is_active else 'rgba(255,255,255,0.08)'};">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="font-weight:700; font-size:1.02rem; color:#f8fafc;">{title}</span>
                                {badge_html}
                            </div>
                            <div style="display:flex; gap:16px; font-size:0.78rem; color:#94a3b8; margin-top:6px; flex-wrap:wrap;">
                                <span>{date_str}</span>
                                <span><b>{msg_count}</b> message(s)</span>
                                {f'<span>Scope: {", ".join(docs[:2])}</span>' if docs else ''}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c_open, c_ren, c_del, c_space = st.columns([1.3, 1.2, 1.2, 4.3])
                with c_open:
                    if st.button("Reopen", key=f"hist_open_{s_id}", use_container_width=True):
                        on_load_session(s_id)
                with c_ren:
                    with st.popover("Rename", use_container_width=True):
                        new_name = st.text_input("New Title", value=title, key=f"ren_input_{s_id}")
                        if st.button("Save", key=f"save_title_{s_id}"):
                            if new_name.strip():
                                on_rename_session(s_id, new_name.strip())
                                st.rerun()
                with c_del:
                    with st.popover("Delete", use_container_width=True):
                        st.markdown("<p style='font-size:0.8rem; color:#cbd5e1; margin:0 0 8px 0;'>Delete this chat session?</p>", unsafe_allow_html=True)
                        if st.button("Confirm", key=f"hist_del_{s_id}", type="primary", use_container_width=True):
                            on_delete_session(s_id)
                            st.rerun()
