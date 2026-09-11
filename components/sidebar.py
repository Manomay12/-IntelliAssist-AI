"""
Sidebar navigation, quick chat controls, and conversation history manager for IntelliAssist AI.
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
import streamlit as st
from utils.config import APP_NAME, APP_TAGLINE, APP_VERSION

NAV_ITEMS = [
    ("Overview", "Dashboard"),
    ("Documents", "Documents"),
    ("AI Chat", "AI Chat"),
    ("Semantic Search", "Semantic Search"),
    ("Intelligence & Modes", "Summarizer"),
    ("Tone & Sentiment", "Sentiment & Intent"),
    ("Analytics", "Analytics"),
    ("History", "History"),
    ("Settings", "Settings")
]

def render_sidebar(
    active_nav: str,
    total_docs: int,
    total_chunks: int,
    provider_name: str,
    is_api_connected: bool,
    recent_sessions: Optional[List[Dict[str, Any]]] = None,
    current_session_id: Optional[str] = None,
    on_load_session: Optional[Callable[[str], None]] = None,
    on_delete_session: Optional[Callable[[str], None]] = None,
    on_new_chat: Optional[Callable[[], None]] = None,
    on_clear_active_chat: Optional[Callable[[], None]] = None,
    on_navigate: Optional[Callable[[str], None]] = None
) -> str:
    """Render the enhanced sidebar with modern navigation, quick actions, and recent chat history."""
    with st.sidebar:
        # App Logo & Branding Header
        st.markdown(f"""
        <div style="padding:10px 4px 18px 4px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; box-shadow:0 4px 14px rgba(6,182,212,0.35);">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                    </svg>
                </div>
                <div>
                    <div style="font-weight:800; font-size:1.15rem; color:#ffffff; letter-spacing:-0.03em; line-height:1.1;">{APP_NAME}</div>
                    <div style="font-size:0.72rem; color:#38bdf8; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">DOCUMENT INTELLIGENCE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main Navigation
        labels = [item[0] for item in NAV_ITEMS]
        raw_keys = [item[1] for item in NAV_ITEMS]
        default_idx = raw_keys.index(active_nav) if active_nav in raw_keys else 0

        selected_label = st.radio(
            "Navigation",
            labels,
            index=default_idx,
            label_visibility="collapsed",
            key=f"sidebar_nav_{active_nav}"
        )
        selected_page = raw_keys[labels.index(selected_label)]

        st.markdown("<div style='margin-top:14px; border-bottom:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

        # Quick Chat Actions Section
        st.markdown("<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin:12px 0 8px 0;'>CHAT ACTIONS</div>", unsafe_allow_html=True)
        col_new, col_clr = st.columns(2)
        with col_new:
            if st.button("New Chat", key="sb_btn_new_chat", use_container_width=True, help="Start a fresh conversation"):
                if on_new_chat:
                    on_new_chat()
        with col_clr:
            with st.popover("Clear Chat", use_container_width=True, help="Clear active conversation messages"):
                st.markdown("<p style='font-size:0.85rem; color:#f8fafc; margin:0 0 8px 0;'>Clear all messages in active session?</p>", unsafe_allow_html=True)
                if st.button("Confirm Clear", key="sb_btn_confirm_clear", type="primary", use_container_width=True):
                    if on_clear_active_chat:
                        on_clear_active_chat()

        # Recent Chat History in Sidebar
        sessions = recent_sessions or []
        st.markdown("<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin:14px 0 8px 0;'>RECENT CHATS</div>", unsafe_allow_html=True)

        if not sessions:
            st.markdown("""
            <div style="font-size:0.78rem; color:#64748b; padding:10px 8px; background:rgba(255,255,255,0.015); border-radius:10px; border:1px dashed rgba(255,255,255,0.06); text-align:center;">
                No saved chats yet
            </div>
            """, unsafe_allow_html=True)
        else:
            for s in sessions[:5]:
                s_id = s.get("session_id", "")
                title = s.get("title", "Conversation")
                if len(title) > 22:
                    title = title[:20] + "..."
                msg_count = s.get("message_count", 0)
                is_active = (s_id == current_session_id)
                active_prefix = "• " if is_active else ""

                c_item, c_del = st.columns([4.2, 1])
                with c_item:
                    btn_label = f"{active_prefix}{title} ({msg_count})"
                    if st.button(btn_label, key=f"sb_hist_load_{s_id}", help=f"Load chat: {s.get('title', '')}", use_container_width=True):
                        if on_load_session:
                            on_load_session(s_id)
                with c_del:
                    if st.button("✕", key=f"sb_hist_del_{s_id}", help="Delete this conversation"):
                        if on_delete_session:
                            on_delete_session(s_id)

            if len(sessions) > 0:
                if st.button(f"View All History ({len(sessions)})", key="sb_view_all_history", use_container_width=True):
                    if on_navigate:
                        on_navigate("History")

        st.markdown("<div style='margin-top:14px; border-bottom:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

        # Live System Telemetry
        st.markdown("<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin:12px 0 8px 0;'>SYSTEM STATUS</div>", unsafe_allow_html=True)

        status_class = "status-online" if is_api_connected else "status-demo"
        api_text = f"{provider_name}" if is_api_connected else "Smart NLP Mode"

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px 12px; font-size:0.78rem; line-height:1.75;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#94a3b8;">Generation:</span>
                <span style="font-weight:600; color:#f8fafc; display:flex; align-items:center;">
                    <span class="status-dot {status_class}"></span> {api_text}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#94a3b8;">Retrieval:</span>
                <span style="font-weight:600; color:#38bdf8;">Hybrid (Dense + BM25)</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#94a3b8;">Vector Index:</span>
                <span style="font-weight:600; color:#10b981;">Online & Ready</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#94a3b8;">Active Library:</span>
                <span style="font-weight:600; color:#f8fafc;">{total_docs} doc(s) • {total_chunks} chunks</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px; border-bottom:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)

        # Version stamp
        st.markdown(f"""
        <div style="text-align:center; padding:10px 0 4px 0; color:#64748b; font-size:0.72rem;">
            {APP_NAME} v{APP_VERSION} • Production RAG
        </div>
        """, unsafe_allow_html=True)

    return selected_page
