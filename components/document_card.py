"""
Document card component for IntelliAssist AI.
Renders responsive document cards with metadata badges and direct action buttons.
"""

from typing import Dict, Any, Callable, Optional
import streamlit as st
from utils.helpers import get_file_icon, format_file_size

def render_document_card(
    doc_info: Dict[str, Any],
    on_open: Optional[Callable[[str], None]] = None,
    on_summarize: Optional[Callable[[str], None]] = None,
    on_chat: Optional[Callable[[str], None]] = None,
    on_delete: Optional[Callable[[str], None]] = None,
    key_prefix: str = "doc"
):
    """Render a modern document card with metadata and actions."""
    filename = doc_info.get("filename", "Unknown Document")
    ext = doc_info.get("file_ext", ".txt")
    icon = get_file_icon(ext)
    pages = doc_info.get("total_pages", 1)
    chunks = doc_info.get("chunk_count", 0)
    size_str = format_file_size(doc_info.get("file_size", 0))
    upload_date = doc_info.get("upload_date", "Today")
    status = doc_info.get("status", "Indexed & Ready")
    
    with st.container():
        st.markdown(f"""
        <div class="modern-card" style="padding:16px 20px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="font-size:2rem; background:rgba(255,255,255,0.05); padding:10px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);">
                        {icon}
                    </div>
                    <div>
                        <div style="font-weight:700; font-size:1.02rem; color:#f8fafc; margin-bottom:4px;">{filename}</div>
                        <div style="display:flex; flex-wrap:wrap; gap:10px; font-size:0.78rem; color:#94a3b8;">
                            <span>📑 <b>{pages}</b> pages</span>
                            <span>🧩 <b>{chunks}</b> chunks</span>
                            <span>💾 <b>{size_str}</b></span>
                            <span>🕒 {upload_date}</span>
                        </div>
                    </div>
                </div>
                <div>
                    <span style="font-size:0.75rem; font-weight:600; padding:4px 10px; border-radius:9999px; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3);">
                        ✓ {status}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons row
        col1, col2, col3, col4, col_space = st.columns([1, 1.2, 1, 1, 3.8])
        with col1:
            if st.button("💬 Chat", key=f"{key_prefix}_chat_{filename}", use_container_width=True):
                if on_chat:
                    on_chat(filename)
        with col2:
            if st.button("📝 Summarize", key=f"{key_prefix}_sum_{filename}", use_container_width=True):
                if on_summarize:
                    on_summarize(filename)
        with col3:
            if st.button("👁️ View", key=f"{key_prefix}_view_{filename}", use_container_width=True):
                if on_open:
                    on_open(filename)
        with col4:
            if st.button("🗑️ Delete", key=f"{key_prefix}_del_{filename}", use_container_width=True):
                if on_delete:
                    on_delete(filename)
