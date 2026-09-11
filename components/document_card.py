"""
Document card component for IntelliAssist AI.
Renders unified, responsive document cards with classification tags, telemetry metrics, and integrated actions.
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
    key_prefix: str = "doc",
    *args,
    **kwargs
):
    """Render a modern document card with metadata, classification, and integrated actions."""
    filename = doc_info.get("filename", "Unknown Document")
    ext = doc_info.get("file_ext", ".txt")
    icon = get_file_icon(ext)
    pages = doc_info.get("total_pages", 1)
    chunks = doc_info.get("chunk_count", 0)
    size_str = format_file_size(doc_info.get("file_size", 0))
    upload_date = doc_info.get("upload_date", "Recently")
    status = doc_info.get("status", "Indexed")
    category = doc_info.get("category", "General Document")
    
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:14px; min-width:0;">
                <div style="font-size:1.5rem; background:rgba(6,182,212,0.12); padding:8px 12px; border-radius:12px; border:1px solid rgba(6,182,212,0.25); flex-shrink:0;">
                    {icon}
                </div>
                <div style="min-width:0;">
                    <div style="font-weight:700; font-size:0.98rem; color:#f8fafc; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-bottom:4px;" title="{filename}">
                        {filename}
                    </div>
                    <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; font-size:0.76rem; color:#94a3b8;">
                        <span style="background:rgba(6,182,212,0.12); color:#38bdf8; padding:2px 8px; border-radius:6px; font-weight:600; border:1px solid rgba(6,182,212,0.25);">{category}</span>
                        <span style="background:rgba(255,255,255,0.04); padding:2px 7px; border-radius:6px; border:1px solid rgba(255,255,255,0.07);">{pages} pages</span>
                        <span style="background:rgba(255,255,255,0.04); padding:2px 7px; border-radius:6px; border:1px solid rgba(255,255,255,0.07);">{chunks} chunks</span>
                        <span style="background:rgba(255,255,255,0.04); padding:2px 7px; border-radius:6px; border:1px solid rgba(255,255,255,0.07);">{size_str}</span>
                        <span style="color:#64748b;">Indexed {upload_date}</span>
                    </div>
                </div>
            </div>
            <div style="flex-shrink:0; margin-left:8px;">
                <span style="font-size:0.72rem; font-weight:600; padding:4px 10px; border-radius:9999px; background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.28);">
                    {status}
                </span>
            </div>
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.06); margin:8px 0 10px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Action buttons row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Chat", key=f"{key_prefix}_chat_{filename}", use_container_width=True):
                if on_chat:
                    on_chat(filename)
        with col2:
            if st.button("Insights", key=f"{key_prefix}_sum_{filename}", use_container_width=True):
                if on_summarize:
                    on_summarize(filename)
        with col3:
            if st.button("View", key=f"{key_prefix}_view_{filename}", use_container_width=True):
                if on_open:
                    on_open(filename)
        with col4:
            if st.button("Delete", key=f"{key_prefix}_del_{filename}", use_container_width=True):
                if on_delete:
                    on_delete(filename)
