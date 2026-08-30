"""
Source citation card component for IntelliAssist AI.
Renders expandable source context references with relevance metrics and page numbers.
"""

from typing import List, Dict, Any
import streamlit as st
from utils.helpers import get_relevance_badge_html, get_file_icon

def render_sources_section(sources: List[Dict[str, Any]], key_prefix: str = "src"):
    """Render expandable source citation cards beneath an AI response."""
    if not sources:
        return

    with st.expander(f"📚 Sources & Citations ({len(sources)} references)", expanded=False):
        for idx, src in enumerate(sources, 1):
            filename = src.get("filename", "Document")
            page_num = src.get("page_number", 1)
            score = src.get("score", 0.0)
            pct = src.get("similarity_percentage", int(score * 100))
            snippet = src.get("text_snippet", src.get("text", ""))
            icon = get_file_icon(filename[filename.rfind("."):]) if "." in filename else "📄"

            badge_html = get_relevance_badge_html(score)

            st.markdown(f"""
            <div class="source-box" style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="font-weight:700; color:#f8fafc;">{filename}</span>
                        <span style="background:rgba(255,255,255,0.08); padding:2px 8px; border-radius:6px; font-size:0.75rem; color:#cbd5e1;">Page {page_num}</span>
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
                <div style="color:#94a3b8; font-size:0.84rem; line-height:1.5; background:rgba(0,0,0,0.25); padding:10px 12px; border-radius:8px; border-left:3px solid #6366f1;">
                    "{snippet}"
                </div>
            </div>
            """, unsafe_allow_html=True)
