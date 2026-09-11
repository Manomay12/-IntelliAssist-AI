"""
Source citation card component for IntelliAssist AI.
Renders expandable source context references with match classification (Semantic / Keyword / Hybrid),
exact page numbers, and passage snippets.
"""

from typing import List, Dict, Any
import streamlit as st
from utils.helpers import get_relevance_badge_html, get_file_icon

def render_sources_section(sources: List[Dict[str, Any]], key_prefix: str = "src"):
    """Render expandable source citation cards beneath an AI response."""
    if not sources:
        return

    strong_cnt = sum(1 for s in sources if s.get("score", 0) >= 0.70)
    header_title = f"Sources & Evidence ({len(sources)} citations • {strong_cnt} high confidence)"

    with st.expander(f"{header_title}", expanded=False):
        for idx, src in enumerate(sources, 1):
            filename = src.get("filename", "Document")
            page_num = src.get("page_number", 1)
            score = src.get("score", 0.0)
            match_type = src.get("match_type", "Semantic Match")
            match_explanation = src.get("match_explanation", "Retrieved via semantic vector search")
            snippet = src.get("text_snippet", src.get("text", ""))
            icon = get_file_icon(filename[filename.rfind("."):]) if "." in filename else "📄"

            # Badge styling depending on match type
            if match_type == "Hybrid Match":
                type_badge = '<span style="background:rgba(139,92,246,0.15); color:#c084fc; border:1px solid rgba(139,92,246,0.3); padding:2px 8px; border-radius:6px; font-size:0.72rem; font-weight:700;">Hybrid Match</span>'
            elif match_type == "Keyword Match":
                type_badge = '<span style="background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); padding:2px 8px; border-radius:6px; font-size:0.72rem; font-weight:700;">Keyword Match</span>'
            else:
                type_badge = '<span style="background:rgba(6,182,212,0.15); color:#38bdf8; border:1px solid rgba(6,182,212,0.3); padding:2px 8px; border-radius:6px; font-size:0.72rem; font-weight:700;">Semantic Match</span>'

            badge_html = get_relevance_badge_html(score)

            st.markdown(f"""
            <div class="source-box" style="margin-bottom:12px; background:rgba(10,14,24,0.75); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:14px 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span style="font-weight:700; color:#f8fafc; font-size:0.92rem;">{filename}</span>
                        <span style="background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:6px; font-size:0.74rem; color:#cbd5e1;">Page {page_num}</span>
                        {type_badge}
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:8px; font-style:italic;">
                    {match_explanation}
                </div>
                <div style="color:#cbd5e1; font-size:0.86rem; line-height:1.55; background:rgba(0,0,0,0.35); padding:10px 14px; border-radius:8px; border-left:3px solid #38bdf8;">
                    "{snippet}"
                </div>
            </div>
            """, unsafe_allow_html=True)
