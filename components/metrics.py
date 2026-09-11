"""
Metric and stat card components for IntelliAssist AI.
Renders clean, modern telemetry tiles with subtle glow and typography.
"""

from typing import Any
import streamlit as st

def render_metric_card(label: str, value: Any, delta: str = "", subtitle: str = "", desc: str = "", *args, **kwargs):
    """Render a modern AI SaaS metric tile."""
    # Defensively resolve subtitle from desc or additional args/kwargs
    final_subtitle = subtitle or desc or (args[0] if len(args) > 0 else "") or kwargs.get("subtitle", "")
    badge_text = delta or kwargs.get("delta", "")
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="metric-label">{label}</span>
            {f'<span style="color:#34d399; font-weight:700; font-size:0.72rem; background:rgba(16,185,129,0.12); padding:2px 8px; border-radius:9999px; border:1px solid rgba(16,185,129,0.25);">{badge_text}</span>' if badge_text else ''}
        </div>
        <div class="metric-value">{value}</div>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">
            {final_subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_grid(doc_count: int, chunk_count: int, questions_count: int, conv_count: int, *args, **kwargs):
    """Render the standard 4-metric dashboard grid."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Indexed Documents", doc_count, "Active", "Managed Corpus")
    with col2:
        render_metric_card("Semantic Chunks", chunk_count, "Indexed", "384-dim Vectors")
    with col3:
        render_metric_card("Queries Answered", questions_count, "Grounded", "RAG Pipeline")
    with col4:
        render_metric_card("Active Sessions", conv_count, "Persisted", "Saved Discussions")
