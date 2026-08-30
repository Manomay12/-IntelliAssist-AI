"""
Metric and stat card components for IntelliAssist AI.
"""

from typing import Any
import streamlit as st

def render_metric_card(label: str, value: Any, icon: str, delta: str = "", subtitle: str = ""):
    """Render a modern AI SaaS metric tile."""
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="metric-label">{label}</span>
            <span style="font-size:1.3rem;">{icon}</span>
        </div>
        <div class="metric-value">{value}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#94a3b8;">
            <span>{subtitle}</span>
            {f'<span style="color:#10b981; font-weight:600;">{delta}</span>' if delta else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_grid(doc_count: int, chunk_count: int, questions_count: int, conv_count: int):
    """Render the standard 4-metric dashboard grid."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Documents", doc_count, "📄", "+ Active", "Managed Files")
    with col2:
        render_metric_card("Indexed Chunks", chunk_count, "🧩", "Vectors Ready", "384-dim Embeddings")
    with col3:
        render_metric_card("Questions Asked", questions_count, "💬", "+ RAG Powered", "Queries Processed")
    with col4:
        render_metric_card("Conversations", conv_count, "🕘", "Persisted", "Saved Sessions")
