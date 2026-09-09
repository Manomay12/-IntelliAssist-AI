"""
Analytics Dashboard Page View for IntelliAssist AI.
Presents system metrics, token counts, format distributions, and AI usage statistics.
"""

from typing import Dict, Any, List
# pyrefly: ignore [missing-import]
import streamlit as st
from components.metrics import render_metric_card
from components.charts import render_doc_distribution_chart
import pandas as pd

def render_analytics_page(
    doc_infos: List[Dict[str, Any]],
    total_chunks: int,
    total_questions: int,
    total_summaries: int,
    total_searches: int,
    avg_latency: float
):
    """Render the comprehensive system analytics dashboard."""
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <h2 style="font-weight:800; color:#ffffff; margin:0; letter-spacing:-0.02em;">📊 Intelligence & System Analytics</h2>
            <span style="font-size:0.75rem; background:rgba(99,102,241,0.15); color:#a5b4fc; padding:3px 10px; border-radius:9999px; border:1px solid rgba(99,102,241,0.3);">
                Telemetry & Insights
            </span>
        </div>
        <p style="color:#94a3b8; font-size:0.92rem; margin:4px 0 0 0;">
            Track document corpus growth, vector database density, query distributions, and processing latencies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Document & Vector Metrics Row
    st.markdown("<h4 style='color:#f8fafc; font-size:1.05rem; font-weight:700; margin:0 0 12px 0;'>📚 Corpus & Vector Database Status</h4>", unsafe_allow_html=True)
    
    total_pages = sum(d.get("total_pages", 1) for d in doc_infos)
    total_chars = sum(d.get("total_chars", 0) for d in doc_infos)
    total_words = sum(d.get("total_words", 0) for d in doc_infos)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Documents", len(doc_infos), "📄", "Active", "Processed Files")
    with col2:
        render_metric_card("Total Pages", total_pages, "📑", "Normalized", "Extracted Pages")
    with col3:
        render_metric_card("Total Chunks", total_chunks, "🧩", "Stored", "500-char Chunks")
    with col4:
        render_metric_card("Indexed Vectors", total_chunks, "⚡", "384-dim", "Cosine Embeddings")

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # AI Usage Metrics Row
    st.markdown("<h4 style='color:#f8fafc; font-size:1.05rem; font-weight:700; margin:0 0 12px 0;'>🤖 AI & Pipeline Performance</h4>", unsafe_allow_html=True)
    
    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
    with u_col1:
        render_metric_card("Questions Asked", total_questions, "💬", "+ RAG", "Conversational Q&A")
    with u_col2:
        render_metric_card("Summaries Built", total_summaries, "📝", "Multi-Mode", "Executive Briefs")
    with u_col3:
        render_metric_card("Semantic Searches", total_searches, "🔎", "Dense Match", "Concept Lookups")
    with u_col4:
        render_metric_card("Average Latency", f"{avg_latency:.2f}s", "⏱️", "Sub-second", "Inference & Search")

    st.markdown("<div style='margin:28px 0;'></div>", unsafe_allow_html=True)

    # Charts and Table Row
    c_pie, c_table = st.columns([1.8, 2.2])

    with c_pie:
        st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
        # Compute doc extension counts
        doc_type_counts = {}
        for d in doc_infos:
            ext = d.get("file_ext", ".txt").upper().replace(".", "")
            doc_type_counts[ext] = doc_type_counts.get(ext, 0) + 1
            
        if not doc_type_counts:
            doc_type_counts = {"PDF": 1, "DOCX": 1, "TXT": 1}
            
        render_doc_distribution_chart(doc_type_counts)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_table:
        st.markdown("""
        <div class="modern-card">
            <h4 style="margin:0 0 12px 0; font-size:1rem; font-weight:700; color:#f8fafc;">📑 Document Breakdown</h4>
        """, unsafe_allow_html=True)
        
        if doc_infos:
            table_data = []
            for d in doc_infos:
                table_data.append({
                    "Filename": d.get("filename", "Doc"),
                    "Type": d.get("file_ext", "").upper(),
                    "Pages": d.get("total_pages", 1),
                    "Chunks": d.get("chunk_count", 0),
                    "Words": f"{d.get('total_words', 0):,}"
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents uploaded yet. Upload documents to see the breakdown table.")
            
        st.markdown("</div>", unsafe_allow_html=True)
