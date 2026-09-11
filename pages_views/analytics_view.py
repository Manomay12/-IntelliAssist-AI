"""
Analytics Dashboard Page View for IntelliAssist AI.
Presents system metrics, token counts, format distributions, document category breakdowns, and AI usage statistics.
"""

from typing import Dict, Any, List
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
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:9999px; background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.25); margin-bottom:8px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#22d3ee; letter-spacing:0.04em; text-transform:uppercase;">System Telemetry</span>
        </div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin:0 0 6px 0;">
            Platform Telemetry & Vector Health
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0; max-width:760px; line-height:1.5;">
            Real-time operational observability across vector index density, retrieval latencies, document corpus distributions, and RAG execution counts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Document & Vector Metrics Row
    st.markdown("<div style='color:#f8fafc; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:12px;'>Vector Space & Corpus Density</div>", unsafe_allow_html=True)
    
    total_pages = sum(d.get("total_pages", 1) for d in doc_infos)
    total_chars = sum(d.get("total_chars", 0) for d in doc_infos)
    total_words = sum(d.get("total_words", 0) for d in doc_infos)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Documents", len(doc_infos), "Active Files", "Library", "Managed Files")
    with col2:
        render_metric_card("Total Pages", total_pages, "Normalized", "Pages", "Extracted Pages")
    with col3:
        render_metric_card("Indexed Chunks", total_chunks, "Stored", "Chunks", "Chunk Store")
    with col4:
        render_metric_card("Dense Vectors", total_chunks, "384-dim", "Embeddings", "Cosine Index")

    st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

    # AI Usage Metrics Row
    st.markdown("<div style='color:#f8fafc; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:12px;'>RAG Pipeline Execution Telemetry</div>", unsafe_allow_html=True)
    
    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
    with u_col1:
        render_metric_card("Questions Evaluated", total_questions, "Grounded", "Queries", "Conversational RAG")
    with u_col2:
        render_metric_card("Syntheses Built", total_summaries, "Multi-Tier", "Reports", "Executive Briefs")
    with u_col3:
        render_metric_card("Hybrid Searches", total_searches, "Dual-Engine", "Searches", "RRF Retrievals")
    with u_col4:
        render_metric_card("Avg Latency", f"{avg_latency:.2f}s", "Fast", "Duration", "RAG Pipeline Latency")

    st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

    # Charts and Table Row
    c_pie, c_table = st.columns([1.8, 2.2])

    with c_pie:
        st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.04em;'>Document Formats</div>", unsafe_allow_html=True)
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
            <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.04em;">Document Inventory & Categories</div>
        """, unsafe_allow_html=True)
        
        if doc_infos:
            table_data = []
            for d in doc_infos:
                table_data.append({
                    "Document": d.get("filename", "Doc"),
                    "Category": d.get("category", "General Document"),
                    "Type": d.get("file_ext", "").upper(),
                    "Pages": d.get("total_pages", 1),
                    "Chunks": d.get("chunk_count", 0),
                    "Words": f"{d.get('total_words', 0):,}"
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents uploaded yet. Upload documents to inspect corpus breakdown.")
            
        st.markdown("</div>", unsafe_allow_html=True)
