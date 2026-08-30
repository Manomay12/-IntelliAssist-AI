"""
Semantic Search Page View for IntelliAssist AI.
Enables conceptual similarity search across document vectors with highlighted snippets and relevance metrics.
"""

from typing import Dict, Any, List, Callable
import streamlit as st
from utils.helpers import get_relevance_badge_html, get_file_icon, highlight_keywords

def render_search_page(
    all_documents: List[str],
    on_search: Callable[[str, int, float, str], List[Dict[str, Any]]],
    on_ask_about_result: Callable[[str], None]
):
    """Render the Semantic Search page."""
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <h2 style="font-weight:800; color:#ffffff; margin:0; letter-spacing:-0.02em;">🔎 Semantic Vector Search</h2>
            <span style="font-size:0.75rem; background:rgba(14,165,233,0.15); color:#38bdf8; padding:3px 10px; border-radius:9999px; border:1px solid rgba(14,165,233,0.3);">
                Dense Embedding Search
            </span>
        </div>
        <p style="color:#94a3b8; font-size:0.92rem; margin:4px 0 0 0;">
            Find meaning, not just keyword matches. Searches 384-dimensional vector embeddings for conceptual proximity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search Configuration Bar
    col_input, col_doc = st.columns([3.5, 1.5])
    with col_input:
        search_query = st.text_input(
            "Semantic Query",
            placeholder="Search concepts (e.g., 'attention mechanism benchmarks', 'clinical sepsis prediction', 'RAG latency')",
            key="semantic_search_input"
        )
    with col_doc:
        doc_filter = st.selectbox(
            "Document Scope",
            ["All Documents"] + all_documents,
            key="search_doc_filter"
        )

    with st.expander("⚙️ Advanced Search Tuning", expanded=False):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            top_k = st.slider("Max Results (Top-K)", min_value=1, max_value=15, value=5, key="search_top_k")
        with t_col2:
            threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=0.8, value=0.15, step=0.05, key="search_thresh")

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    # Search execution
    if search_query.strip():
        results = on_search(search_query, top_k, threshold, doc_filter)
        
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:14px;">
            Found <b style="color:#f8fafc;">{len(results)}</b> matching chunk(s) across vector index
        </div>
        """, unsafe_allow_html=True)

        if not results:
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding:32px 20px;">
                <div style="font-size:2.5rem; margin-bottom:10px;">🔍</div>
                <div style="font-weight:700; color:#f8fafc; font-size:1.05rem; margin-bottom:4px;">No relevant semantic matches found</div>
                <p style="font-size:0.85rem; color:#94a3b8; max-width:380px; margin:0 auto;">
                    Try lowering the similarity threshold in 'Advanced Tuning' or rephrasing your search query.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for idx, res in enumerate(results, 1):
                filename = res.get("filename", "Document")
                page_num = res.get("page_number", 1)
                score = res.get("score", 0.0)
                badge_html = get_relevance_badge_html(score)
                snippet = highlight_keywords(res.get("text", ""), search_query)
                icon = get_file_icon(filename[filename.rfind("."):]) if "." in filename else "📄"

                with st.container():
                    st.markdown(f"""
                    <div class="modern-card" style="margin-bottom:14px; border-left:4px solid #6366f1;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <div style="display:flex; align-items:center; gap:10px;">
                                <span style="font-size:1.2rem;">{icon}</span>
                                <span style="font-weight:700; color:#f8fafc; font-size:1.02rem;">{filename}</span>
                                <span style="background:rgba(255,255,255,0.08); padding:2px 8px; border-radius:6px; font-size:0.75rem; color:#cbd5e1;">Page {page_num}</span>
                            </div>
                            <div>
                                {badge_html}
                            </div>
                        </div>
                        <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.6; background:rgba(0,0,0,0.25); padding:12px 14px; border-radius:8px; margin-bottom:10px;">
                            {snippet}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"💬 Ask AI About This Context (Result #{idx})", key=f"btn_ask_res_{idx}"):
                        on_ask_about_result(f"Explain the following excerpt from {filename} (Page {page_num}):\n\n\"{snippet}\"")
    else:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:40px 20px;">
            <div style="font-size:3rem; margin-bottom:12px;">🔎</div>
            <h3 style="font-weight:700; color:#f8fafc; font-size:1.15rem; margin:0 0 6px 0;">Enter a search query to explore semantic vectors</h3>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:440px; margin:0 auto;">
                Our embedding model compares semantic similarity rather than simple character matching to discover deep conceptual relationships.
            </p>
        </div>
        """, unsafe_allow_html=True)
