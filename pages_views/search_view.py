"""
Semantic & Hybrid Search Page View for IntelliAssist AI.
Enables dense vector and BM25 lexical search across document vectors with highlighted snippets and relevance metrics.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st
from utils.helpers import get_relevance_badge_html, get_file_icon, highlight_keywords


def render_search_page(
    all_documents: List[str],
    on_search: Callable[[str, int, float, str], List[Dict[str, Any]]],
    on_ask_about_result: Callable[[str], None],
    doc_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    question_generator: Optional[Any] = None,
    *args,
    **kwargs
):
    """Render the Hybrid & Semantic Vector Search page."""
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:9999px; background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.25); margin-bottom:8px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#22d3ee; letter-spacing:0.04em; text-transform:uppercase;">Hybrid Retrieval Engine</span>
        </div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin:0 0 6px 0;">
            Semantic & Keyword Search
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0; max-width:760px; line-height:1.5;">
            Dual-channel search across your document space. Synthesizes 384-dimensional dense vectors with BM25 Okapi lexical ranking via Reciprocal Rank Fusion.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # If a suggested inquiry was clicked, update the text input's state BEFORE it is instantiated
    if "pending_search_query" in st.session_state and st.session_state.pending_search_query:
        st.session_state.semantic_search_input = st.session_state.pending_search_query
        st.session_state.pending_search_query = None

    # Search Configuration Bar
    col_input, col_doc = st.columns([3.6, 1.4])
    with col_input:
        search_query = st.text_input(
            "Query String",
            placeholder="Search concepts, technical terms, or phrases (e.g., 'attention mechanism benchmarks', 'clinical sepsis prediction')...",
            key="semantic_search_input",
            label_visibility="collapsed"
        )
    with col_doc:
        doc_filter = st.selectbox(
            "Document Scope",
            ["All Documents"] + all_documents,
            key="search_doc_filter",
            label_visibility="collapsed"
        )

    # Manage dynamic suggestion seed
    if "search_sug_seed" not in st.session_state:
        st.session_state.search_sug_seed = 0
    if "search_prev_doc_filter" not in st.session_state:
        st.session_state.search_prev_doc_filter = doc_filter

    if st.session_state.search_prev_doc_filter != doc_filter:
        st.session_state.search_prev_doc_filter = doc_filter
        st.session_state.search_sug_seed += 1

    # Dynamic AI Suggested Questions Row
    if question_generator and (all_documents or (doc_registry and doc_filter in doc_registry)):
        target_text = ""
        if doc_registry and doc_filter in doc_registry:
            target_text = doc_registry[doc_filter].get("full_text", "")

        suggested_queries = question_generator.generate_questions(
            doc_name=doc_filter,
            doc_text=target_text,
            count=4,
            shuffle_seed=st.session_state.search_sug_seed
        )

        st.markdown("<div style='margin:12px 0 6px 0;'></div>", unsafe_allow_html=True)
        col_sug_hdr, col_sug_refresh = st.columns([5.5, 1.2])
        with col_sug_hdr:
            scope_label = f"'{doc_filter}'" if doc_filter != "All Documents" else "All Documents"
            st.markdown(f"""
            <div style="font-size:0.8rem; font-weight:700; color:#a5b4fc; display:flex; align-items:center; gap:6px;">
                <span>Suggested Inquiries for {scope_label}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_sug_refresh:
            if st.button("Shuffle", key=f"btn_shuffle_search_sug_{doc_filter}_{st.session_state.search_sug_seed}", help="Generate fresh exploratory questions for this scope", use_container_width=True):
                st.session_state.search_sug_seed += 1
                st.rerun()

        sug_cols = st.columns(len(suggested_queries))
        for s_idx, (scol, sq) in enumerate(zip(sug_cols, suggested_queries)):
            with scol:
                words = sq.split()
                short_label = words[0] + " " + " ".join(words[1:4])
                if len(short_label) > 28:
                    short_label = short_label[:26] + ".."
                if st.button(short_label, key=f"sug_btn_{s_idx}_{st.session_state.search_sug_seed}_{doc_filter}", help=sq, use_container_width=True):
                    st.session_state.pending_search_query = sq
                    st.rerun()

    with st.expander("Search Tuning & Retrieval Settings", expanded=False):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            top_k = st.slider("Max Passages (Top-K)", min_value=1, max_value=15, value=5, key="search_top_k")
        with t_col2:
            threshold = st.slider("Relevance Cutoff Threshold", min_value=0.0, max_value=0.8, value=0.15, step=0.05, key="search_thresh")

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    # Search execution
    if search_query.strip():
        results = on_search(search_query, top_k, threshold, doc_filter)
        
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <div style="font-size:0.85rem; color:#94a3b8;">
                Retrieved <span style="font-weight:700; color:#f8fafc;">{len(results)}</span> candidate passage(s)
            </div>
            <div style="font-size:0.75rem; color:#64748b; font-family:monospace;">
                Query: "{search_query[:40]}{'...' if len(search_query) > 40 else ''}"
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not results:
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding:36px 20px;">
                <div style="font-weight:700; color:#f8fafc; font-size:1.05rem; margin-bottom:6px;">No relevant passages found</div>
                <p style="font-size:0.85rem; color:#94a3b8; max-width:440px; margin:0 auto; line-height:1.5;">
                    The query did not exceed the relevance threshold. Try lowering the cutoff in 'Search Tuning' or rephrasing your search keywords.
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
                match_type = res.get("match_type", "Semantic Match")
                match_explanation = res.get("match_explanation", "")

                # Match type styling
                if match_type == "Hybrid Match":
                    badge_style = "background:rgba(139,92,246,0.12); color:#c4b5fd; border:1px solid rgba(139,92,246,0.3);"
                elif match_type == "Keyword Match":
                    badge_style = "background:rgba(245,158,11,0.12); color:#fcd34d; border:1px solid rgba(245,158,11,0.3);"
                else:
                    badge_style = "background:rgba(6,182,212,0.12); color:#67e8f9; border:1px solid rgba(6,182,212,0.3);"

                with st.container():
                    st.markdown(f"""
                    <div class="modern-card" style="margin-bottom:14px; border-left:3px solid #06b6d4; transition: transform 0.15s ease;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                <span style="font-weight:700; color:#f8fafc; font-size:0.98rem;">{filename}</span>
                                <span style="background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:6px; font-size:0.72rem; color:#94a3b8; font-weight:600;">
                                    Page {page_num}
                                </span>
                                <span style="{badge_style} padding:2px 8px; border-radius:6px; font-size:0.72rem; font-weight:700; letter-spacing:0.02em;">
                                    {match_type}
                                </span>
                            </div>
                            <div>
                                {badge_html}
                            </div>
                        </div>
                        <div style="color:#cbd5e1; font-size:0.88rem; line-height:1.65; background:rgba(7,9,14,0.6); padding:12px 14px; border-radius:8px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.04);">
                            {snippet}
                        </div>
                        {f'''<div style="font-size:0.72rem; color:#64748b; font-family:monospace; margin-bottom:8px;">
                            {match_explanation}
                        </div>''' if match_explanation else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c_btn1, c_space = st.columns([2.5, 4.5])
                    with c_btn1:
                        if st.button(f"Analyze Passage #{idx} in Chat", key=f"btn_ask_res_{idx}", use_container_width=True):
                            on_ask_about_result(f"Explain the following excerpt from {filename} (Page {page_num}):\n\n\"{snippet}\"")
                    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:48px 24px;">
            <div style="font-weight:700; color:#f8fafc; font-size:1.15rem; margin-bottom:8px;">
                Enter a query to explore semantic and lexical vectors
            </div>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:480px; margin:0 auto; line-height:1.6;">
                Our dual-channel pipeline matches dense vector embeddings for conceptual meaning while evaluating BM25 term frequencies for exact matches.
            </p>
        </div>
        """, unsafe_allow_html=True)
