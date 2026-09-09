"""
Document Summarizer & Comparison Page View for IntelliAssist AI.
Generates multi-mode summaries, extracts topics, discovers key entities, and performs cross-document side-by-side comparisons.
"""

from typing import Dict, Any, List, Callable, Optional
# pyrefly: ignore [missing-import]
import streamlit as st

SUMMARY_MODES = [
    "Executive Summary",
    "Detailed Summary",
    "Key Findings",
    "Bullet Points",
    "Quick Summary"
]

def render_summarizer_page(
    all_documents: List[str],
    on_summarize: Callable[[str, str], Dict[str, Any]],
    on_compare: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    current_summary_data: Optional[Dict[str, Any]] = None,
    default_doc: Optional[str] = None
):
    """Render the Document Summarizer and Cross-Document Comparison workspace."""
    st.markdown("""
    <div style="margin-bottom:18px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <h2 style="font-weight:800; color:#ffffff; margin:0; letter-spacing:-0.02em;">📝 Document Intelligence & Synthesis</h2>
            <span style="font-size:0.75rem; background:rgba(16,185,129,0.15); color:#10b981; padding:3px 10px; border-radius:9999px; border:1px solid rgba(16,185,129,0.3);">
                Multi-Mode Synthesis & Cross-Doc Comparison
            </span>
        </div>
        <p style="color:#94a3b8; font-size:0.92rem; margin:4px 0 0 0;">
            Synthesize complex papers into executive summaries or perform side-by-side cross-document comparisons.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not all_documents:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:36px 20px;">
            <div style="font-size:2.8rem; margin-bottom:12px;">📄</div>
            <div style="font-weight:700; font-size:1.1rem; color:#f8fafc; margin-bottom:6px;">No documents available</div>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:400px; margin:0 auto 16px auto;">
                Please upload a PDF, DOCX, or TXT document on the Documents page or load sample documents to generate AI summaries and comparisons.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Tabs: Single Document Summarizer vs Cross-Document Comparison
    tab_single, tab_compare = st.tabs(["📝 Single Document Summarizer", "⚖️ Cross-Document Comparison"])

    with tab_single:
        # Configuration Controls
        c_doc, c_mode, c_btn = st.columns([3, 2.5, 1.5])
        with c_doc:
            default_idx = 0
            if default_doc and default_doc in all_documents:
                default_idx = all_documents.index(default_doc)

            selected_doc = st.selectbox(
                "Select Document",
                all_documents,
                index=default_idx,
                key="summarizer_doc_select"
            )
        with c_mode:
            selected_mode = st.selectbox(
                "Summary Style & Mode",
                SUMMARY_MODES,
                index=0,
                key="summarizer_mode_select"
            )
        with c_btn:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("✨ Summarize", type="primary", use_container_width=True, key="btn_generate_summary")

        summary_data = current_summary_data
        if generate_clicked or not summary_data or summary_data.get("doc_name") != selected_doc or summary_data.get("mode") != selected_mode:
            with st.spinner(f"Analyzing and generating {selected_mode} for '{selected_doc}'..."):
                summary_data = on_summarize(selected_doc, selected_mode)

        if summary_data:
            st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

            summary_text = summary_data.get("summary", "")
            mode_label = summary_data.get("mode", selected_mode)

            md_export = f"# Summary: {selected_doc}\n**Mode:** {mode_label}\n\n{summary_text}\n\n## Key Topics\n" + ", ".join(summary_data.get("topics", []))

            # Actions Row: Download & Copy
            act1, act2, act_space = st.columns([1.5, 1.5, 5])
            with act1:
                st.download_button(
                    "📥 Download Summary",
                    data=md_export,
                    file_name=f"summary_{selected_doc[:20]}_{mode_label.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="btn_download_sum"
                )
            with act2:
                if st.button("📋 Copy Summary", key="btn_copy_sum", use_container_width=True):
                    st.toast("Summary copied to clipboard!", icon="📋")

            # Main AI Summary Card
            st.markdown(f"""
            <div class="modern-card" style="background:rgba(30, 41, 59, 0.7); border-color:rgba(99, 102, 241, 0.35); padding:24px 28px; margin:16px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:10px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:1.3rem;">📋</span>
                        <span style="font-weight:700; font-size:1.15rem; color:#f8fafc;">{mode_label}</span>
                        <span style="font-size:0.75rem; background:rgba(99,102,241,0.2); color:#a5b4fc; padding:2px 8px; border-radius:6px;">{selected_doc}</span>
                    </div>
                    <div style="font-size:0.75rem; color:#64748b;">
                        Synthesized by IntelliAssist AI
                    </div>
                </div>
                <div style="line-height:1.75; color:#f1f5f9; font-size:0.96rem;">
            """, unsafe_allow_html=True)

            st.markdown(summary_text)
            st.markdown("</div></div>", unsafe_allow_html=True)

            # Topics and Entities Grid
            col_topics, col_entities = st.columns(2)

            with col_topics:
                st.markdown("""
                <div class="modern-card" style="height:100%;">
                    <h4 style="margin:0 0 12px 0; font-size:0.95rem; font-weight:700; color:#f8fafc;">🏷️ Key Topics & Concepts</h4>
                """, unsafe_allow_html=True)

                topics = summary_data.get("topics", [])
                if topics:
                    chips_html = "".join([f"<span class='chip'>#{t}</span>" for t in topics])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#64748b; font-size:0.85rem;'>No dominant topics extracted.</span>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with col_entities:
                st.markdown("""
                <div class="modern-card" style="height:100%;">
                    <h4 style="margin:0 0 12px 0; font-size:0.95rem; font-weight:700; color:#f8fafc;">🔬 Important Entities Discovered</h4>
                """, unsafe_allow_html=True)

                entities_dict = summary_data.get("entities", {})
                has_entities = False
                for cat, items in entities_dict.items():
                    if items:
                        has_entities = True
                        st.markdown(f"<div style='font-size:0.78rem; font-weight:600; color:#a5b4fc; margin-top:6px;'>{cat}:</div>", unsafe_allow_html=True)
                        chips_html = "".join([f"<span class='chip' style='background:rgba(255,255,255,0.05); color:#cbd5e1; border-color:rgba(255,255,255,0.1);'>{item}</span>" for item in items])
                        st.markdown(chips_html, unsafe_allow_html=True)

                if not has_entities:
                    st.markdown("<span style='color:#64748b; font-size:0.85rem;'>General conceptual content.</span>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # Key Takeaways Section
            takeaways = summary_data.get("takeaways", [])
            if takeaways:
                st.markdown("""
                <div class="modern-card" style="margin-top:16px;">
                    <h4 style="margin:0 0 12px 0; font-size:0.95rem; font-weight:700; color:#f8fafc;">🎯 Core Actionable Takeaways</h4>
                """, unsafe_allow_html=True)
                for idx, pt in enumerate(takeaways, 1):
                    st.markdown(f"""
                    <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:8px; font-size:0.88rem; color:#cbd5e1;">
                        <span style="background:#6366f1; color:#ffffff; font-weight:700; border-radius:50%; width:20px; height:20px; display:inline-flex; align-items:center; justify-content:center; font-size:0.72rem; flex-shrink:0;">{idx}</span>
                        <span>{pt}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_compare:
        st.markdown("""
        <div style="margin-bottom:14px;">
            <h3 style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:0 0 4px 0;">⚖️ Cross-Document Comparative Matrix</h3>
            <p style="font-size:0.85rem; color:#94a3b8; margin:0;">
                Select two documents from your indexed library to analyze similarities, differences, metrics, and thematic alignment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if len(all_documents) < 2:
            st.info("Please index at least two documents to enable cross-document comparison.")
        else:
            col_a, col_b, col_c_btn = st.columns([3, 3, 1.5])
            with col_a:
                doc_a = st.selectbox("Document A", all_documents, index=0, key="comp_doc_a_select")
            with col_b:
                default_b_idx = 1 if len(all_documents) > 1 else 0
                doc_b = st.selectbox("Document B", all_documents, index=default_b_idx, key="comp_doc_b_select")
            with col_c_btn:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                btn_run_compare = st.button("⚖️ Compare", type="primary", use_container_width=True, key="btn_run_compare")

            if btn_run_compare or "last_comparison_report" in st.session_state:
                if btn_run_compare and on_compare:
                    with st.spinner(f"Comparing '{doc_a}' vs '{doc_b}'..."):
                        comp_res = on_compare(doc_a, doc_b)
                        st.session_state.last_comparison_report = comp_res
                
                comp_data = st.session_state.get("last_comparison_report")
                if comp_data:
                    st.markdown("<div style='margin:18px 0;'></div>", unsafe_allow_html=True)

                    c_act1, c_act2, c_space = st.columns([1.8, 1.5, 5])
                    with c_act1:
                        st.download_button(
                            "📥 Download Comparison Report",
                            data=comp_data.get("full_report", ""),
                            file_name=f"comparison_{doc_a[:12]}_vs_{doc_b[:12]}.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key="btn_download_comparison"
                        )
                    with c_act2:
                        if st.button("📋 Copy Matrix", key="btn_copy_comp", use_container_width=True):
                            st.toast("Comparison matrix copied!", icon="📋")

                    # Display Markdown Comparison Matrix Table
                    st.markdown("### 📊 Side-by-Side Comparison Matrix")
                    st.markdown(comp_data.get("comparison_table", ""))

                    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

                    # Display Executive Synthesis
                    st.markdown(f"""
                    <div class="modern-card" style="background:rgba(30, 41, 59, 0.7); border-color:rgba(99, 102, 241, 0.4); padding:20px 24px; margin-top:14px;">
                    """, unsafe_allow_html=True)
                    st.markdown(comp_data.get("executive_synthesis", ""))
                    st.markdown("</div>", unsafe_allow_html=True)
