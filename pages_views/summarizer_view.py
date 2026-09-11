"""
Document Intelligence, Summarization & Multi-Perspective Synthesis Page View for IntelliAssist AI.
Generates multi-mode summaries, specialized academic, student, research, and professional analyses,
extracts topics/entities, and performs cross-document side-by-side comparisons.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st

SUMMARY_MODES = [
    "Executive Summary",
    "Detailed Summary",
    "Key Findings",
    "Bullet Points",
    "Quick Summary"
]

STUDENT_FEATURES = [
    "Study Guide",
    "Explain Simply",
    "Explain Technically",
    "Generate Quiz",
    "Flashcards"
]

RESEARCH_FEATURES = [
    "Key Contributions",
    "Problem Statement",
    "Methodology",
    "Limitations"
]

PROFESSIONAL_FEATURES = [
    "Executive Briefing",
    "Decisions",
    "Risks",
    "Action Items"
]


def render_summarizer_page(
    all_documents: List[str],
    on_summarize: Callable[[str, str], Dict[str, Any]],
    on_compare: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    current_summary_data: Optional[Dict[str, Any]] = None,
    default_doc: Optional[str] = None,
    doc_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    summarizer_service: Optional[Any] = None,
    *args,
    **kwargs
):
    """Render the Document Summarizer and Cross-Document Synthesis workspace."""
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:9999px; background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25); margin-bottom:8px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#8b5cf6; box-shadow:0 0 8px #8b5cf6;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#a78bfa; letter-spacing:0.04em; text-transform:uppercase;">Synthesis & Intelligence Engine</span>
        </div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin:0 0 6px 0;">
            Document Intelligence & Synthesis
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0; max-width:760px; line-height:1.5;">
            Extract strategic insights, produce multi-tier summaries, generate academic and research breakdowns, or compare multiple documents side by side.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not all_documents:
        st.markdown("""
        <div class="modern-card" style="text-align:center; padding:48px 24px;">
            <div style="font-weight:700; font-size:1.15rem; color:#f8fafc; margin-bottom:8px;">No documents available in library</div>
            <p style="font-size:0.88rem; color:#94a3b8; max-width:440px; margin:0 auto; line-height:1.6;">
                Upload a document in the Documents view or load sample datasets to unlock multi-perspective AI synthesis, research breakdowns, and comparative analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Tabs: Core Synthesis, Student Mode, Research Mode, Professional Mode, Smart Insights, Comparison
    tab_synth, tab_student, tab_research, tab_pro, tab_insights, tab_compare = st.tabs([
        "Synthesis Modes",
        "Student Mode",
        "Research Mode",
        "Professional Mode",
        "Smart Insights",
        "Cross-Doc Comparison"
    ])

    PLACEHOLDER_DOC = "-- Select a document --"
    doc_options = [PLACEHOLDER_DOC] + all_documents

    # -------------------------------------------------------------
    # TAB 1: Multi-Mode Synthesis
    # -------------------------------------------------------------
    with tab_synth:
        c_doc, c_mode, c_btn = st.columns([3, 2.5, 1.5])
        with c_doc:
            default_idx = 0
            if default_doc and default_doc in doc_options:
                default_idx = doc_options.index(default_doc)

            selected_doc = st.selectbox(
                "Target Document",
                doc_options,
                index=default_idx,
                key="summarizer_doc_select"
            )
        with c_mode:
            selected_mode = st.selectbox(
                "Synthesis Profile",
                SUMMARY_MODES,
                index=0,
                key="summarizer_mode_select"
            )
        with c_btn:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("Synthesize", type="primary", use_container_width=True, key="btn_generate_summary")

        if selected_doc == PLACEHOLDER_DOC:
            if generate_clicked:
                st.warning("Please choose a document from the dropdown above before synthesizing.")
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding:44px 24px; margin-top:20px;">
                <div style="font-size:1.1rem; font-weight:700; color:#f8fafc; margin-bottom:8px;">Select a Document to Begin</div>
                <p style="color:#94a3b8; font-size:0.9rem; max-width:520px; margin:0 auto; line-height:1.6;">
                    Choose any indexed document to generate an executive brief, extract key topics, uncover dominant entities, and produce skimmable bullet points.
                </p>
            </div>
            """, unsafe_allow_html=True)
            summary_data = None
        else:
            summary_data = current_summary_data
            if generate_clicked:
                with st.spinner(f"Synthesizing {selected_mode} for '{selected_doc}'..."):
                    summary_data = on_summarize(selected_doc, selected_mode)
            elif summary_data and summary_data.get("doc_name") == selected_doc and summary_data.get("mode") == selected_mode:
                pass
            else:
                summary_data = None
                st.markdown(f"""
                <div class="modern-card" style="text-align:center; padding:36px 20px; margin-top:16px;">
                    <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom:6px;">Ready to Synthesize: <span style="color:#a5b4fc;">{selected_doc}</span></div>
                    <p style="font-size:0.88rem; color:#94a3b8; max-width:480px; margin:0 auto 16px auto; line-height:1.5;">
                        Profile: <b>{selected_mode}</b>. Click 'Synthesize' above to generate comprehensive structured takeaways.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        if summary_data:
            st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)
            summary_text = summary_data.get("summary", "")
            mode_label = summary_data.get("mode", selected_mode)
            provider_badge = summary_data.get("provider", "IntelliAssist AI")

            md_export = f"# Summary: {selected_doc}\n**Profile:** {mode_label}\n\n{summary_text}\n\n## Key Topics\n" + ", ".join(summary_data.get("topics", []))

            # Action bar
            act1, act2, act_space = st.columns([1.8, 1.5, 5])
            with act1:
                st.download_button(
                    "Download Markdown",
                    data=md_export,
                    file_name=f"summary_{selected_doc[:20]}_{mode_label.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="btn_download_sum"
                )
            with act2:
                if st.button("Copy Summary", key="btn_copy_sum", use_container_width=True):
                    st.toast("Summary copied to clipboard!")

            # Main Summary Card
            st.markdown(f"""
            <div class="modern-card" style="padding:16px 20px; margin:16px 0 12px 0; border-left:3px solid #8b5cf6;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-weight:700; font-size:1.05rem; color:#f8fafc;">{mode_label}</span>
                        <span style="font-size:0.75rem; background:rgba(139,92,246,0.15); color:#c4b5fd; padding:2px 8px; border-radius:6px; font-weight:600;">{selected_doc}</span>
                    </div>
                    <div style="font-size:0.75rem; color:#94a3b8; font-family:monospace;">
                        Engine: <span style="color:#c7d2fe;">{provider_badge}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="modern-card" style="padding:24px 28px; margin-bottom:16px; line-height:1.8; color:#f1f5f9;">
            """, unsafe_allow_html=True)
            st.markdown(summary_text)
            st.markdown("</div>", unsafe_allow_html=True)

            # Topics and Entities Grid
            col_topics, col_entities = st.columns(2)
            with col_topics:
                st.markdown("""
                <div class="modern-card" style="height:100%;">
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.04em;">Key Topics & Concepts</div>
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
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.04em;">Discovered Entities</div>
                """, unsafe_allow_html=True)
                entities_dict = summary_data.get("entities", {})
                has_entities = False
                for cat, items in entities_dict.items():
                    if items:
                        has_entities = True
                        st.markdown(f"<div style='font-size:0.75rem; font-weight:600; color:#a5b4fc; margin-top:6px;'>{cat}:</div>", unsafe_allow_html=True)
                        chips_html = "".join([f"<span class='chip' style='background:rgba(255,255,255,0.05); color:#cbd5e1; border-color:rgba(255,255,255,0.08);'>{item}</span>" for item in items])
                        st.markdown(chips_html, unsafe_allow_html=True)
                if not has_entities:
                    st.markdown("<span style='color:#64748b; font-size:0.85rem;'>General conceptual content.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Helper function to get text and summarizer
    def _get_doc_text(dname: str) -> str:
        if doc_registry and dname in doc_registry:
            return doc_registry[dname].get("full_text", "")
        if "document_registry" in st.session_state and dname in st.session_state.document_registry:
            return st.session_state.document_registry[dname].get("full_text", "")
        return ""

    def _get_summarizer():
        if summarizer_service:
            return summarizer_service
        if "summarizer" in st.session_state:
            return st.session_state.summarizer
        return None

    # -------------------------------------------------------------
    # TAB 2: Student Mode
    # -------------------------------------------------------------
    with tab_student:
        st.markdown("""
        <div style="margin-bottom:16px;">
            <h3 style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:0 0 4px 0;">Student & Study Workspace</h3>
            <p style="font-size:0.85rem; color:#94a3b8; margin:0;">
                Translate complex academic documents into intuitive explanations, study guides, quizzes, and active recall flashcards.
            </p>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([3, 2.5, 1.5])
        with sc1:
            stu_doc = st.selectbox("Document", all_documents, key="stu_doc_sel")
        with sc2:
            stu_feat = st.selectbox("Pedagogy Feature", STUDENT_FEATURES, key="stu_feat_sel")
        with sc3:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            btn_run_student = st.button("Generate", type="primary", use_container_width=True, key="btn_run_stu")

        if btn_run_student or f"stu_res_{stu_doc}_{stu_feat}" in st.session_state:
            if btn_run_student:
                sm = _get_summarizer()
                d_text = _get_doc_text(stu_doc)
                if sm and d_text:
                    with st.spinner(f"Generating {stu_feat} for '{stu_doc}'..."):
                        st.session_state[f"stu_res_{stu_doc}_{stu_feat}"] = sm.generate_student_mode(d_text, stu_feat, stu_doc)
                else:
                    st.warning("Document content not accessible for processing.")

            res = st.session_state.get(f"stu_res_{stu_doc}_{stu_feat}")
            if res:
                st.markdown(f"""
                <div class="modern-card" style="padding:24px 28px; margin-top:16px; border-left:3px solid #06b6d4; line-height:1.7;">
                """, unsafe_allow_html=True)
                st.markdown(res.get("content", ""))
                st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 3: Research Mode
    # -------------------------------------------------------------
    with tab_research:
        st.markdown("""
        <div style="margin-bottom:16px;">
            <h3 style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:0 0 4px 0;">Research & Academic Analysis</h3>
            <p style="font-size:0.85rem; color:#94a3b8; margin:0;">
                Rigorous peer-review breakdown: extract problem statements, methodology pipelines, empirical claims, and documented limitations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns([3, 2.5, 1.5])
        with rc1:
            res_doc = st.selectbox("Document", all_documents, key="res_doc_sel")
        with rc2:
            res_feat = st.selectbox("Research Component", RESEARCH_FEATURES, key="res_feat_sel")
        with rc3:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            btn_run_research = st.button("Extract", type="primary", use_container_width=True, key="btn_run_res")

        if btn_run_research or f"res_out_{res_doc}_{res_feat}" in st.session_state:
            if btn_run_research:
                sm = _get_summarizer()
                d_text = _get_doc_text(res_doc)
                if sm and d_text:
                    with st.spinner(f"Extracting {res_feat} from '{res_doc}'..."):
                        st.session_state[f"res_out_{res_doc}_{res_feat}"] = sm.generate_research_mode(d_text, res_feat, res_doc)
                else:
                    st.warning("Document content not accessible for processing.")

            res = st.session_state.get(f"res_out_{res_doc}_{res_feat}")
            if res:
                st.markdown(f"""
                <div class="modern-card" style="padding:24px 28px; margin-top:16px; border-left:3px solid #8b5cf6; line-height:1.7;">
                """, unsafe_allow_html=True)
                st.markdown(res.get("content", ""))
                st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 4: Professional Mode
    # -------------------------------------------------------------
    with tab_pro:
        st.markdown("""
        <div style="margin-bottom:16px;">
            <h3 style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:0 0 4px 0;">Executive & Enterprise Intelligence</h3>
            <p style="font-size:0.85rem; color:#94a3b8; margin:0;">
                Translate technical reports into strategic business intelligence: decisions log, enterprise risk matrices, and prioritized action roadmaps.
            </p>
        </div>
        """, unsafe_allow_html=True)

        pc1, pc2, pc3 = st.columns([3, 2.5, 1.5])
        with pc1:
            pro_doc = st.selectbox("Document", all_documents, key="pro_doc_sel")
        with pc2:
            pro_feat = st.selectbox("Executive Angle", PROFESSIONAL_FEATURES, key="pro_feat_sel")
        with pc3:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            btn_run_pro = st.button("Generate Brief", type="primary", use_container_width=True, key="btn_run_pro")

        if btn_run_pro or f"pro_out_{pro_doc}_{pro_feat}" in st.session_state:
            if btn_run_pro:
                sm = _get_summarizer()
                d_text = _get_doc_text(pro_doc)
                if sm and d_text:
                    with st.spinner(f"Preparing {pro_feat} for '{pro_doc}'..."):
                        st.session_state[f"pro_out_{pro_doc}_{pro_feat}"] = sm.generate_professional_mode(d_text, pro_feat, pro_doc)
                else:
                    st.warning("Document content not accessible for processing.")

            res = st.session_state.get(f"pro_out_{pro_doc}_{pro_feat}")
            if res:
                st.markdown(f"""
                <div class="modern-card" style="padding:24px 28px; margin-top:16px; border-left:3px solid #10b981; line-height:1.7;">
                """, unsafe_allow_html=True)
                st.markdown(res.get("content", ""))
                st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 5: Smart Insights
    # -------------------------------------------------------------
    with tab_insights:
        st.markdown("""
        <div style="margin-bottom:16px;">
            <h3 style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:0 0 4px 0;">Automated Document Diagnostics</h3>
            <p style="font-size:0.85rem; color:#94a3b8; margin:0;">
                Structural telemetry, reading complexity, dominant metrics, and prioritized action items computed via deterministic NLP.
            </p>
        </div>
        """, unsafe_allow_html=True)

        ic1, ic2 = st.columns([4, 1.5])
        with ic1:
            ins_doc = st.selectbox("Analyze Document", all_documents, key="ins_doc_sel")
        with ic2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            btn_run_ins = st.button("Run Diagnostics", type="primary", use_container_width=True, key="btn_run_ins")

        if btn_run_ins or f"ins_out_{ins_doc}" in st.session_state:
            if btn_run_ins:
                sm = _get_summarizer()
                d_text = _get_doc_text(ins_doc)
                if sm and d_text:
                    with st.spinner(f"Running diagnostics for '{ins_doc}'..."):
                        st.session_state[f"ins_out_{ins_doc}"] = sm.get_smart_insights(d_text, ins_doc)
                else:
                    st.warning("Document content not accessible for processing.")

            ins_data = st.session_state.get(f"ins_out_{ins_doc}")
            if ins_data:
                # Metric tiles
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Reading Time</div>
                        <div class="metric-value">{ins_data.get('reading_time_minutes', 1)} min</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Complexity</div>
                        <div class="metric-value" style="font-size:1.1rem; color:#38bdf8;">{ins_data.get('reading_level', 'General')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Words</div>
                        <div class="metric-value">{ins_data.get('total_words', 0):,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Key Sentences</div>
                        <div class="metric-value">{ins_data.get('total_sentences', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

                # Findings & Actions
                f_col, a_col = st.columns(2)
                with f_col:
                    st.markdown("""
                    <div class="modern-card" style="height:100%;">
                        <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.04em;">Top Empirical Findings</div>
                    """, unsafe_allow_html=True)
                    findings = ins_data.get("key_findings", [])
                    if findings:
                        for f in findings:
                            st.markdown(f"<div style='font-size:0.85rem; color:#cbd5e1; margin-bottom:8px; line-height:1.5;'>• {f}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#64748b; font-size:0.85rem;'>Standard conceptual content.</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with a_col:
                    st.markdown("""
                    <div class="modern-card" style="height:100%;">
                        <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.04em;">Actionable Directives</div>
                    """, unsafe_allow_html=True)
                    actions = ins_data.get("action_items", [])
                    if actions:
                        for a in actions:
                            st.markdown(f"<div style='font-size:0.85rem; color:#cbd5e1; margin-bottom:8px; line-height:1.5;'>→ {a}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#64748b; font-size:0.85rem;'>No explicit action items detected.</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 6: Cross-Document Comparison
    # -------------------------------------------------------------
    with tab_compare:
        st.markdown("""
        <div style="margin-bottom:16px;">
            <h3 style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:0 0 4px 0;">Cross-Document Comparative Matrix</h3>
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
                btn_run_compare = st.button("Compare", type="primary", use_container_width=True, key="btn_run_compare")

            if btn_run_compare or "last_comparison_report" in st.session_state:
                if btn_run_compare and on_compare:
                    with st.spinner(f"Comparing '{doc_a}' vs '{doc_b}'..."):
                        comp_res = on_compare(doc_a, doc_b)
                        st.session_state.last_comparison_report = comp_res
                
                comp_data = st.session_state.get("last_comparison_report")
                if comp_data:
                    st.markdown("<div style='margin:18px 0;'></div>", unsafe_allow_html=True)

                    c_act1, c_act2, c_space = st.columns([2, 1.5, 5])
                    with c_act1:
                        st.download_button(
                            "Download Report",
                            data=comp_data.get("full_report", ""),
                            file_name=f"comparison_{doc_a[:12]}_vs_{doc_b[:12]}.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key="btn_download_comparison"
                        )
                    with c_act2:
                        if st.button("Copy Matrix", key="btn_copy_comp", use_container_width=True):
                            st.toast("Comparison matrix copied to clipboard!")

                    # Display Markdown Comparison Matrix Table
                    st.markdown("#### Side-by-Side Comparison Matrix")
                    st.markdown(comp_data.get("comparison_table", ""))

                    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

                    # Display Executive Synthesis
                    st.markdown("""
                    <div class="modern-card" style="padding:20px 24px; margin-top:14px; border-left:3px solid #8b5cf6;">
                    """, unsafe_allow_html=True)
                    st.markdown(comp_data.get("executive_synthesis", ""))
                    st.markdown("</div>", unsafe_allow_html=True)
