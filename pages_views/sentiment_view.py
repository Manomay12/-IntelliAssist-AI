"""
Sentiment and Intent Analysis Page View for IntelliAssist AI.
Visualizes emotional tone, user intent categories, confidence metrics, and chunk-by-chunk sentiment trajectories.
"""

from typing import Dict, Any, List, Callable, Optional
import streamlit as st
from components.charts import render_sentiment_donut_chart, render_intent_bar_chart, render_sentiment_trend_chart


def render_sentiment_page(
    all_documents: List[str],
    on_analyze_document: Callable[[str], Dict[str, Any]],
    on_analyze_custom_text: Callable[[str], Dict[str, Any]],
    default_doc: Optional[str] = None,
    *args,
    **kwargs
):
    """Render the Sentiment and Intent Analysis dashboard."""
    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:9999px; background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25); margin-bottom:8px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#8b5cf6; box-shadow:0 0 8px #8b5cf6;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#a78bfa; letter-spacing:0.04em; text-transform:uppercase;">Tone & Intent Analytics</span>
        </div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin:0 0 6px 0;">
            Sentiment & Intent Intelligence
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0; max-width:760px; line-height:1.5;">
            Evaluate sentiment polarity, confidence scores, intent distributions, and narrative tone progression across document passages.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_doc, tab_text = st.tabs(["Analyze Document", "Analyze Custom Text"])

    # Tab 1: Analyze Document
    with tab_doc:
        if not all_documents:
            st.markdown("""
            <div class="modern-card" style="text-align:center; padding:48px 24px;">
                <div style="font-weight:700; font-size:1.15rem; color:#f8fafc; margin-bottom:8px;">No documents available to analyze</div>
                <p style="font-size:0.88rem; color:#94a3b8; max-width:440px; margin:0 auto; line-height:1.6;">
                    Upload or load sample documents from the Documents page to inspect sentiment distributions and tone trajectories.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            default_idx = 0
            if default_doc and default_doc in all_documents:
                default_idx = all_documents.index(default_doc)

            col_select, col_btn = st.columns([4, 1.5])
            with col_select:
                selected_doc = st.selectbox(
                    "Target Document",
                    all_documents,
                    index=default_idx,
                    key="sentiment_doc_select"
                )
            with col_btn:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                run_analysis = st.button("Analyze Tone", type="primary", use_container_width=True, key="btn_run_sent_analysis")

            res = on_analyze_document(selected_doc)
            
            st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

            col_sent, col_intent = st.columns(2)
            with col_sent:
                st.markdown("""
                <div class="modern-card">
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.04em;">Sentiment Polarity Distribution</div>
                """, unsafe_allow_html=True)
                render_sentiment_donut_chart(res.get("sentiment", {}))
                
                label = res.get("sentiment", {}).get("label", "Neutral")
                score = res.get("sentiment", {}).get("score", 0.0)
                st.markdown(f"""
                <div style="text-align:center; font-size:0.85rem; color:#94a3b8; margin-top:12px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">
                    Overall Polarity: <b style="color:#ffffff;">{score:+.2f}</b> &bull; Dominant Tone: <b style="color:#ffffff;">{label}</b>
                </div>
                </div>
                """, unsafe_allow_html=True)

            with col_intent:
                st.markdown("""
                <div class="modern-card">
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.04em;">Intent Classification Breakdown</div>
                """, unsafe_allow_html=True)
                render_intent_bar_chart(res.get("intent", {}))
                
                top_intent = res.get("intent", {}).get("primary_intent", "Informational")
                conf = res.get("intent", {}).get("confidence", 0)
                st.markdown(f"""
                <div style="text-align:center; font-size:0.85rem; color:#94a3b8; margin-top:12px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">
                    Primary Intent: <b style="color:#ffffff;">{top_intent}</b> ({conf}% confidence)
                </div>
                </div>
                """, unsafe_allow_html=True)

            # Trajectory Chart
            st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="modern-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
                    <div>
                        <div style="font-size:0.95rem; font-weight:700; color:#f8fafc;">Passage Tone Trajectory & Narrative Flow</div>
                        <p style="margin:2px 0 0 0; font-size:0.82rem; color:#94a3b8;">Trace how emotional tone and technical optimism evolve from introduction to conclusion.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            trends = res.get("trends", [])
            if trends:
                peak_chunk = max(trends, key=lambda x: x.get("polarity", x.get("score", 0)))
                dip_chunk = min(trends, key=lambda x: x.get("polarity", x.get("score", 0)))

                first_score = trends[0].get("polarity", 0)
                last_score = trends[-1].get("polarity", 0)
                if last_score > first_score + 0.15:
                    arc_text = "Ascending Optimism (Resolves with constructive solutions and positive findings)"
                    arc_color = "#34d399"
                elif last_score < first_score - 0.15:
                    arc_text = "Critical Evaluation (Shifts towards discussion of limitations or risks)"
                    arc_color = "#f87171"
                else:
                    arc_text = "Objective Analytical Arc (Sustained steady objective academic discourse)"
                    arc_color = "#93c5fd"

                st.markdown(f"""
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:14px;">
                    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:10px; padding:10px 14px;">
                        <div style="font-size:0.72rem; color:#34d399; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;">Peak Optimism Section</div>
                        <div style="font-weight:700; color:#ffffff; font-size:0.95rem; margin-top:2px;">{peak_chunk.get('chunk_label', 'Chunk')}</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">Polarity: <b style="color:#34d399;">+{peak_chunk.get('polarity', 0):.2f}</b> &bull; {peak_chunk.get('label')}</div>
                    </div>
                    <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:10px; padding:10px 14px;">
                        <div style="font-size:0.72rem; color:#f87171; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;">Lowest Dip / Critique</div>
                        <div style="font-weight:700; color:#ffffff; font-size:0.95rem; margin-top:2px;">{dip_chunk.get('chunk_label', 'Chunk')}</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">Polarity: <b style="color:#f87171;">{dip_chunk.get('polarity', 0):.2f}</b> &bull; {dip_chunk.get('label')}</div>
                    </div>
                    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:10px; padding:10px 14px;">
                        <div style="font-size:0.72rem; color:#a5b4fc; font-weight:700; text-transform:uppercase; letter-spacing:0.03em;">Narrative Trajectory</div>
                        <div style="font-weight:700; color:{arc_color}; font-size:0.85rem; margin-top:2px; line-height:1.4;">{arc_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            render_sentiment_trend_chart(trends)

            st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-top:12px; padding:8px 14px; background:rgba(7,9,14,0.5); border-radius:8px; font-size:0.78rem; color:#94a3b8; border:1px solid rgba(255,255,255,0.04);">
                <div>Hover over any node to inspect chunk page number, confidence, and excerpt snippet.</div>
                <div style="display:flex; gap:12px; align-items:center;">
                    <span style="display:inline-flex; align-items:center; gap:5px;"><span style="width:7px; height:7px; border-radius:50%; background:#10b981;"></span> Positive / Solution</span>
                    <span style="display:inline-flex; align-items:center; gap:5px;"><span style="width:7px; height:7px; border-radius:50%; background:#94a3b8;"></span> Neutral Context</span>
                    <span style="display:inline-flex; align-items:center; gap:5px;"><span style="width:7px; height:7px; border-radius:50%; background:#ef4444;"></span> Limitations / Risk</span>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 2: Analyze Custom Text
    with tab_text:
        st.markdown("<div style='font-size:0.88rem; color:#94a3b8; margin-bottom:10px;'>Paste or type any passage to test sentiment and intent classification in real time:</div>", unsafe_allow_html=True)
        custom_input = st.text_area(
            "Test text",
            "IntelliAssist AI delivered high performance with grounded retrieval accuracy, though initial ingestion latency could be slightly optimized for large PDF files.",
            height=120,
            key="custom_sentiment_input",
            label_visibility="collapsed"
        )
        if st.button("Evaluate Custom Text", key="btn_eval_custom_text", type="primary"):
            custom_res = on_analyze_custom_text(custom_input)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
                render_sentiment_donut_chart(custom_res.get("sentiment", {}))
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='modern-card'>", unsafe_allow_html=True)
                render_intent_bar_chart(custom_res.get("intent", {}))
                st.markdown("</div>", unsafe_allow_html=True)
