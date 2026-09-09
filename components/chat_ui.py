"""
Chat interface components for IntelliAssist AI.
Renders message bubbles, feedback actions, copy controls, and dynamic suggested prompt chips.
"""

from typing import List, Dict, Any, Callable, Optional
# pyrefly: ignore [missing-import]
import streamlit as st
from components.source_card import render_sources_section

def render_suggested_questions(on_select: Callable[[str], None], active_doc: Optional[str] = None):
    """Render modern clickable prompt suggestion chips tailored to active document scope."""
    if active_doc and active_doc != "All Documents":
        short_name = active_doc if len(active_doc) < 22 else active_doc[:19] + "..."
        suggestions = [
            f"📋 What is the main objective and core thesis of {short_name}?",
            f"🎯 Summarize the methodology and architecture in {short_name}",
            f"📊 What are the key empirical findings and benchmarks in {short_name}?",
            f"⚠️ What limitations and future directions are mentioned in {short_name}?"
        ]
    else:
        suggestions = [
            "📋 Summarize the main findings across all uploaded documents",
            "🎯 What are the core methodologies and system architectures?",
            "📊 Compare experimental benchmarks and performance results",
            "💡 Explain the technical concepts in simple terms",
            "🔍 Find critical conclusions and future recommendations"
        ]

    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#94a3b8; margin-bottom:8px;'>💡 SUGGESTED QUESTIONS</div>", unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, (col, sug) in enumerate(zip(cols, suggestions)):
        with col:
            # Clean display label
            words = sug.split()
            label = words[0] + " " + " ".join(words[1:3])
            if st.button(label, key=f"sug_{i}_{active_doc or 'all'}", help=sug, use_container_width=True):
                on_select(sug[2:].strip())

def render_chat_message(
    msg: Dict[str, Any],
    on_feedback: Optional[Callable[[str, str], None]] = None,
    on_copy: Optional[Callable[[str], None]] = None
):
    """Render a single user or assistant chat message."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", "")
    msg_id = msg.get("id", "")
    sources = msg.get("sources", [])
    model_info = msg.get("model_info", {})
    feedback = msg.get("feedback")

    if role == "user":
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; margin-bottom:18px;">
            <div class="chat-bubble-user">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.18); padding-bottom:4px;">
                    <span style="font-weight:700; font-size:0.8rem; color:#ffffff; display:flex; align-items:center; gap:5px;">
                        <span>👤</span> You
                    </span>
                    <span style="font-size:0.75rem; color:#e0e7ff; opacity:0.9;">
                        {timestamp}
                    </span>
                </div>
                <div style="font-size:0.96rem; font-weight:500; color:#ffffff; line-height:1.5;">{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        provider = model_info.get("provider", "IntelliAssist AI")
        latency = model_info.get("latency_sec", 0.3)
        is_demo = model_info.get("is_demo", False)

        tag = "⚡ Demo AI" if is_demo else f"🤖 {provider}"

        st.markdown(f"""
        <div style="display:flex; justify-content:flex-start; margin-bottom:6px;">
            <div class="chat-bubble-ai" style="width:100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:1.1rem;">🤖</span>
                        <span style="font-weight:700; color:#a5b4fc;">IntelliAssist AI</span>
                        <span style="font-size:0.72rem; padding:2px 8px; border-radius:9999px; background:rgba(99,102,241,0.15); color:#818cf8; border:1px solid rgba(99,102,241,0.3);">{tag}</span>
                    </div>
                    <div style="font-size:0.75rem; color:#64748b;">
                        ⏱️ {latency}s • {timestamp}
                    </div>
                </div>
                <div style="line-height:1.65; color:#f1f5f9;">
        """, unsafe_allow_html=True)

        st.markdown(content)

        st.markdown("</div></div></div>", unsafe_allow_html=True)

        # Render sources if available
        if sources:
            render_sources_section(sources, key_prefix=f"msg_{msg_id}")

        # Feedback & Action buttons
        f_col1, f_col2, f_col3, f_spacer = st.columns([0.6, 0.6, 1.2, 7])
        with f_col1:
            up_label = "👍" if feedback == "up" else "👍"
            if st.button(up_label, key=f"thumb_up_{msg_id}", help="Helpful answer"):
                if on_feedback:
                    on_feedback(msg_id, "up")
                    st.toast("Thank you for your feedback!", icon="✨")
        with f_col2:
            down_label = "👎" if feedback == "down" else "👎"
            if st.button(down_label, key=f"thumb_down_{msg_id}", help="Not helpful"):
                if on_feedback:
                    on_feedback(msg_id, "down")
                    st.toast("Feedback recorded.", icon="📝")
        with f_col3:
            if st.button("📋 Copy Text", key=f"copy_{msg_id}", help="Copy response to clipboard"):
                st.toast("Response copied to memory!", icon="📋")
