"""
Interactive Plotly chart components for IntelliAssist AI.
Renders modern analytics, sentiment distributions, intent breakdowns, and chunk trends.
"""

from typing import List, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Custom Dark Palette
DARK_THEME_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#cbd5e1", size=12),
    margin=dict(l=20, r=20, t=35, b=20),
)

def render_sentiment_donut_chart(sentiment_data: Dict[str, Any]):
    """Render a modern donut chart showing sentiment distribution."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment_data.get("positive_score", 0.33) * 100,
        sentiment_data.get("neutral_score", 0.34) * 100,
        sentiment_data.get("negative_score", 0.33) * 100
    ]
    colors = ["#10b981", "#3b82f6", "#ef4444"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.68,
        marker=dict(colors=colors, line=dict(color="#090d16", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#ffffff"),
        hoverinfo="label+value+percent"
    )])
    
    label = sentiment_data.get("label", "Neutral")
    confidence = sentiment_data.get("confidence", 75)
    
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{label}</b><br><span style='font-size:11px; color:#94a3b8;'>{confidence}% Conf</span>",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
            font_color="#ffffff"
        )],
        height=260
    )
    st.plotly_chart(fig, use_container_width=True)

def render_intent_bar_chart(intent_data: Dict[str, Any]):
    """Render horizontal bar chart for intent classification scores."""
    all_intents = intent_data.get("all_intents", {})
    if not all_intents:
        return
        
    df = pd.DataFrame({
        "Intent": list(all_intents.keys()),
        "Confidence (%)": list(all_intents.values())
    }).sort_values("Confidence (%)", ascending=True)

    fig = px.bar(
        df,
        x="Confidence (%)",
        y="Intent",
        orientation="h",
        color="Confidence (%)",
        color_continuous_scale=["#6366f1", "#0ea5e9", "#10b981"]
    )
    
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 100]),
        yaxis=dict(showgrid=False),
        height=260
    )
    st.plotly_chart(fig, use_container_width=True)

def render_sentiment_trend_chart(trend_data: List[Dict[str, Any]]):
    """Render an intuitive sentiment trajectory curve across sequential chunks with distinct sentiment zones."""
    if not trend_data:
        st.info("Upload and analyze a multi-chunk document to see sentiment trends.")
        return

    df = pd.DataFrame(trend_data)
    
    # Ensure all expected columns exist
    if "polarity" not in df.columns and "score" in df.columns:
        df["polarity"] = df["score"]
    if "page_number" not in df.columns:
        df["page_number"] = 1
    if "snippet" not in df.columns:
        df["snippet"] = "No excerpt available"

    fig = go.Figure()

    # 1. Shaded Sentiment Zones
    # Positive Zone (+0.15 to +1.15)
    fig.add_hrect(
        y0=0.15, y1=1.15,
        fillcolor="rgba(16, 185, 129, 0.08)",
        line_width=0,
        annotation_text="🟢 Positive Zone (Optimistic / Solutions / Strengths)",
        annotation_position="top left",
        annotation_font=dict(color="#34d399", size=10)
    )

    # Neutral Zone (-0.15 to +0.15)
    fig.add_hrect(
        y0=-0.15, y1=0.15,
        fillcolor="rgba(148, 163, 184, 0.05)",
        line_width=0,
        annotation_text="⚪ Neutral Zone (Factual / Descriptive / Architecture)",
        annotation_position="top left",
        annotation_font=dict(color="#94a3b8", size=10)
    )

    # Critical / Negative Zone (-1.15 to -0.15)
    fig.add_hrect(
        y0=-1.15, y1=-0.15,
        fillcolor="rgba(239, 68, 68, 0.08)",
        line_width=0,
        annotation_text="🔴 Critical Zone (Challenges / Limitations / Bottlenecks)",
        annotation_position="bottom left",
        annotation_font=dict(color="#f87171", size=10)
    )

    # 2. Boundary and Baseline Lines
    fig.add_hline(y=0.15, line_dash="dot", line_color="rgba(16, 185, 129, 0.25)", line_width=1)
    fig.add_hline(y=-0.15, line_dash="dot", line_color="rgba(239, 68, 68, 0.25)", line_width=1)
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.35)", line_width=1.5)

    # Custom hover data preparation
    custom_data = list(zip(
        df["chunk_label"],
        df["page_number"],
        df["label"],
        df["confidence"],
        df["snippet"]
    ))

    # 3. Main Trajectory Curve with Area Fill
    fig.add_trace(go.Scatter(
        x=df["chunk_index"],
        y=df["polarity"],
        mode="lines+markers",
        name="Chunk Polarity",
        line=dict(color="#818cf8", width=3, shape="spline"),
        marker=dict(
            size=9,
            color=df["color"],
            line=dict(color="#ffffff", width=1.8)
        ),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.08)",
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (Page %{customdata[1]})<br>"
            "Tone: <b>%{customdata[2]}</b> (Polarity: %{y:+.2f})<br>"
            "Confidence: %{customdata[3]}%<br>"
            "<span style='font-size:11px; color:#cbd5e1;'>Excerpt: \"%{customdata[4]}\"</span>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(
            text="<b>Document Emotional & Thematic Arc (Chunk-by-Chunk Progression)</b>",
            font=dict(size=14, color="#f8fafc")
        ),
        xaxis=dict(
            title="Document Narrative Progression (Chunk 1 ➔ End of Document)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            dtick=max(1, len(df) // 10)
        ),
        yaxis=dict(
            title="Tone Polarity Score",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            range=[-1.18, 1.18],
            tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
            ticktext=[
                "-1.0 (Critical)",
                "-0.5 (Concern)",
                "0.0 (Neutral)",
                "+0.5 (Positive)",
                "+1.0 (Optimistic)"
            ]
        ),
        height=360
    )
    st.plotly_chart(fig, use_container_width=True)

def render_doc_distribution_chart(doc_types_dict: Dict[str, int]):
    """Render pie chart of document format distributions."""
    if not doc_types_dict:
        return
        
    labels = list(doc_types_dict.keys())
    values = list(doc_types_dict.values())
    
    colors = ["#ef4444", "#3b82f6", "#10b981", "#8b5cf6"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#090d16", width=2)),
        textinfo="label+value"
    )])
    
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(text="<b>Document Formats</b>", font_size=13),
        height=240,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(fig, use_container_width=True)
