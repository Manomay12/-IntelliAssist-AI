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
    """Render a sentiment trajectory curve across sequential chunks."""
    if not trend_data:
        st.info("Upload and analyze a multi-chunk document to see sentiment trends.")
        return

    df = pd.DataFrame(trend_data)
    
    fig = go.Figure()
    
    # Add zero baseline
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Neutral Baseline")

    # Add trend line with area fill
    fig.add_trace(go.Scatter(
        x=df["chunk_index"],
        y=df["polarity"],
        mode="lines+markers",
        name="Sentiment Polarity",
        line=dict(color="#6366f1", width=3, shape="spline"),
        marker=dict(size=7, color=df["color"], line=dict(color="#ffffff", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.12)",
        text=df["chunk_label"] + " (" + df["label"] + ")",
        hoverinfo="text+y"
    ))

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(text="<b>Sentiment Trajectory Across Document Chunks</b>", font_size=14),
        xaxis=dict(title="Chunk Sequence", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Polarity Score (-1.0 to +1.0)", showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[-1.1, 1.1]),
        height=300
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
