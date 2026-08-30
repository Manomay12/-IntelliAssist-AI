"""
CSS Design System and styling injection for IntelliAssist AI.
Implements a sleek, modern AI SaaS dashboard aesthetic with glassmorphism, rounded cards,
high-contrast action toolbars, custom scrollbars, status badges, and responsive typography.
"""

import streamlit as st

def inject_custom_styles():
    """Inject custom CSS rules into the Streamlit app header."""
    st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Root Variables */
    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --primary-light: rgba(99, 102, 241, 0.15);
        --secondary: #0ea5e9;
        --accent: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #090d16;
        --card-bg: rgba(30, 41, 59, 0.5);
        --card-border: rgba(255, 255, 255, 0.12);
        --card-hover: rgba(255, 255, 255, 0.08);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Base Typography & Body */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Streamlit Main Container Tweaks */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    /* Custom Modern Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.3);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.5);
    }

    /* Top Action Bar Buttons & Popovers - High Visibility */
    div[data-testid="stPopover"] > div > button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        white-space: nowrap !important;
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
    }

    div[data-testid="stPopover"] > div > button:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(145deg, #334155 0%, #1e293b 100%) !important;
        border-color: rgba(99, 102, 241, 0.7) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    }

    /* Primary Accent Buttons */
    button[kind="primary"],
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* Popover Body Container */
    div[data-testid="stPopoverBody"] {
        background: #0f172a !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 14px !important;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.7) !important;
        padding: 16px !important;
        min-width: 320px !important;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(139, 92, 246, 0.14) 50%, rgba(14, 165, 233, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }

    /* Streamlit Bordered Container Custom Styling (Unified Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        backdrop-filter: blur(16px) !important;
        transition: all 0.22s ease-in-out !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.45) !important;
    }

    /* Small compact action buttons inside card containers */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] > button {
        min-height: 32px !important;
        height: 32px !important;
        font-size: 0.76rem !important;
        padding: 2px 6px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] > button:hover {
        background: rgba(99, 102, 241, 0.25) !important;
        border-color: rgba(99, 102, 241, 0.6) !important;
        color: #ffffff !important;
    }

    /* Modern Card Component */
    .modern-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(16px);
        transition: all 0.25s ease-in-out;
        margin-bottom: 16px;
    }
    .modern-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.5);
    }

    /* Control Toolbar Card */
    .toolbar-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
    }

    /* Metric Stat Card */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 14px 18px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2px 0;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    /* Status Dot */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online {
        background: #10b981;
        box-shadow: 0 0 10px #10b981;
    }
    .status-demo {
        background: #0ea5e9;
        box-shadow: 0 0 10px #0ea5e9;
    }

    /* Tag and Topic Chips */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 3px 4px 3px 0;
        transition: all 0.2s ease;
    }
    .chip:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
    }

    /* High-Visibility User Chat Bubble */
    .chat-bubble-user {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin-left: auto;
        max-width: 80%;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35);
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .chat-bubble-user * {
        color: #ffffff !important;
    }

    /* High-Visibility AI Chat Bubble */
    .chat-bubble-ai {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #f8fafc;
        border-radius: 18px 18px 18px 4px;
        padding: 18px 22px;
        margin-right: auto;
        max-width: 90%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* Citation Box */
    .source-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.85rem;
    }

    /* Progress Step Indicator */
    .step-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.02);
        font-size: 0.88rem;
    }
    .step-done {
        color: #10b981;
        font-weight: 600;
    }
    .step-active {
        color: #6366f1;
        font-weight: 700;
        background: rgba(99, 102, 241, 0.1);
    }
    .step-pending {
        color: #64748b;
    }

    /* Footer styling */
    .app-footer {
        text-align: center;
        padding-top: 2rem;
        color: #64748b;
        font-size: 0.78rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)
