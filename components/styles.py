"""
CSS Design System for IntelliAssist AI.
Implements a premium dark modern AI SaaS workspace inspired by elite research tools:
deep near-black background, subtle grid lines, cyan & violet ambient glow, vector mesh aesthetics,
refined glassmorphic cards, crisp typography, and professional animations.
"""

import streamlit as st

def inject_custom_styles():
    """Inject premium CSS rules into the Streamlit app header."""
    st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Root Variables */
    :root {
        --bg-void: #07090e;
        --bg-surface: rgba(13, 18, 30, 0.72);
        --bg-surface-elevated: rgba(20, 27, 45, 0.85);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(56, 189, 248, 0.4);
        --primary-cyan: #06b6d4;
        --primary-blue: #3b82f6;
        --primary-violet: #8b5cf6;
        --accent-glow: rgba(6, 182, 212, 0.16);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }

    /* Base Typography & Background Atmosphere */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-void) !important;
        color: var(--text-primary) !important;
    }

    /* Subtle Atmospheric Grid & Radial Glow */
    .stApp {
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.10) 0%, transparent 45%),
            radial-gradient(circle at 85% 75%, rgba(139, 92, 246, 0.07) 0%, transparent 40%),
            linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 64px 64px, 64px 64px !important;
        background-position: center top, center top, center center, center center !important;
    }

    /* Streamlit Main Container Tweaks */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 5.5rem !important;
        max-width: 1280px !important;
    }

    /* Custom Modern Minimal Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.4);
    }

    /* Top Action Bar Buttons & Popovers */
    div[data-testid="stPopover"] > div > button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(145deg, #111827 0%, #0b0f19 100%) !important;
        border: 1px solid var(--border-subtle) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        border-radius: 12px !important;
        padding: 9px 16px !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
        white-space: nowrap !important;
        min-height: 40px !important;
        letter-spacing: -0.01em !important;
    }

    div[data-testid="stPopover"] > div > button:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(145deg, #1f293d 0%, #111827 100%) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.25) !important;
    }

    /* Primary Accent Buttons */
    button[kind="primary"],
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #7c3aed 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
    }
    button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 50%, #6d28d9 100%) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 6px 24px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Popover Body Container */
    div[data-testid="stPopoverBody"] {
        background: #0d121f !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.75) !important;
        padding: 18px !important;
        min-width: 320px !important;
    }

    /* Streamlit Form Input Fields */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(13, 18, 30, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(56, 189, 248, 0.7) !important;
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2) !important;
    }

    /* Hero Section Component */
    .hero-container {
        text-align: center;
        padding: 48px 24px 36px 24px;
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.28);
        color: #38bdf8;
        padding: 6px 18px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.1;
        margin: 0 0 16px 0;
        color: #ffffff;
    }

    .hero-gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 auto 32px auto;
        line-height: 1.6;
        letter-spacing: -0.01em;
    }

    /* Dropzone Upload Component */
    .upload-dropzone {
        background: rgba(13, 18, 30, 0.6);
        border: 2px dashed rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 38px 28px;
        text-align: center;
        backdrop-filter: blur(16px);
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: pointer;
        position: relative;
    }

    .upload-dropzone:hover {
        border-color: rgba(56, 189, 248, 0.8);
        background: rgba(13, 18, 30, 0.85);
        box-shadow: 0 12px 32px rgba(6, 182, 212, 0.15);
        transform: translateY(-2px);
    }

    /* Modern Card Component */
    .modern-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 22px 26px;
        backdrop-filter: blur(20px);
        transition: all 0.24s ease-in-out;
        margin-bottom: 18px;
    }

    .modern-card:hover {
        border-color: var(--border-hover);
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.55), 0 0 20px rgba(6, 182, 212, 0.10);
        transform: translateY(-2px);
    }

    /* Intelligence Pipeline Visualizer */
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        margin: 32px 0;
        padding: 24px;
        background: rgba(13, 18, 30, 0.65);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        backdrop-filter: blur(16px);
    }

    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
        position: relative;
        z-index: 2;
    }

    .pipeline-step-node {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: #38bdf8;
        margin-bottom: 8px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
    }

    .pipeline-step:hover .pipeline-step-node {
        border-color: #38bdf8;
        box-shadow: 0 0 16px rgba(6, 182, 212, 0.4);
        transform: scale(1.06);
    }

    .pipeline-step-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .pipeline-step-desc {
        font-size: 0.72rem;
        color: var(--text-muted);
    }

    .pipeline-connector {
        flex: 0.8;
        height: 2px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.4) 0%, rgba(139, 92, 246, 0.4) 100%);
        margin: 0 4px 24px 4px;
    }

    /* Metric Stat Card */
    .metric-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.85) 0%, rgba(11, 15, 25, 0.95) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 16px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.22s ease;
    }

    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.5);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0 2px 0;
    }

    .metric-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
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

    /* High-Visibility User Chat Bubble */
    .chat-bubble-user {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        border: 1px solid rgba(255, 255, 255, 0.16);
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px;
        padding: 15px 20px;
        margin-left: auto;
        max-width: 82%;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3);
        font-size: 0.94rem;
        line-height: 1.6;
    }

    /* High-Visibility AI Chat Bubble */
    .chat-bubble-ai {
        background: rgba(13, 18, 30, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.10);
        color: #f8fafc;
        border-radius: 18px 18px 18px 4px;
        padding: 20px 24px;
        margin-right: auto;
        max-width: 92%;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
        font-size: 0.94rem;
        line-height: 1.68;
    }

    /* Source Citation Cards */
    .source-box {
        background: rgba(10, 14, 24, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.85rem;
    }

    /* Tag and Category Chips */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(6, 182, 212, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.28);
        margin: 2px 4px 2px 0;
    }

    /* Step Item during Indexing */
    .step-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 9px 14px;
        border-radius: 10px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.03);
        font-size: 0.86rem;
    }
    .step-done {
        color: #10b981;
        font-weight: 600;
        border-left: 3px solid #10b981;
    }
    .step-active {
        color: #38bdf8;
        font-weight: 700;
        background: rgba(6, 182, 212, 0.12);
        border-left: 3px solid #38bdf8;
    }

    /* Reduced Motion Accessibility */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }

    /* Footer styling */
    .app-footer {
        text-align: center;
        padding-top: 2.5rem;
        color: var(--text-muted);
        font-size: 0.80rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3.5rem;
        letter-spacing: -0.01em;
    }
    </style>
    """, unsafe_allow_html=True)
