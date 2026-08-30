"""
Settings Page View for IntelliAssist AI.
Manages AI model configurations, API keys (Google Gemini, NVIDIA NIM, OpenAI), vector DB parameters, and data lifecycle management.
"""

from typing import Dict, Any, Callable
import streamlit as st
from utils.config import (
    AVAILABLE_PROVIDERS,
    AVAILABLE_GEMINI_MODELS,
    AVAILABLE_NVIDIA_MODELS,
    AVAILABLE_OPENAI_MODELS,
    GEMINI_API_KEY,
    NVIDIA_API_KEY,
    OPENAI_API_KEY,
    APP_VERSION
)

def render_settings_page(
    current_settings: Dict[str, Any],
    on_save_settings: Callable[[Dict[str, Any]], None],
    on_clear_history: Callable[[], None],
    on_rebuild_index: Callable[[], None],
    on_clear_all_data: Callable[[], None]
):
    """Render the Settings page."""
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <h2 style="font-weight:800; color:#ffffff; margin:0; letter-spacing:-0.02em;">⚙️ System & AI Settings</h2>
            <span style="font-size:0.75rem; background:rgba(99,102,241,0.15); color:#a5b4fc; padding:3px 10px; border-radius:9999px; border:1px solid rgba(99,102,241,0.3);">
                Configuration
            </span>
        </div>
        <p style="color:#94a3b8; font-size:0.92rem; margin:4px 0 0 0;">
            Configure Google Gemini, NVIDIA NIM, OpenAI, hyperparameter temperatures, and vector embeddings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. AI Model & Provider Configuration
    st.markdown("""
    <div class="modern-card">
        <h3 style="margin:0 0 14px 0; font-size:1.1rem; font-weight:700; color:#f8fafc;">🤖 AI Model & Provider Setup</h3>
    """, unsafe_allow_html=True)

    c_prov, c_model = st.columns(2)
    with c_prov:
        cur_prov = current_settings.get("provider", "Google Gemini")
        provider = st.selectbox(
            "LLM Provider",
            AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(cur_prov) if cur_prov in AVAILABLE_PROVIDERS else 0,
            key="set_provider_select"
        )
    with c_model:
        if "Gemini" in provider:
            models_list = AVAILABLE_GEMINI_MODELS
        elif "NVIDIA" in provider:
            models_list = AVAILABLE_NVIDIA_MODELS
        elif "OpenAI" in provider:
            models_list = AVAILABLE_OPENAI_MODELS
        else:
            models_list = ["Academic-Local-NLP-v3 (Exhaustive Grounded)"]
            
        cur_model = current_settings.get("model_name", models_list[0])
        model_name = st.selectbox(
            "Model Name",
            models_list,
            index=models_list.index(cur_model) if cur_model in models_list else 0,
            key="set_model_select"
        )

    # API Keys
    if "Gemini" in provider:
        st.text_input(
            "Google Gemini API Key",
            value=current_settings.get("gemini_api_key", GEMINI_API_KEY),
            type="password",
            placeholder="AQ...",
            help="Google Gemini API Key is active",
            key="set_gemini_key"
        )
        st.success("🟢 Google Gemini API Connected & Ready.")
    elif "NVIDIA" in provider:
        st.text_input(
            "NVIDIA NIM / AI API Key",
            value=current_settings.get("nvidia_api_key", NVIDIA_API_KEY),
            type="password",
            placeholder="nvapi-...",
            help="NVIDIA NIM API Key is active",
            key="set_nvidia_key"
        )
        st.success("🟢 NVIDIA NIM / AI API Key Connected & Ready.")
    elif "OpenAI" in provider:
        st.text_input(
            "OpenAI API Key",
            value=current_settings.get("openai_api_key", OPENAI_API_KEY),
            type="password",
            placeholder="sk-...",
            help="Get your key from OpenAI Platform",
            key="set_openai_key"
        )
    else:
        st.info("💡 **Smart Demo AI is active.** You can test full RAG Q&A, summaries, and deep explanations without any API keys!")

    col_temp, col_tok = st.columns(2)
    with col_temp:
        temp = st.slider(
            "Temperature (Creativity vs. Factuality)",
            min_value=0.0,
            max_value=1.0,
            value=float(current_settings.get("temperature", 0.3)),
            step=0.05,
            key="set_temp_slider"
        )
    with col_tok:
        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=256,
            max_value=4096,
            value=int(current_settings.get("max_tokens", 2048)),
            step=128,
            key="set_tokens_slider"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Embeddings & Vector Database
    st.markdown("""
    <div class="modern-card">
        <h3 style="margin:0 0 14px 0; font-size:1.1rem; font-weight:700; color:#f8fafc;">🧩 Vector Database & Embedding Model</h3>
    """, unsafe_allow_html=True)

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.selectbox(
            "Embedding Model Architecture",
            ["TF-IDF + Neural Dense (Hybrid 384-dim)", "BERT / Sentence-Transformers", "Gemini text-embedding-004", "OpenAI text-embedding-3-small"],
            index=0,
            disabled=True,
            key="set_embed_select"
        )
    with v_col2:
        st.selectbox(
            "Vector Database Engine",
            ["High-Speed Cosine Memory + Persistent Store", "FAISS Inverted File (Flat)", "ChromaDB Engine"],
            index=0,
            disabled=True,
            key="set_vdb_select"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Data & Storage Lifecycle Management
    st.markdown("""
    <div class="modern-card">
        <h3 style="margin:0 0 14px 0; font-size:1.1rem; font-weight:700; color:#f8fafc;">🗑️ Data & Maintenance Actions</h3>
    """, unsafe_allow_html=True)

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        if st.button("🧹 Clear Chat History", use_container_width=True, key="btn_wipe_chats"):
            on_clear_history()
            st.toast("Chat history cleared successfully!", icon="🧹")
    with d_col2:
        if st.button("🔄 Rebuild Vector Index", use_container_width=True, key="btn_reindex"):
            on_rebuild_index()
            st.toast("Vector index rebuilt from disk!", icon="🔄")
    with d_col3:
        if st.button("⚠️ Wipe All Documents & DB", use_container_width=True, key="btn_wipe_all"):
            on_clear_all_data()
            st.toast("All documents and database wiped.", icon="🗑️")

    st.markdown("</div>", unsafe_allow_html=True)

    # Save button
    if st.button("💾 Save All Settings", type="primary", key="btn_save_all_settings"):
        updated_settings = {
            "provider": provider,
            "model_name": model_name,
            "temperature": temp,
            "max_tokens": max_tokens,
            "gemini_api_key": st.session_state.get("set_gemini_key", GEMINI_API_KEY),
            "nvidia_api_key": st.session_state.get("set_nvidia_key", NVIDIA_API_KEY),
            "openai_api_key": st.session_state.get("set_openai_key", OPENAI_API_KEY)
        }
        on_save_settings(updated_settings)
        st.toast("Settings saved successfully!", icon="✅")
