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
    <div style="margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:9999px; background:rgba(6,182,212,0.1); border:1px solid rgba(6,182,212,0.25); margin-bottom:8px;">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#06b6d4; box-shadow:0 0 8px #06b6d4;"></span>
            <span style="font-size:0.75rem; font-weight:700; color:#22d3ee; letter-spacing:0.04em; text-transform:uppercase;">System Configuration</span>
        </div>
        <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin:0 0 6px 0;">
            Settings & Model Parameters
        </h1>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0; max-width:760px; line-height:1.5;">
            Configure AI provider endpoints, API credentials, generation hyperparameters, and index maintenance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. AI Model & Provider Configuration
    st.markdown("""
    <div class="modern-card">
        <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:14px; text-transform:uppercase; letter-spacing:0.04em;">Language Model & Provider Setup</div>
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
            "Model Identifier",
            models_list,
            index=models_list.index(cur_model) if cur_model in models_list else 0,
            key="set_model_select"
        )

    # API Keys (Secured & Hidden for Privacy)
    has_gemini_key = bool(current_settings.get("gemini_api_key") or GEMINI_API_KEY)
    has_nvidia_key = bool(current_settings.get("nvidia_api_key") or NVIDIA_API_KEY)
    has_openai_key = bool(current_settings.get("openai_api_key") or OPENAI_API_KEY)

    if "Gemini" in provider:
        if has_gemini_key:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:10px 14px; margin:12px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                    <span style="font-weight:700; color:#34d399; font-size:0.85rem;">
                        Google Gemini API Key Active (Server Secret Verified)
                    </span>
                    <span style="font-size:0.72rem; background:rgba(16,185,129,0.15); color:#a7f3d0; padding:2px 8px; border-radius:6px; font-weight:600;">
                        Server Secret
                    </span>
                </div>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#94a3b8;">
                    Connected securely via backend environment variables. The raw API key is masked and never exposed to the browser.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.text_input(
                "Custom Gemini Key (Optional Override)",
                value="",
                type="password",
                placeholder="•••••••••••••••••••••••••••••••• (Leave blank to keep server key)",
                help="Your server API key is active. Leave this empty to continue using it, or paste a custom key to override.",
                key="set_gemini_key"
            )
        else:
            st.markdown("""
            <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25); border-radius:10px; padding:10px 14px; margin:12px 0;">
                <div style="font-weight:700; color:#fbbf24; font-size:0.85rem;">No Gemini API Key Configured</div>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#94a3b8;">
                    Enter your Gemini API key below or switch to Smart Demo AI to test RAG and summarization without API keys.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.text_input(
                "Google Gemini API Key",
                value="",
                type="password",
                placeholder="Paste your Gemini API key (e.g. AIza...)",
                key="set_gemini_key"
            )

    elif "NVIDIA" in provider:
        if has_nvidia_key:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:10px 14px; margin:12px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                    <span style="font-weight:700; color:#34d399; font-size:0.85rem;">
                        NVIDIA NIM API Key Active (Server Secret Verified)
                    </span>
                    <span style="font-size:0.72rem; background:rgba(16,185,129,0.15); color:#a7f3d0; padding:2px 8px; border-radius:6px; font-weight:600;">
                        Server Secret
                    </span>
                </div>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#94a3b8;">
                    Connected securely via backend environment variables. The raw API key is masked and never exposed to the browser.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.text_input(
                "Custom NVIDIA Key (Optional Override)",
                value="",
                type="password",
                placeholder="•••••••••••••••••••••••••••••••• (Leave blank to keep server key)",
                help="Your server API key is active. Leave blank to keep the server key.",
                key="set_nvidia_key"
            )
        else:
            st.markdown("""
            <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25); border-radius:10px; padding:10px 14px; margin:12px 0;">
                <div style="font-weight:700; color:#fbbf24; font-size:0.85rem;">No NVIDIA NIM API Key Configured</div>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#94a3b8;">
                    Enter your NVIDIA NIM API key below or switch to Smart Demo AI to test without an API key.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.text_input(
                "NVIDIA NIM / AI API Key",
                value="",
                type="password",
                placeholder="Paste your NVIDIA NIM API key (e.g. nvapi-...)",
                key="set_nvidia_key"
            )

    elif "OpenAI" in provider:
        if has_openai_key:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); border-radius:10px; padding:10px 14px; margin:12px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                    <span style="font-weight:700; color:#34d399; font-size:0.85rem;">
                        OpenAI API Key Active (Server Secret Verified)
                    </span>
                    <span style="font-size:0.72rem; background:rgba(16,185,129,0.15); color:#a7f3d0; padding:2px 8px; border-radius:6px; font-weight:600;">
                        Server Secret
                    </span>
                </div>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#94a3b8;">
                    Connected securely via backend environment variables. The raw API key is masked and never exposed to the browser.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.text_input(
                "Custom OpenAI Key (Optional Override)",
                value="",
                type="password",
                placeholder="•••••••••••••••••••••••••••••••• (Leave blank to keep server key)",
                help="Your server API key is active. Leave blank to keep the server key.",
                key="set_openai_key"
            )
        else:
            st.markdown("""
            <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25); border-radius:10px; padding:10px 14px; margin:12px 0;">
                <div style="font-weight:700; color:#fbbf24; font-size:0.85rem;">No OpenAI API Key Configured</div>
                <p style="margin:4px 0 0 0; font-size:0.78rem; color:#94a3b8;">
                    Enter your OpenAI API key below or switch to Smart Demo AI to test without an API key.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.text_input(
                "OpenAI API Key",
                value="",
                type="password",
                placeholder="Paste your OpenAI key (e.g. sk-...)",
                key="set_openai_key"
            )
    else:
        st.info("Local NLP Engine active. Full RAG Q&A, synthesis, and deep explanations run locally without external API keys.")

    col_temp, col_tok = st.columns(2)
    with col_temp:
        temp = st.slider(
            "Temperature (Sampling Randomness)",
            min_value=0.0,
            max_value=1.0,
            value=float(current_settings.get("temperature", 0.3)),
            step=0.05,
            key="set_temp_slider"
        )
    with col_tok:
        max_tokens = st.slider(
            "Max Generation Tokens",
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
        <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:14px; text-transform:uppercase; letter-spacing:0.04em;">Vector Space & Retrieval Architecture</div>
    """, unsafe_allow_html=True)

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.selectbox(
            "Embedding Model Architecture",
            ["all-MiniLM-L6-v2 (384-dim Dense + BM25 Hybrid)", "Sentence-Transformers / BERT", "Gemini text-embedding-004", "OpenAI text-embedding-3-small"],
            index=0,
            disabled=True,
            key="set_embed_select"
        )
    with v_col2:
        st.selectbox(
            "Vector Database Engine",
            ["Pure NumPy Cosine + BM25Okapi with Reciprocal Rank Fusion", "FAISS Inverted File (Flat)", "ChromaDB Engine"],
            index=0,
            disabled=True,
            key="set_vdb_select"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Data & Storage Lifecycle Management
    st.markdown("""
    <div class="modern-card">
        <div style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:14px; text-transform:uppercase; letter-spacing:0.04em;">Data Storage & Index Maintenance</div>
    """, unsafe_allow_html=True)

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        if st.button("Clear Chat History", use_container_width=True, key="btn_wipe_chats"):
            on_clear_history()
            st.toast("Chat history cleared successfully!")
    with d_col2:
        if st.button("Rebuild Vector Index", use_container_width=True, key="btn_reindex"):
            on_rebuild_index()
            st.toast("Vector index rebuilt from disk!")
    with d_col3:
        if st.button("Wipe All Documents & DB", use_container_width=True, key="btn_wipe_all"):
            on_clear_all_data()
            st.toast("All documents and vector store purged.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Save button
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if st.button("Save Settings", type="primary", key="btn_save_all_settings"):
        custom_gem = st.session_state.get("set_gemini_key", "").strip()
        custom_nvd = st.session_state.get("set_nvidia_key", "").strip()
        custom_oai = st.session_state.get("set_openai_key", "").strip()

        updated_settings = {
            "provider": provider,
            "model_name": model_name,
            "temperature": temp,
            "max_tokens": max_tokens,
            "gemini_api_key": custom_gem if custom_gem else (current_settings.get("gemini_api_key") or GEMINI_API_KEY),
            "nvidia_api_key": custom_nvd if custom_nvd else (current_settings.get("nvidia_api_key") or NVIDIA_API_KEY),
            "openai_api_key": custom_oai if custom_oai else (current_settings.get("openai_api_key") or OPENAI_API_KEY)
        }
        on_save_settings(updated_settings)
        st.toast("Settings saved successfully!")
