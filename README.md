# IntelliAssist AI — Smart Document AI Assistant

> **Major Project — Artificial Intelligence & Natural Language Processing**  
> *A production-style, enterprise AI document intelligence platform built with Python, Streamlit, LangChain-style RAG pipelines, dense 384-dimensional vector embeddings, and multi-provider Generative AI.*

---

## 📌 1. Project Overview & Problem Statement

Students, enterprise analysts, and researchers struggle to manage, navigate, and synthesize information across hundreds of pages of PDFs, technical project reports, and lecture notes. Traditional keyword search (Ctrl+F) fails to capture semantic nuance, synonymy, or conceptual relationships.

**IntelliAssist AI** addresses this challenge by providing an end-to-end conversational AI document intelligence system. Leveraging modern **Retrieval-Augmented Generation (RAG)**, dense vector embeddings, and persistent cosine vector indexing, it enables users to upload multiple documents, ask natural language questions, receive factually grounded answers with page-level source citations, generate multi-mode executive summaries, analyze sentiment & user intent, and explore deep semantic vector search.

---

## 🏛️ 2. Core System Architecture

The end-to-end pipeline follows a modular 7-stage architecture:

```
┌──────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│  Documents   │ ───► │ 1. Text Extraction      │ ───► │ 2. Text Normalization│
│(PDF/DOCX/TXT)│      │  (pypdf / python-docx)  │      │    & Page Tracking   │
└──────────────┘      └─────────────────────────┘      └──────────────────────┘
                                                                  │
                                                                  ▼
┌──────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│ 5. Cosine DB │ ◄─── │ 4. Embedding Generator  │ ◄─── │ 3. Recursive Chunker │
│(Persistent)  │      │  (Hybrid 384-dim Dense) │      │  (500 char / 100 ov) │
└──────────────┘      └─────────────────────────┘      └──────────────────────┘
       │
       │  Semantic Query Embedding & Cosine Similarity
       ▼
┌──────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│ 6. Top-K RAG │ ───► │ 7. LLM Response Builder │ ───► │ Verified Answer with │
│ Context Rank │      │ (Gemini / OpenAI / Demo)│      │ Exact Source Citation│
└──────────────┘      └─────────────────────────┘      └──────────────────────┘
```

---

## ✨ 3. Key Features

- **🏠 Interactive Dashboard**: Live metric counters, Quick Ask AI widget, Recent Documents cards with instant actions, and real-time AI activity logs.
- **📄 Multi-Format Ingestion**: Drag-and-drop support for **PDF**, **DOCX**, and **TXT** files with an animated 7-step visual indexing pipeline and 1-click sample document loader.
- **💬 Conversational RAG Chat**: Modern conversational interface with streaming simulation, markdown tables, code syntax highlighting, copy-to-clipboard, feedback buttons, and expandable page citations with percentage match badges.
- **🔎 Semantic Vector Search**: Dedicated conceptual search engine that queries dense 384-dimensional vector embeddings with interactive snippet highlighting and relevance thresholds.
- **📝 Multi-Mode Summarizer**: 5 distinct summary modes (*Quick Summary, Detailed Summary, Bullet Points, Key Findings, Executive Summary*) along with automated key topic extraction, categorized entity discovery, and downloadable Markdown briefs.
- **🧠 Sentiment & Intent Intelligence**: Visual sentiment donut chart, multi-class user intent classification (Informational, Question, Request, Complaint, Opinion, Technical), and chunk-by-chunk sentiment trajectory line charts.
- **📊 Analytics Telemetry**: Corpus metrics, token distributions, file type breakdowns, query latencies, and structured data tables.
- **🕘 Conversation History**: Persisted chat sessions with resume, rename, search filter, and deletion capabilities.
- **⚙️ Settings & Demo Mode**: Full control over LLM providers (Google Gemini, OpenAI, Hugging Face, or Offline Smart Demo AI), temperature, token limits, and vector database maintenance.

---

## 🛠️ 4. Technology Stack

- **Frontend / UI**: Streamlit with custom CSS design system (Dark Mode, Glassmorphism, Rounded Cards, Custom Badges)
- **Programming Language**: Python 3.11 / 3.13
- **Document Processing**: `pypdf`, `python-docx`
- **NLP & Embeddings**: Hybrid Dense 384-dimensional semantic projection + TF-IDF Vectorization (`scikit-learn`, `numpy`)
- **Vector Database**: High-speed Cosine Similarity Vector Index with JSON/NumPy persistence
- **Generative AI / LLMs**: Google Gemini (`google-genai`), OpenAI (`openai`), and Smart Local Demo NLP Engine
- **Data Visualizations**: `plotly`, `pandas`

---

## 🚀 5. Installation & Quick Start

### Prerequisites
- Python 3.10+ or `uv` package manager installed.

### Step 1: Clone or Navigate to Directory
```bash
cd intelliassist_ai
```

### Step 2: Create Virtual Environment
```bash
# Using uv (Recommended)
uv venv .venv
# Or standard Python
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
# (Linux / macOS)
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Using uv
uv pip install -r requirements.txt
# Or standard pip
pip install -r requirements.txt
```

### Step 4: Configure API Keys (Optional)
Copy `.env.example` to `.env` if you want to use your personal Gemini or OpenAI keys:
```bash
cp .env.example .env
```
*(Note: If no API key is provided, the application runs automatically in **Smart Demo Mode** with 100% functionality).*

### Step 5: Launch Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🎓 6. College Project 3–5 Minute Demonstration Flow

For an optimal presentation to the evaluator:

1. **Dashboard (0:00 - 0:45)**: Show the modern AI SaaS dashboard, metric tiles, and explain the problem statement.
2. **Documents & Ingestion (0:45 - 1:30)**: Click **"Load Instant Academic Samples"** on the Documents page. Point out the animated 7-step indexing pipeline and document library statistics.
3. **AI Chat with RAG (1:30 - 2:45)**: Navigate to **AI Chat**. Ask: *"What are the core findings and benchmarks of the research paper?"*. Show the generated answer, copy text, thumbs-up feedback, and click to expand the **Sources & Citations** box showing exact page numbers and similarity match percentages.
4. **Semantic Search (2:45 - 3:15)**: Go to **Semantic Search**, search for *"attention mechanism"*, and show how dense vectors find conceptual matches with highlighted snippets.
5. **Summarizer & Sentiment (3:15 - 4:15)**: Open **Summarizer** to generate an *Executive Summary* with entity chips. Switch to **Sentiment & Intent** to show the sentiment donut, intent classification, and chunk-by-chunk tone trajectory chart.
6. **Analytics & Settings (4:15 - 5:00)**: Show the telemetry charts and explain the multi-provider LLM architecture in Settings.

---

## 📁 7. Project Directory Structure

```
intelliassist_ai/
├── app.py                      # Main entrypoint, page routing & state management
├── pages_views/
│   ├── dashboard_view.py       # 🏠 Dashboard
│   ├── documents_view.py       # 📄 Document Upload & Library
│   ├── chat_view.py            # 💬 AI Chat & RAG
│   ├── search_view.py          # 🔎 Semantic Search
│   ├── summarizer_view.py      # 📝 Multi-Mode Summarizer
│   ├── sentiment_view.py       # 🧠 Sentiment & Intent Intelligence
│   ├── analytics_view.py       # 📊 System Analytics
│   ├── history_view.py         # 🕘 Conversation History
│   └── settings_view.py        # ⚙️ Settings & LLM Config
├── components/
│   ├── styles.py               # Modern AI SaaS CSS Design System
│   ├── sidebar.py              # Navigation bar & system status monitor
│   ├── chat_ui.py              # Chat bubbles, suggested prompts, copy actions
│   ├── document_card.py        # Document cards with quick action buttons
│   ├── source_card.py          # Expandable source citation references
│   ├── metrics.py              # Stats counter tiles
│   └── charts.py               # Plotly charts (donut, intent, trend, formats)
├── services/
│   ├── document_processor.py   # PDF, DOCX, TXT extraction & cleaning
│   ├── chunker.py              # Recursive overlapping text splitting
│   ├── embeddings.py           # Hybrid Dense 384-dim semantic embeddings
│   ├── vector_store.py         # Cosine vector database with disk persistence
│   ├── rag_engine.py           # RAG retrieval, context ranking & answer generation
│   ├── llm_service.py          # Multi-provider LLM interface (Gemini/OpenAI/Demo)
│   ├── summarizer.py           # Summarizer, topic modeling & entity extractor
│   ├── sentiment_analyzer.py   # Sentiment polarity & intent classifier
│   └── conversation_manager.py # Chat session persistence & markdown export
├── utils/
│   ├── config.py               # Constants and configuration
│   ├── helpers.py              # Formatters and badge renderers
│   └── sample_docs.py          # Academic sample generator
├── sample_data/                # Pre-packaged PDF, DOCX, TXT sample files
├── requirements.txt            # Python dependencies
├── .env.example                # API key template
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 📄 License & Attribution
Developed for the **Major Project in Artificial Intelligence & Generative AI**.  
*Built with Python • NLP • RAG • Embeddings • Vector Search • Generative AI*
