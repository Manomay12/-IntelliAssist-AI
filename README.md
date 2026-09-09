# ⚡ IntelliAssist AI — Smart Document AI Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![RAG Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-792EE5?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![PyTest](https://img.shields.io/badge/Tests-32%20Passed%20(100%25)-success?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](LICENSE)

**A production-style, enterprise-grade AI Document Intelligence and Conversational Retrieval-Augmented Generation (RAG) platform.**  
*Built with Python, Streamlit, Sentence-Transformers neural embeddings, persistent Cosine Vector Indexing, cross-document comparison, and multi-provider Generative AI.*

</div>

---

## 📌 1. Project Overview & Problem Statement

Students, researchers, and enterprise analysts face severe information overload when trying to extract insights and synthesize knowledge from dense academic papers, multi-page technical reports, and notes. Traditional keyword search (`Ctrl+F`) fails to capture semantic nuance, conceptual synonyms, and contextual relationships.

**IntelliAssist AI** solves this challenge with an end-to-end conversational AI document intelligence system:
- **Factually Grounded RAG Chat**: Natural language Q&A with strict document scoping, verified page citations, and low-confidence hallucination protection.
- **Dense Neural Semantic Embeddings**: 384-dimensional vector representations powered by `sentence-transformers/all-MiniLM-L6-v2` with Gemini/OpenAI cloud fallbacks.
- **Cross-Document Comparison**: Automated side-by-side comparative matrices contrasting methodologies, benchmark findings, and technical tradeoffs between documents.
- **Production Resilience**: SHA-256 duplicate detection, corrupt file safeguards, and scanned-PDF detection.
- **Multi-Mode Summarization & NLP**: 5 summary modes, entity discovery, sentiment polarity donut charts, and chunk trajectory trends.

---

## 🏛️ 2. Core System Architecture

### 📊 Visual Architecture Flow

![IntelliAssist AI Architecture](assets/architecture_diagram.svg)

### 🔄 End-to-End Pipeline Diagram

```
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│  User Uploads   │ ───► │ 1. Ingestion & Security │ ───► │ 2. Text Normalizer  │
│ (PDF/DOCX/TXT)  │      │    (SHA-256 & Corrupt)  │      │    & Page Tracking   │
└─────────────────┘      └─────────────────────────┘      └──────────────────────┘
                                                                     │
                                                                     ▼
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│ 5. Vector Store │ ◄─── │ 4. Embeddings Engine    │ ◄─── │ 3. Recursive Chunker │
│  (Cosine Index) │      │  (all-MiniLM-L6-v2)     │      │ (500 chars / 100 ov) │
└─────────────────┘      └─────────────────────────┘      └──────────────────────┘
       │
       │  Semantic Query Embedding & Retrieval
       ▼
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│ 6. Top-K Ranker │ ───► │ 7. Multi-Provider LLM   │ ───► │ Verified Answer with │
│ (Relevance Cal) │      │ (Gemini / OpenAI / Demo)│      │ Exact Source Citation│
└─────────────────┘      └─────────────────────────┘      └──────────────────────┘
```

---

## 📸 3. Application Preview

<div align="center">

### 🏠 Modern AI SaaS Dashboard
*Interactive metrics, recent uploads, Quick Ask AI, and telemetry logs.*

```
+-----------------------------------------------------------------------------------+
|  ⚡ IntelliAssist AI                         [📕 3 Files] [🧩 394 Chunks] [🟢 Ready]|
+-----------------------------------------------------------------------------------+
|  [ 📄 3 Total Files ]   [ 🧩 394 Vectors ]   [ 💬 14 Inquiries ]   [ ⚡ 18ms Latency ] |
|                                                                                   |
|  🎯 Quick Ask AI: [ "Explain how attention works in transformers"         [Ask AI] ]|
|                                                                                   |
|  📚 Indexed Documents                 |  📈 System Activity Telemetry              |
|  • AI_Research_Transformers.pdf       |  • [14:32:10] RAG Answer: 6 sources cited  |
|  • Project_Report_IntelliAssist.docx  |  • [14:30:15] Indexed 394 chunks into DB   |
|  • Healthcare_ML_Overview.txt         |  • [14:28:02] SentenceTransformer loaded   |
+-----------------------------------------------------------------------------------+
```

### 💬 Grounded AI Chat with Citations
*Page-level citations, calibrated retrieval relevance badges, and dynamic question suggestions.*

```
+-----------------------------------------------------------------------------------+
| 👤 You: Explain how the multi-head attention mechanism operates.                   |
+-----------------------------------------------------------------------------------+
| 🤖 IntelliAssist AI:                                                              |
| 1. 🎯 Executive Summary: Multi-head attention allows the model to jointly attend  |
|    to information from different representation subspaces at different positions. |
| 2. 🔍 In-Depth Breakdown: Queries, Keys, and Values are linearly projected h times|
|    with dimension d_k = d_model / h.                                              |
|                                                                                   |
| 📚 Sources & Citations (3 references cited):                                      |
| 📕 AI_Research_Paper.pdf (Page 2) — [🟢 94% Relevance (Strong match)]             |
|    "Multi-head attention projects queries, keys and values h times with..."       |
+-----------------------------------------------------------------------------------+
```

### ⚖️ Cross-Document Comparison Matrix
*Automated side-by-side comparative analysis of two selected documents.*

| Comparative Dimension | 📄 AI_Research_Paper.pdf | 📄 Project_Report.docx |
| :--- | :--- | :--- |
| **Primary Domain** | Transformer Architectures & RAG | Enterprise AI Document Intelligence |
| **Core Method** | Scaled Dot-Product & Self-Attention | Modular Ingestion, Embeddings & Vector DB |
| **Key Benchmark** | 28.4 BLEU (EN-DE), 41.8 BLEU (EN-FR) | 94.6% F1 Faithfulness, 18ms Retrieval |
| **Technologies** | PyTorch, Transformers, GPUs | Python, Streamlit, all-MiniLM-L6-v2 |

</div>

---

## 🧠 4. Technical Highlights

- **Semantic Vector Embeddings**: 384-dimensional dense neural embeddings powered by `sentence-transformers/all-MiniLM-L6-v2` with support for Google Gemini `text-embedding-004` and OpenAI `text-embedding-3-small`.
- **Calibrated Retrieval Relevance**: Transparent relevance scoring (🟢 Strong >75%, 🟡 Moderate 55–75%, 🔴 Weak <55%) clearly distinguishing vector cosine similarity from LLM reasoning confidence.
- **Hallucination Protection**: Out-of-domain queries trigger an honest *"Insufficient Document Evidence"* notice rather than generating ungrounded facts.
- **SHA-256 Duplicate Protection**: Computes unique file hashes upon upload to prevent redundant chunk re-indexing.
- **Corrupt & Scanned File Resilience**: Gracefully detects 0-byte files, password-protected PDFs, and image-only scanned PDFs with actionable diagnostics.
- **Multi-Document & Target Scoping**: Query across all uploaded documents simultaneously or target a specific file with strict context boundaries.
- **Cross-Document Comparison Engine**: Automatic side-by-side extraction of domains, objectives, methodologies, and metrics.
- **Conversation Persistence**: Save, load, rename, clear, and export conversations to formatted Markdown transcripts.

---

## 🛠️ 5. Technology Stack

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Frontend / UI** | Streamlit (Python) | Custom CSS Design System, Glassmorphism, Dark Mode, Telemetry |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` (384-dim dense vectors), Gemini & OpenAI APIs |
| **Vector Store** | In-Memory Cosine DB | High-speed NumPy dot-product vector matrix with JSON disk persistence |
| **Document Ingestion**| `pypdf`, `python-docx` | Multi-strategy extraction, SHA-256 hashing, layout cleaning |
| **RAG & Synthesis** | Multi-Provider LLMs | Google Gemini 2.0 / 1.5, OpenAI GPT-4o, and Smart Local NLP Engine |
| **Analytics & Viz** | `plotly`, `pandas` | Interactive sentiment donuts, intent bar charts, chunk trend lines |
| **Unit Testing** | `pytest` | 32 automated unit tests across all modular services |
| **Containerization**| Docker & Docker Compose | Python 3.11-slim container with healthcheck |

---

## 🚀 6. Installation & Quick Start

### Method 1: Local Setup (Recommended)

#### 1. Clone the repository
```bash
git clone https://github.com/Manomay12/-IntelliAssist-AI.git
cd -IntelliAssist-AI
```

#### 2. Create and activate virtual environment
```bash
# Using standard Python
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure API Keys (Optional)
```bash
cp .env.example .env
```
*(Note: If no API key is provided, the application runs in **Smart Demo AI Mode** with 100% full offline functionality).*

#### 5. Launch the application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

### Method 2: Docker Deployment

Run IntelliAssist AI with a single command:

```bash
docker compose up --build
```
Access the application at **`http://localhost:8501`**.

---

## 🧪 7. Running Automated Tests

IntelliAssist AI includes a comprehensive `pytest` test suite with 32 unit tests:

```bash
# Run full test suite with verbose output
pytest -v tests/
```

### Test Suite Structure
```
tests/
├── conftest.py                       # Fixtures for vector store, sample docs, and engines
├── test_document_processor.py        # PDF, DOCX, TXT parsing, SHA-256 hash, corrupt files
├── test_chunker.py                   # Recursive text splitting, boundaries & metadata
├── test_embeddings.py                # 384-dim shape, L2 normalization & model reporting
├── test_vector_store.py              # Cosine retrieval, deletion, duplicate detection & persistence
├── test_rag.py                       # Typo normalization, strict scoping & hallucination guard
├── test_sentiment.py                 # Sentiment polarity, multi-class intent & chunk trends
└── test_summarizer_and_compare.py    # 5 summary modes, entity discovery & cross-doc comparison
```

---

## 📁 8. Project Structure

```
intelliassist_ai/
├── app.py                      # Main entrypoint, page routing & state management
├── requirements.txt            # Python dependencies (sentence-transformers, pytest, etc.)
├── Dockerfile                  # Production Docker container definition
├── docker-compose.yml          # Container orchestration configuration
├── .dockerignore               # Container build exclusions
├── .gitignore                  # Git repository ignore rules
├── README.md                   # Comprehensive documentation
│
├── assets/                     # Architecture diagrams and preview assets
│   └── architecture_diagram.svg
│
├── tests/                      # Professional pytest unit test suite (32 tests)
│   ├── conftest.py
│   ├── test_document_processor.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_rag.py
│   ├── test_sentiment.py
│   └── test_summarizer_and_compare.py
│
├── pages_views/                # Modular Page Views
│   ├── dashboard_view.py       # 🏠 Dashboard
│   ├── documents_view.py       # 📄 Document Upload & Library
│   ├── chat_view.py            # 💬 AI Chat & RAG
│   ├── search_view.py          # 🔎 Semantic Search
│   ├── summarizer_view.py      # 📝 Multi-Mode Summarizer & Compare
│   ├── sentiment_view.py       # 🧠 Sentiment & Intent Intelligence
│   ├── analytics_view.py       # 📊 System Telemetry
│   ├── history_view.py         # 🕘 Conversation History
│   └── settings_view.py        # ⚙️ Settings & LLM Config
│
├── components/                 # Reusable UI Components
│   ├── styles.py               # Custom CSS Design System
│   ├── sidebar.py              # Navigation bar & system status
│   ├── chat_ui.py              # Chat bubbles, prompt chips, feedback
│   ├── document_card.py        # Document library cards
│   ├── source_card.py          # Calibrated source citation references
│   ├── metrics.py              # Stats counter tiles
│   └── charts.py               # Plotly telemetry visualizations
│
├── services/                   # Core Backend & AI Services
│   ├── document_processor.py   # Text extraction, SHA-256 hash & error handling
│   ├── chunker.py              # Recursive overlapping text chunking
│   ├── embeddings.py           # Neural embeddings (all-MiniLM-L6-v2) & fallbacks
│   ├── vector_store.py         # Cosine vector database & duplicate check
│   ├── rag_engine.py           # RAG retrieval, context ranking & hallucination guard
│   ├── llm_service.py          # Multi-provider LLM interface (Gemini/OpenAI/Demo)
│   ├── summarizer.py           # Summarizer & Cross-Document Comparison engine
│   ├── sentiment_analyzer.py   # Sentiment polarity & multi-class intent classifier
│   └── conversation_manager.py # Chat session persistence & Markdown exporter
│
├── utils/                      # Utilities & Configuration
│   ├── config.py               # Application configuration constants
│   ├── helpers.py              # Formatting & calibrated relevance badge HTML
│   └── sample_docs.py          # Academic sample documents generator
│
└── sample_data/                # Bundled PDF, DOCX, TXT sample evaluation documents
```

---

## 🌐 9. Cloud & Container Deployment

IntelliAssist AI is fully optimized for production deployment on **Render**, **Docker**, and **Docker Compose**.

### Option A: Deploy on Render (Recommended)

Render provides free cloud hosting with automatic continuous deployment directly from your GitHub repository.

#### Method 1: 1-Click Render Blueprint (Fastest)
1. Push your changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure Render deployment"
   git push origin main
   ```
2. Log into the [Render Dashboard](https://dashboard.render.com).
3. Click **New +** → **Blueprint**.
4. Connect your `intelliassist-ai` GitHub repository.
5. Render will detect `render.yaml` automatically.
6. Enter your secret environment variables (e.g., `GEMINI_API_KEY`, `NVIDIA_API_KEY`, or `OPENAI_API_KEY`) when prompted.
7. Click **Apply**. Render will build and deploy your app.

#### Method 2: Manual Web Service on Render
If you prefer configuring the Web Service manually via Render dashboard:
1. Click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the following fields:
   - **Name**: `intelliassist-ai`
   - **Environment**: `Python 3`
   - **Region**: Oregon (or nearest)
   - **Branch**: `main`
   - **Build Command**: `./build.sh` (or `pip install -r requirements.txt`)
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection true
     ```
   - **Health Check Path**: `/_stcore/health`
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: `false`
   - `GEMINI_API_KEY`: *(Your Google Gemini Key)*
   - `NVIDIA_API_KEY`: *(Your NVIDIA NIM Key)*
   - `DEFAULT_LLM_PROVIDER`: `Google Gemini` (or `NVIDIA NIM / AI`)
5. Click **Create Web Service**.

> **Note on Free Tier**: Render free tier instances spin down after 15 minutes of inactivity. When a new user visits, it may take ~45–50 seconds to wake up.

---

### Option B: Run with Docker

Build and run the production Docker container locally or on any cloud VM:

```bash
# Build the optimized Docker image
docker build -t intelliassist-ai .

# Run container mapping port 8501
docker run -d -p 8501:8501 \
  -e GEMINI_API_KEY="your-gemini-key" \
  -e NVIDIA_API_KEY="your-nvidia-key" \
  --name intelliassist intelliassist-ai
```

Access the application in your browser at `http://localhost:8501`.

---

### Option C: Run with Docker Compose

```bash
# Start container with volume persistence
docker compose up -d

# View logs
docker compose logs -f

# Stop container
docker compose down
```

---

## ⚠️ 10. Current Limitations & Scope

- **Scanned PDFs**: Scanned PDFs containing only bitmap images without selectable text layers require an OCR preprocessing engine (such as Tesseract).
- **Extractive vs Generative in Demo Mode**: In offline Demo Mode without external API keys, summaries and RAG syntheses utilize deterministic sentence centrality ranking and keyword extraction. Connecting a Gemini or OpenAI key enables full generative synthesis.
- **Single-Node Vector Store**: The in-memory cosine vector store is optimized for document collections up to ~50,000 chunks. Very large multi-gigabyte enterprise archives would benefit from distributed vector indexes (e.g. Milvus, Pinecone, or FAISS).

---

## 🚀 11. Future Roadmap

- [x] Genuine `sentence-transformers/all-MiniLM-L6-v2` neural semantic embeddings
- [x] Cross-document side-by-side comparison matrix
- [x] SHA-256 duplicate document detection
- [x] Calibrated Retrieval Relevance indicators (🟢/🟡/🔴)
- [x] Low-confidence hallucination protection
- [x] Comprehensive 32-test `pytest` unit test suite
- [x] Docker & Docker Compose containerization
- [x] Cloud deployment configuration for Render (`render.yaml`, `build.sh`)
- [x] Vector SVG architecture diagram
- [ ] OCR pipeline for image-only scanned PDFs (Tesseract / EasyOCR)
- [ ] Hybrid BM25 + Dense Vector retrieval with Reciprocal Rank Fusion (RRF)
- [ ] Multi-lingual document translation & cross-language Q&A
- [ ] Cloud deployment templates (AWS ECS / GCP Cloud Run)

---

## 📄 12. License & Attribution

Developed for the **Major Project in Artificial Intelligence & Generative AI**.  
*Licensed under the [MIT License](LICENSE).*
