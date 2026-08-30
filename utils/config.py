"""
Configuration module for IntelliAssist AI.
Manages application constants, default parameters, and environment settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Ensure all directories exist
for directory in [DATA_DIR, UPLOAD_DIR, VECTOR_DB_DIR, CONVERSATIONS_DIR, SAMPLE_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Application metadata
APP_NAME = "IntelliAssist AI"
APP_TAGLINE = "Smart Document Intelligence"
APP_SUBTITLE = "AI-Powered Document Analysis, Semantic Search & Conversational RAG"
APP_VERSION = "2.5.0 Enterprise Academic Edition"
APP_AUTHOR = "Major Project AI Team"

# Supported file formats
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json", ".log"]
MAX_FILE_SIZE_MB = 50

# Default RAG & Chunking Parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 6
DEFAULT_SIMILARITY_THRESHOLD = 0.15

# LLM Providers and Models
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "Google Gemini")
AVAILABLE_PROVIDERS = [
    "Google Gemini",
    "NVIDIA NIM / AI",
    "OpenAI",
    "Demo Mode (Smart AI)"
]

AVAILABLE_GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

AVAILABLE_NVIDIA_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "mistralai/mistral-large-2-instruct",
    "nvidia/nemotron-4-340b-instruct",
]

AVAILABLE_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048

# API Keys from environment (Securely loaded via .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
