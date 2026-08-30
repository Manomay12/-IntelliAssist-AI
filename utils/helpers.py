"""
Helper utilities for formatting, text processing, and UI badge rendering.
"""

import time
import math
import hashlib
from typing import List, Dict, Any

def format_file_size(size_in_bytes: int) -> str:
    """Format bytes into a human readable file size string."""
    if size_in_bytes <= 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_in_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_in_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_file_icon(file_extension: str) -> str:
    """Return an appropriate modern icon for file types."""
    ext = file_extension.lower().strip()
    if ext == ".pdf":
        return "📕"
    elif ext in [".doc", ".docx"]:
        return "📘"
    elif ext in [".txt", ".md"]:
        return "📄"
    return "📁"

def generate_doc_id(content_or_filename: str) -> str:
    """Generate a stable short ID from a document string."""
    return hashlib.md5(content_or_filename.encode("utf-8")).hexdigest()[:10]

def highlight_keywords(text: str, query: str, max_length: int = 350) -> str:
    """Highlight query terms in a snippet and truncate cleanly."""
    if not text:
        return ""
    
    words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
    
    # Locate first occurrence of any keyword to center the snippet
    first_idx = len(text)
    for word in words:
        pos = text.lower().find(word)
        if 0 <= pos < first_idx:
            first_idx = pos
            
    start = max(0, first_idx - 60)
    end = min(len(text), start + max_length)
    snippet = text[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
        
    return snippet

def get_relevance_badge_html(score: float) -> str:
    """Return a styled relevance percentage badge HTML."""
    pct = int(min(100, max(0, score * 100)))
    if pct >= 80:
        color = "#10b981"  # Emerald green
        bg = "rgba(16, 185, 129, 0.15)"
    elif pct >= 60:
        color = "#3b82f6"  # Blue
        bg = "rgba(59, 130, 246, 0.15)"
    else:
        color = "#f59e0b"  # Amber
        bg = "rgba(245, 158, 11, 0.15)"
        
    return f"""<span style="display:inline-flex; align-items:center; gap:4px; font-weight:600; font-size:0.75rem; padding:3px 10px; border-radius:9999px; background:{bg}; color:{color}; border:1px solid {color}40;">
        ⚡ {pct}% match
    </span>"""
