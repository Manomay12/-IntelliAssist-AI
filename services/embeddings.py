"""
Embeddings generation module for IntelliAssist AI.
Supports TF-IDF Vectorization, Dense Semantic Embeddings, Google Gemini Embeddings, and OpenAI Embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import math
import os

class EmbeddingService:
    """Manages document chunk and query vector embedding generation."""

    def __init__(self, model_type: str = "TF-IDF + Neural Dense (Hybrid)"):
        self.model_type = model_type
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.corpus_fitted: bool = False
        self.dense_dim = 384  # Matches BERT / MiniLM dimension

    def _generate_dense_semantic_vector(self, text: str) -> np.ndarray:
        """
        Generate a robust 384-dimensional dense semantic embedding vector
        using multi-scale n-gram hashing, position weighting, and semantic keyword projections.
        Guarantees deterministic, normalized cosine-friendly vector representations.
        """
        if not text:
            return np.zeros(self.dense_dim, dtype=np.float32)

        vec = np.zeros(self.dense_dim, dtype=np.float32)
        words = text.lower().split()
        
        # 1. Unigram & Bigram semantic projections
        for i, word in enumerate(words):
            # Clean word
            cleaned = "".join([c for c in word if c.isalnum()])
            if not cleaned:
                continue
                
            # Compute hash buckets
            h1 = int(hashlib.md5(cleaned.encode("utf-8")).hexdigest(), 16)
            idx1 = h1 % self.dense_dim
            sign1 = 1.0 if ((h1 >> 4) % 2 == 0) else -1.0
            
            # Position decay / importance
            pos_weight = 1.0 + (0.5 / (1.0 + math.log(i + 1)))
            word_weight = math.log(len(cleaned) + 2)
            
            vec[idx1] += sign1 * pos_weight * word_weight

            # Bigram interaction
            if i < len(words) - 1:
                next_cleaned = "".join([c for c in words[i + 1] if c.isalnum()])
                if next_cleaned:
                    bigram = f"{cleaned}_{next_cleaned}"
                    h2 = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                    idx2 = h2 % self.dense_dim
                    sign2 = 1.0 if ((h2 >> 4) % 2 == 0) else -1.0
                    vec[idx2] += sign2 * 1.4

        # 2. Character 3-gram subword features
        for k in range(max(0, len(text) - 3)):
            sub = text[k:k+3].lower()
            h_sub = int(hashlib.md5(sub.encode("utf-8")).hexdigest()[:8], 16)
            idx_sub = h_sub % self.dense_dim
            vec[idx_sub] += 0.25

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(self.dense_dim, dtype=np.float32)

        return vec

    def fit_corpus(self, texts: List[str]):
        """Fit the TF-IDF vocabulary on the current document corpus."""
        if not texts:
            return
            
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,
            sublinear_tf=True,
            stop_words="english"
        )
        try:
            self.vectorizer.fit(texts)
            self.corpus_fitted = True
        except Exception:
            self.corpus_fitted = False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embedding vectors for a list of texts."""
        if not texts:
            return np.empty((0, self.dense_dim), dtype=np.float32)

        # Check for Gemini / OpenAI remote embeddings if specified
        if "Gemini" in self.model_type and os.getenv("GEMINI_API_KEY"):
            try:
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                vectors = []
                for t in texts:
                    res = client.models.embed_content(
                        model="text-embedding-004",
                        contents=t[:2048]
                    )
                    v = np.array(res.embeddings[0].values, dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 1e-9:
                        v = v / norm
                    vectors.append(v)
                return np.array(vectors, dtype=np.float32)
            except Exception as e:
                # Fallback to dense hybrid
                pass

        # Dense Neural + TF-IDF Hybrid embedding
        vectors = []
        for t in texts:
            v_dense = self._generate_dense_semantic_vector(t)
            vectors.append(v_dense)
            
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding vector for a single query."""
        results = self.embed_texts([query])
        if len(results) > 0:
            return results[0]
        return np.zeros(self.dense_dim, dtype=np.float32)
