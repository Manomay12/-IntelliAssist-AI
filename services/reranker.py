"""
Neural Reranking module for IntelliAssist AI.
Uses cross-encoder models (cross-encoder/ms-marco-MiniLM-L-6-v2) to compute deep cross-attention
relevance between queries and retrieved passages, with fast cosine fallback.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

_CROSS_ENCODER_CACHE: Dict[str, Any] = {}

class NeuralReranker:
    """Reranks candidate retrieved chunks using a cross-encoder neural model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._available: Optional[bool] = None

    def _ensure_model(self):
        """Lazily initialize the cross-encoder model."""
        if self._available is not None:
            return
        if not self.enabled:
            self._available = False
            return

        try:
            from sentence_transformers import CrossEncoder
            global _CROSS_ENCODER_CACHE
            if self.model_name not in _CROSS_ENCODER_CACHE:
                logger.info("Initializing CrossEncoder model '%s'...", self.model_name)
                _CROSS_ENCODER_CACHE[self.model_name] = CrossEncoder(self.model_name, max_length=512)
            self._model = _CROSS_ENCODER_CACHE[self.model_name]
            self._available = True
            logger.info("NeuralReranker initialized successfully with %s", self.model_name)
        except Exception as e:
            self._available = False
            logger.info("CrossEncoder unavailable (%s). Using calibrated hybrid ranking.", e)

    def is_available(self) -> bool:
        """Check if neural cross-encoder is available."""
        self._ensure_model()
        return bool(self._available)

    def rerank(
        self,
        query: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        top_k: Optional[int] = None,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks by passing (query, chunk_text) pairs through the neural cross-encoder.
        Computes calibrated scores via sigmoid normalization.
        """
        target_chunks = chunks if chunks is not None else (candidates if candidates is not None else [])
        if not target_chunks:
            return []

        clean_q = query.strip()
        if not clean_q:
            return target_chunks[:top_k] if top_k else target_chunks

        self._ensure_model()

        if not self._available or self._model is None:
            # Fallback: keep existing hybrid/dense scores
            sorted_chunks = sorted(target_chunks, key=lambda c: c.get("score", 0.0), reverse=True)
            return sorted_chunks[:top_k] if top_k else sorted_chunks

        try:
            start_time = time.time()
            pairs = [[clean_q, c.get("text", "")] for c in target_chunks]
            raw_scores = self._model.predict(pairs)

            # Sigmoid activation: 1 / (1 + exp(-score))
            sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))

            reranked_chunks = []
            for chunk, sig_score in zip(target_chunks, sigmoid_scores):
                c = dict(chunk)
                rerank_score = float(sig_score)
                # Combine original hybrid score with neural reranker score (60% reranker, 40% retrieval)
                orig_score = float(chunk.get("score", rerank_score))
                combined_confidence = max(0.0, min(0.99, 0.60 * rerank_score + 0.40 * orig_score))

                c["rerank_score"] = round(rerank_score, 4)
                c["score"] = round(combined_confidence, 4)
                c["similarity_percentage"] = int(round(combined_confidence * 100))
                c["reranked"] = True
                reranked_chunks.append(c)

            reranked_chunks.sort(key=lambda c: c["score"], reverse=True)
            elapsed = time.time() - start_time
            logger.debug("Neural reranked %d chunks in %.3fs", len(target_chunks), elapsed)
            return reranked_chunks[:top_k] if top_k else reranked_chunks
        except Exception as e:
            logger.warning("Neural reranking failed (%s). Falling back to retrieval rank.", e)
            return chunks[:top_k] if top_k else chunks
