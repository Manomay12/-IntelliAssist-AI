"""
Vector Database engine for IntelliAssist AI.
Manages chunk storage, dense embedding indexing, BM25 Okapi lexical indexing,
Hybrid Reciprocal Rank Fusion retrieval, duplicate detection, and persistent disk caching.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from utils.config import VECTOR_DB_DIR
from services.embeddings import EmbeddingService
from services.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

class VectorStore:
    """In-memory & persistent Vector Database supporting dense cosine and hybrid RRF retrieval."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None, db_dir: Path = VECTOR_DB_DIR):
        self.embedding_service = embedding_service or EmbeddingService()
        self.db_dir = Path(db_dir)
        self.chunks_file = self.db_dir / "chunks_metadata.json"
        self.vectors_file = self.db_dir / "vectors_index.npy"

        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.empty((0, self.embedding_service.dense_dim), dtype=np.float32)
        self.hybrid_retriever = HybridRetriever()

        self.load_from_disk()

    def add_documents(self, chunks: List[Dict[str, Any]], persist: bool = True) -> int:
        """Add new chunks to the vector database, update BM25 index, and optionally persist to disk."""
        if not chunks:
            return 0

        # Extract texts to embed
        texts = [c.get("text", "") for c in chunks]
        new_vectors = self.embedding_service.embed_texts(texts)

        if len(self.chunks) == 0 or self.vectors.shape[0] == 0:
            self.chunks = list(chunks)
            self.vectors = new_vectors
        else:
            self.chunks.extend(chunks)
            self.vectors = np.vstack([self.vectors, new_vectors])

        # Synchronize BM25 lexical index
        self.hybrid_retriever.update_bm25_index(self.chunks)

        if persist:
            self.save_to_disk()
        logger.info("Indexed %d new chunks into Vector Store (total: %d)", len(chunks), len(self.chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.0,
        filter_doc: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform search against indexed chunks.
        When use_hybrid is True (default), fuses Dense Cosine Embeddings with BM25 Lexical search via RRF.
        Returns top-k matching chunks with calibrated scores and match explanations.
        """
        if len(self.chunks) == 0 or self.vectors.shape[0] == 0:
            return []

        # Embed the query
        q_vec = self.embedding_service.embed_query(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-9:
            return []

        q_vec = q_vec / q_norm

        # Compute cosine similarity
        dense_scores = np.dot(self.vectors, q_vec)

        if use_hybrid:
            hybrid_results = self.hybrid_retriever.hybrid_search(
                query=query,
                chunks=self.chunks,
                dense_scores=dense_scores,
                top_k=top_k * 2,  # retrieve extra candidates for thresholding
                filter_doc=filter_doc
            )
            # Filter by threshold
            filtered = [r for r in hybrid_results if r.get("score", 0.0) >= threshold]
            return filtered[:top_k]

        # Pure Dense Fallback
        results = []
        for idx, score in enumerate(dense_scores):
            chunk = self.chunks[idx]
            if filter_doc and filter_doc != "All Documents" and chunk.get("filename") != filter_doc:
                continue

            final_score = max(0.0, min(0.99, float(score)))
            if final_score >= threshold:
                item = dict(chunk)
                item["score"] = round(final_score, 4)
                item["similarity_percentage"] = int(round(final_score * 100))
                item["match_type"] = "Semantic Match"
                item["match_explanation"] = f"Retrieved via semantic embedding similarity (cosine: {final_score:.2f})"
                results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_documents(self) -> List[str]:
        """Return a sorted list of unique document filenames in the store."""
        docs = set(c.get("filename", "") for c in self.chunks if c.get("filename"))
        return sorted(list(docs))

    def get_indexed_file_hashes(self) -> Dict[str, str]:
        """Return mapping of file_hash -> filename for all indexed documents."""
        mapping = {}
        for c in self.chunks:
            f_hash = c.get("file_hash")
            f_name = c.get("filename")
            if f_hash and f_name and f_hash not in mapping:
                mapping[f_hash] = f_name
        return mapping

    def has_document_hash(self, file_hash: str) -> bool:
        """Check if a document with this SHA-256 hash is already indexed."""
        if not file_hash:
            return False
        return any(c.get("file_hash") == file_hash for c in self.chunks)

    def get_document_by_hash(self, file_hash: str) -> Optional[str]:
        """Retrieve the filename matching an existing file hash if indexed."""
        for c in self.chunks:
            if c.get("file_hash") == file_hash:
                return c.get("filename")
        return None

    def get_document_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks belonging to a specific document."""
        return [c for c in self.chunks if c.get("filename") == filename]

    def delete_document(self, filename: str, persist: bool = True) -> int:
        """Remove all chunks associated with a specific document and re-index."""
        if not filename:
            return 0

        keep_indices = [i for i, c in enumerate(self.chunks) if c.get("filename") != filename]
        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            if len(keep_indices) > 0 and self.vectors.shape[0] > 0:
                self.vectors = self.vectors[keep_indices]
            else:
                self.vectors = np.empty((0, self.embedding_service.dense_dim), dtype=np.float32)

            self.hybrid_retriever.update_bm25_index(self.chunks)
            if persist:
                self.save_to_disk()
            logger.info("Deleted document '%s' (removed %d chunks)", filename, removed_count)

        return removed_count

    def clear(self):
        """Clear all stored vectors and chunks."""
        self.chunks = []
        self.vectors = np.empty((0, self.embedding_service.dense_dim), dtype=np.float32)
        self.hybrid_retriever = HybridRetriever()
        if self.chunks_file.exists():
            self.chunks_file.unlink()
        if self.vectors_file.exists():
            self.vectors_file.unlink()
        logger.info("Cleared VectorStore in-memory and on disk.")

    def save_to_disk(self):
        """Persist chunk metadata and vector array to disk."""
        try:
            self.db_dir.mkdir(parents=True, exist_ok=True)
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            np.save(self.vectors_file, self.vectors)
        except Exception as e:
            logger.error("Failed to persist vector store to disk: %s", e)

    def load_from_disk(self):
        """Load indexed chunks and vectors from disk if available."""
        try:
            if self.chunks_file.exists() and self.vectors_file.exists():
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.vectors = np.load(self.vectors_file)
                self.hybrid_retriever.update_bm25_index(self.chunks)
                logger.info("Loaded %d chunks from vector disk cache.", len(self.chunks))
        except Exception as e:
            logger.warning("Failed to load vector store from disk: %s", e)
            self.chunks = []
            self.vectors = np.empty((0, self.embedding_service.dense_dim), dtype=np.float32)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self.get_all_documents())
