"""
Hybrid Retrieval Engine for IntelliAssist AI.
Combines Dense Semantic Vector Retrieval (sentence-transformers / embeddings)
with BM25 Okapi Lexical Retrieval using Reciprocal Rank Fusion (RRF).
Explicitly distinguishes and tags Semantic Matches vs Keyword Matches vs Hybrid Matches.
"""

import math
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

class BM25Index:
    """
    In-memory BM25 Okapi lexical index implemented with pure NumPy/Python.
    Supports term frequency, inverse document frequency (IDF), and document length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, corpus: Optional[List[str]] = None):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, freq)]
        self.idf: Dict[str, float] = {}
        if corpus is not None:
            self.build_index(corpus)

    @property
    def doc_count(self) -> int:
        return self.corpus_size

    @property
    def avgdl(self) -> float:
        return self.avg_doc_len

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        if not text:
            return []
        return [token for token in re.findall(r'[a-zA-Z0-9]+', text.lower()) if len(token) > 1]

    def build_index(self, documents: List[str]):
        """Index a corpus of document strings."""
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.doc_lens = []
            self.doc_freqs = {}
            self.inverted_index = {}
            self.idf = {}
            return

        self.doc_lens = []
        self.inverted_index = {}
        df_counter: Counter = Counter()

        for doc_idx, doc_text in enumerate(documents):
            tokens = self.tokenize(doc_text)
            self.doc_lens.append(len(tokens))
            term_counts = Counter(tokens)

            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, count))
                df_counter[term] += 1

        self.doc_freqs = dict(df_counter)
        self.avg_doc_len = float(sum(self.doc_lens)) / max(1, self.corpus_size)

        # Precompute Lucene/BM25 Okapi smoothed IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF formulation: ln((N - n + 0.5) / (n + 0.5) + 1.0)
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for all documents given a query string."""
        if self.corpus_size == 0:
            return np.zeros(0, dtype=np.float32)

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return scores

        for token in q_tokens:
            if token not in self.inverted_index:
                continue

            term_idf = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]

            for doc_idx, tf in postings:
                doc_len = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                if denom > 0:
                    score_contrib = term_idf * ((tf * (self.k1 + 1.0)) / denom)
                    scores[doc_idx] += float(score_contrib)

        return scores


class HybridRetriever:
    """
    Orchestrates Dense Semantic retrieval and BM25 Lexical retrieval,
    combining ranked outputs using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25_index = BM25Index()
        self._indexed_chunk_ids: List[str] = []

    def update_bm25_index(self, chunks: List[Dict[str, Any]]):
        """Rebuild or sync BM25 index with current chunks."""
        texts = [c.get("text", "") for c in chunks]
        self.bm25_index.build_index(texts)
        self._indexed_chunk_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
        logger.info("BM25 index updated with %d chunks", len(chunks))

    def hybrid_search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        dense_scores: np.ndarray,
        top_k: int = 6,
        filter_doc: Optional[str] = None,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search over chunks using both dense vectors and BM25 lexical scores.
        Fuses ranks using Reciprocal Rank Fusion (RRF).
        Tags matches as 'Semantic Match', 'Keyword Match', or 'Hybrid Match'.
        """
        if not chunks or len(dense_scores) != len(chunks):
            return []

        # Filter candidate indices if filter_doc is provided
        valid_indices = []
        for i, c in enumerate(chunks):
            if filter_doc and filter_doc != "All Documents" and c.get("filename") != filter_doc:
                continue
            valid_indices.append(i)

        if not valid_indices:
            return []

        # 1. Dense Semantic Ranking
        dense_ranked = sorted(valid_indices, key=lambda idx: dense_scores[idx], reverse=True)
        dense_ranks = {doc_idx: rank for rank, doc_idx in enumerate(dense_ranked, 1)}

        # 2. BM25 Lexical Ranking
        bm25_scores = self.bm25_index.score(query)
        if len(bm25_scores) != len(chunks):
            self.update_bm25_index(chunks)
            bm25_scores = self.bm25_index.score(query)

        bm25_ranked = sorted(valid_indices, key=lambda idx: bm25_scores[idx], reverse=True)
        bm25_ranks = {doc_idx: rank for rank, doc_idx in enumerate(bm25_ranked, 1)}

        # 3. Reciprocal Rank Fusion (RRF):
        # RRF_Score(d) = sum_m [ weight_m / (k + rank_m(d)) ]
        rrf_scores = {}
        for idx in valid_indices:
            r_dense = dense_ranks.get(idx, len(valid_indices) + 1)
            r_bm25 = bm25_ranks.get(idx, len(valid_indices) + 1)
            
            # If BM25 score is 0, give pure dense weight
            b_score = bm25_scores[idx]
            if b_score > 0:
                rrf_score = (dense_weight / (self.rrf_k + r_dense)) + (bm25_weight / (self.rrf_k + r_bm25))
            else:
                rrf_score = dense_weight / (self.rrf_k + r_dense)
            rrf_scores[idx] = rrf_score

        # Sort by RRF score descending
        sorted_indices = sorted(valid_indices, key=lambda idx: rrf_scores[idx], reverse=True)

        results = []
        for idx in sorted_indices[:top_k]:
            chunk = dict(chunks[idx])
            d_score = float(dense_scores[idx])
            b_score = float(bm25_scores[idx])
            rrf_score = float(rrf_scores[idx])

            # Determine Match Type and Semantic Explanation
            has_keyword = b_score > 0.05
            has_semantic = d_score >= 0.20

            if has_semantic and has_keyword:
                match_type = "Hybrid Match"
                match_explanation = f"Matched via both dense semantic proximity ({d_score:.2f}) and lexical keyword terms (BM25: {b_score:.2f})."
            elif has_keyword:
                match_type = "Keyword Match"
                match_explanation = f"Exact lexical term match in passage (BM25 score: {b_score:.2f})."
            else:
                match_type = "Semantic Match"
                match_explanation = f"Retrieved because this passage is semantically related to your query (cosine: {d_score:.2f})."

            # Authentically calibrated confidence calculation:
            # Combines normalized dense similarity with lexical bonus
            # Bounded strictly [0.0, 0.99]
            calibrated_confidence = max(0.0, min(0.99, float(d_score * 0.70 + min(0.30, b_score * 0.05))))

            chunk["score"] = round(calibrated_confidence, 4)
            chunk["dense_score"] = round(d_score, 4)
            chunk["bm25_score"] = round(b_score, 4)
            chunk["rrf_score"] = round(rrf_score, 6)
            chunk["similarity_percentage"] = int(round(calibrated_confidence * 100))
            chunk["match_type"] = match_type
            chunk["match_explanation"] = match_explanation

            results.append(chunk)

        return results
