"""
Document Classification service for IntelliAssist AI.
Performs semantic ML classification across standard document genres:
Research Paper, Annual Report, Academic Material, Policy Document, Business Report, Technical Document, General Document.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

DOC_CLASSES = [
    "Research Paper",
    "Annual Report",
    "Academic Material",
    "Policy Document",
    "Business Report",
    "Technical Document",
    "General Document"
]

DOC_EXEMPLARS: Dict[str, List[str]] = {
    "Research Paper": [
        "Abstract introduction related work methodology experimental setup dataset evaluation results ablation study conclusion future work references",
        "We propose a novel deep learning architecture evaluated on empirical benchmarks with precision recall F1 BLEU score",
        "Scientific investigation peer-reviewed conference paper journal publication empirical findings hypothesis testing"
    ],
    "Annual Report": [
        "Annual report financial statements balance sheet profit loss consolidated accounts fiscal year shareholder equity revenue dividends",
        "Corporate governance executive committee letter to shareholders operating cash flow audit report 10-K filing",
        "Company performance financial highlights strategic review market outlook risk factors financial review"
    ],
    "Academic Material": [
        "Lecture notes course syllabus chapter exercises textbook curriculum learning objectives student study guide quiz questions",
        "Educational coursework university lecture tutorial problem sets homework assignment academic exam preparation",
        "Pedagogical explanations foundational principles student textbook key concepts revision flashcards"
    ],
    "Policy Document": [
        "Terms and conditions privacy policy compliance regulatory framework guidelines code of conduct governance obligations",
        "Legal framework statutory requirements standard operating procedure rules liability enforcement protocol",
        "Data protection policy security guidelines compliance audit standard procedures obligations"
    ],
    "Business Report": [
        "Executive summary market analysis competitive landscape commercial business plan quarterly review revenue projections ROI",
        "Strategic business initiative sales pipeline customer churn cost optimization market share growth metrics",
        "Project deliverables business case stakeholder requirements operational roadmap key performance indicators"
    ],
    "Technical Document": [
        "Software architecture API documentation system specification endpoint schema deployment guide code repository data pipeline",
        "Developer manual cloud infrastructure configuration parameters installation guide troubleshooting protocols SDK",
        "Database schema network topology microservices hardware specifications maintenance guide technical whitepaper"
    ],
    "General Document": [
        "General informative article informational overview notes memorandum correspondence text writeup reference summary",
        "Unclassified narrative documentation general discussion background information miscellaneous text"
    ]
}

class DocumentClassifier:
    """Classifies document content into categories using semantic embedding similarity."""

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service
        self._class_vectors: Dict[str, np.ndarray] = {}
        self._initialized: bool = False

    def _ensure_class_vectors(self):
        """Precompute centroids for document genres."""
        if self._initialized:
            return
        if self.embedding_service is None:
            from services.embeddings import EmbeddingService
            self.embedding_service = EmbeddingService()

        for category, texts in DOC_EXEMPLARS.items():
            vecs = self.embedding_service.embed_texts(texts)
            if len(vecs) > 0:
                centroid = np.mean(vecs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    centroid = centroid / norm
                self._class_vectors[category] = centroid
        self._initialized = True

    def classify_document(self, text: str, filename: str = "") -> Dict[str, Any]:
        """
        Classify document based on beginning text excerpt, abstract, and headings.
        Returns: {
            'category': str,
            'confidence': float,
            'confidence_percentage': int,
            'all_scores': Dict[str, float]
        }
        """
        if not text or len(text.strip()) < 30:
            return {
                "category": "General Document",
                "confidence": 0.50,
                "confidence_percentage": 50,
                "all_scores": {c: round(1.0 / len(DOC_CLASSES), 3) for c in DOC_CLASSES}
            }

        self._ensure_class_vectors()

        # Sample representative excerpt (up to first 2500 characters which contains title, abstract, executive summary)
        sample_text = text[:2500]

        # Embed document excerpt
        doc_vec = self.embedding_service.embed_query(sample_text)
        doc_norm = np.linalg.norm(doc_vec)
        if doc_norm > 1e-9:
            doc_vec = doc_vec / doc_norm

        scores = {}
        for cat, centroid in self._class_vectors.items():
            sim = float(np.dot(doc_vec, centroid))
            scores[cat] = max(0.0, sim)

        # Softmax normalization
        vals = np.array(list(scores.values()))
        temp = 0.12
        exp_vals = np.exp((vals - np.max(vals)) / temp)
        probs = exp_vals / np.sum(exp_vals)

        prob_dict = {
            cat: round(float(prob), 4)
            for cat, prob in zip(scores.keys(), probs)
        }

        top_category = max(prob_dict.items(), key=lambda x: x[1])[0]
        top_prob = prob_dict[top_category]

        return {
            "category": top_category,
            "confidence": top_prob,
            "confidence_percentage": int(round(top_prob * 100)),
            "all_scores": prob_dict
        }
