"""
Semantic Query Intent Classifier for IntelliAssist AI.
Uses embedding-based cosine similarity against class prototypes across 8 core intent classes:
QUESTION_ANSWERING, SUMMARIZATION, COMPARISON, EXTRACTION, SEARCH, STUDY, ANALYSIS, ACTION_ITEMS.
Provides genuine ML/embedding classification with confidence calibration and fallback.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

INTENT_CLASSES = [
    "QUESTION_ANSWERING",
    "SUMMARIZATION",
    "COMPARISON",
    "EXTRACTION",
    "SEARCH",
    "STUDY",
    "ANALYSIS",
    "ACTION_ITEMS"
]

# Curated semantic exemplar anchor prompts for zero-shot embedding classification
INTENT_EXEMPLARS: Dict[str, List[str]] = {
    "QUESTION_ANSWERING": [
        "What is the main idea behind this approach?",
        "How does the attention mechanism compute weights?",
        "Who proposed this architecture and in what year?",
        "Why did the experiment fail on the baseline dataset?",
        "Where is the training data collected from?",
        "Can you explain what this equation means in simple terms?"
    ],
    "SUMMARIZATION": [
        "Provide an executive summary of this entire document",
        "Give me a concise TLDR overview of the paper",
        "Summarize the key points and main takeaways in bullet points",
        "What is the overall summary of this report?",
        "Brief breakdown of the document contents"
    ],
    "COMPARISON": [
        "Compare the Transformer model against the LSTM baseline",
        "What are the differences between RAG-Sequence and RAG-Token?",
        "How does model A contrast with model B in terms of accuracy?",
        "What are the pros and cons of approach X versus approach Y?",
        "Evaluate the trade-offs between dense retrieval and sparse retrieval"
    ],
    "EXTRACTION": [
        "Extract all key metrics, percentages, and benchmark numbers",
        "List all organizations, authors, and technologies mentioned",
        "Pull out the dataset specifications and hyperparameter table",
        "Extract every date and milestone mentioned in the document",
        "Retrieve specific quantitative statistics and numbers"
    ],
    "SEARCH": [
        "Find passages mentioning cosine similarity and dot product",
        "Search for references to gradient descent or learning rate",
        "Locate the section discussing hardware specifications",
        "Query text chunks related to clinical ICU mortality",
        "Find where the author mentions ablation studies"
    ],
    "STUDY": [
        "Generate a 5-question multiple choice quiz for studying this topic",
        "Create flashcards for the core definitions and concepts",
        "Help me understand and revise this topic for an upcoming exam",
        "Explain this technically then explain it simply like I am a beginner",
        "Provide comprehensive study notes and key revision points"
    ],
    "ANALYSIS": [
        "Critically analyze the methodology and empirical validity of the results",
        "What are the main risks, limitations, and potential failure modes?",
        "Perform a deep architectural analysis of the proposed framework",
        "Assess the statistical significance of the reported benchmark improvements",
        "Evaluate the theoretical foundation and assumptions"
    ],
    "ACTION_ITEMS": [
        "What are the concrete next steps and action items recommended?",
        "List all decisions and implementation tasks required",
        "What should the team execute first based on these findings?",
        "Identify actionable recommendations and follow-up milestones",
        "Outline a practical deployment checklist from the guidelines"
    ]
}

class QueryIntentClassifier:
    """Classifies user queries into document intelligence intents using semantic vector embeddings."""

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service
        self._anchor_vectors: Dict[str, np.ndarray] = {}
        self._initialized: bool = False

    def _ensure_anchors(self):
        """Precompute and cache class exemplar centroid vectors."""
        if self._initialized:
            return
        if self.embedding_service is None:
            from services.embeddings import EmbeddingService
            self.embedding_service = EmbeddingService()

        for intent, exemplars in INTENT_EXEMPLARS.items():
            vecs = self.embedding_service.embed_texts(exemplars)
            if len(vecs) > 0:
                # Normalize and compute centroid vector
                centroid = np.mean(vecs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    centroid = centroid / norm
                self._anchor_vectors[intent] = centroid
        self._initialized = True
        logger.info("Initialized QueryIntentClassifier with %d intent classes", len(self._anchor_vectors))

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify a query into primary intent and confidence distribution.
        Returns: {
            'primary_intent': str,
            'confidence': float (0.0 to 1.0),
            'confidence_percentage': int,
            'scores': Dict[str, float]
        }
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "intent": "QUESTION_ANSWERING",
                "primary_intent": "QUESTION_ANSWERING",
                "confidence": 0.50,
                "confidence_percentage": 50,
                "scores": {c: round(1.0 / len(INTENT_CLASSES), 3) for c in INTENT_CLASSES}
            }

        self._ensure_anchors()

        # Generate normalized query vector
        q_vec = self.embedding_service.embed_query(clean_q)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm

        raw_similarities: Dict[str, float] = {}
        for intent, anchor_vec in self._anchor_vectors.items():
            sim = float(np.dot(q_vec, anchor_vec))
            # Bound cosine similarity
            raw_similarities[intent] = max(0.0, sim)

        # Apply softmax with temperature for calibrated probability distribution
        sim_values = np.array(list(raw_similarities.values()))
        temperature = 0.15
        exp_vals = np.exp((sim_values - np.max(sim_values)) / temperature)
        probs = exp_vals / np.sum(exp_vals)

        prob_dict = {
            intent: round(float(prob), 4)
            for intent, prob in zip(raw_similarities.keys(), probs)
        }

        primary_intent = max(prob_dict.items(), key=lambda x: x[1])[0]
        top_prob = prob_dict[primary_intent]

        return {
            "intent": primary_intent,
            "primary_intent": primary_intent,
            "confidence": top_prob,
            "confidence_percentage": int(round(top_prob * 100)),
            "scores": prob_dict
        }
