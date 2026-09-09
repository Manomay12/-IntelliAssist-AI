"""
Dynamic AI Question Generator for IntelliAssist AI.
Generates document-specific, content-grounded exploratory questions for Semantic Search and AI Chat,
with rotation, variety, and shuffling capabilities so users receive fresh suggestions each time.
"""

import re
import random
from typing import List, Dict, Any, Optional

class QuestionGenerator:
    """Generates dynamic, content-aware suggested questions for any document."""

    # Curated knowledge seeds for standard benchmark documents
    KNOWN_DOC_POOLS = {
        "attention": [
            "What is the core mathematical formulation of Scaled Dot-Product Attention?",
            "How does Multi-Head Attention allow the model to jointly attend to information from different representation subspaces?",
            "Why does the Transformer eliminate recurrent LSTM/GRU layers entirely?",
            "What BLEU score was achieved on the WMT 2014 English-to-German translation task?",
            "How do sinusoidal Positional Encodings inject sequence order into the attention layers?",
            "What were the GPU training times and hardware specifications for base and big models?",
            "How does masked multi-head attention in the decoder preserve the auto-regressive property?",
            "What advantages does self-attention provide regarding maximum path lengths across long sequences?",
            "How does label smoothing affect the model's perplexity and translation quality?",
            "What dropout rates and optimizer settings (Adam) were used during training?"
        ],
        "rag": [
            "How does Retrieval-Augmented Generation combine parametric and non-parametric memory?",
            "What is the difference between RAG-Sequence and RAG-Token formulations?",
            "Which dense neural retriever and pre-trained generator models are utilized?",
            "What empirical benchmarks were evaluated (e.g. Natural Questions, TriviaQA, CuratedTREC)?",
            "How does RAG compare against closed-book parametric models like T5 and GPT-3?",
            "How is the Wikipedia knowledge index indexed and searched using FAISS?",
            "How does the marginalization over retrieved latent documents work during generation?",
            "What benefits does RAG provide in mitigating factual hallucinations in NLP models?",
            "Can RAG update its world knowledge base without retraining the underlying language model?",
            "How does RAG perform on open-domain Jeopardy question answering?"
        ],
        "mimic": [
            "What clinical mortality and sepsis prediction models were evaluated in the study?",
            "What features and vital signs from the MIMIC-III database were extracted for patient cohort analysis?",
            "What AUROC and AUPRC metrics were achieved by the predictive algorithms?",
            "How was early detection of septic shock handled within the ICU monitoring timeframe?",
            "What data preprocessing and missing value imputation techniques were applied to ICU timeseries?",
            "How did deep learning architectures (e.g. LSTM/GRU) compare against baseline clinical scores (SOFA, SAPS-II)?",
            "What are the ethical considerations and limitations of deploying predictive models in intensive care?",
            "How did the models handle patient demographic balance and physiological variability?",
            "What were the lead time windows for predicting onset of critical deterioration?",
            "What feature importance rankings were identified for acute kidney injury and septic shock?"
        ]
    }

    # Universal exploration templates for any generic or custom uploaded document
    UNIVERSAL_POOLS = [
        "What are the primary conclusions and executive takeaways presented across all indexed files?",
        "Compare the technical methodologies and system architectures described in the documents.",
        "What empirical metrics, accuracy benchmarks, or performance scores are highlighted?",
        "Explain the most complex domain concepts mentioned here in clear, accessible terms.",
        "What critical limitations, assumptions, and future research directions are acknowledged?",
        "How do the findings in these files apply to practical real-world implementation?",
        "What datasets, experimental setups, or baseline comparisons were used for evaluation?",
        "Summarize the key differences and commonalities between the uploaded resources."
    ]

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            random.seed(random_seed)

    def generate_questions(
        self,
        doc_name: Optional[str] = None,
        doc_text: Optional[str] = None,
        count: int = 4,
        shuffle_seed: Optional[int] = None
    ) -> List[str]:
        """
        Generate a list of count distinct, content-relevant questions.
        If shuffle_seed is provided, uses it to vary the selection deterministically.
        """
        rng = random.Random(shuffle_seed) if shuffle_seed is not None else random

        # Case 1: All Documents / No specific document selected
        if not doc_name or doc_name.strip() in ["All Documents", "-- Select a document to summarize --", ""]:
            pool = list(self.UNIVERSAL_POOLS)
            rng.shuffle(pool)
            return pool[:count]

        # Case 2: Check for known benchmark document matches
        doc_lower = doc_name.lower()
        matched_pool: Optional[List[str]] = None
        for key, pool in self.KNOWN_DOC_POOLS.items():
            if key in doc_lower:
                matched_pool = list(pool)
                break

        if matched_pool:
            rng.shuffle(matched_pool)
            return matched_pool[:count]

        # Case 3: Dynamic content extraction from the document's actual text
        pool = self._extract_questions_from_text(doc_name, doc_text or "")
        if len(pool) < count:
            clean_name = doc_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            fallbacks = [
                f"What is the main objective and central thesis of {clean_name}?",
                f"How does {clean_name} approach its core methodology or problem statement?",
                f"What key metrics, results, or discoveries are documented in {clean_name}?",
                f"What challenges, limitations, or risks are identified in {clean_name}?",
                f"Summarize the most impactful recommendations provided in {clean_name}."
            ]
            for fb in fallbacks:
                if fb not in pool:
                    pool.append(fb)

        rng.shuffle(pool)
        return pool[:count]

    def _extract_questions_from_text(self, doc_name: str, text: str) -> List[str]:
        """Extract content-specific questions by analyzing sentences, headings, and numerical findings."""
        questions: List[str] = []
        clean_name = doc_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")

        if not text or len(text.strip()) < 50:
            return [
                f"What is the executive summary of {clean_name}?",
                f"What are the key findings and conclusions in {clean_name}?",
                f"Explain the technical concepts described in {clean_name}.",
                f"What are the notable limitations and future steps outlined in {clean_name}?"
            ]

        # 1. Extract potential section headings or capitalized titles
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:25]:
            if 10 < len(line) < 60 and not line.endswith(".") and re.match(r'^[A-Z0-9]', line):
                heading = line.strip("#-* ")
                if len(heading) > 10:
                    questions.append(f"What does {clean_name} state regarding '{heading}'?")

        # 2. Extract sentences with percentages, metrics, or performance numbers
        metric_matches = re.findall(r'([^.?!;\n]*?\b(?:\d+(?:\.\d+)?%|\b(?:BLEU|AUROC|AUPRC|accuracy|latency|F1|score)\b)[^.?!;\n]*)', text, re.IGNORECASE)
        for m in metric_matches[:5]:
            snippet = m.strip()
            if 20 < len(snippet) < 100:
                questions.append(f"How was the result '{snippet[:60]}...' measured and evaluated?")

        # 3. Extract methodology/architecture references
        method_matches = re.findall(r'\b(?:we propose|we introduce|we design|our approach|algorithm|architecture|framework|model)\b([^.?!;\n]{15,90})', text, re.IGNORECASE)
        for m in method_matches[:4]:
            phrase = m.strip()
            if len(phrase) > 15:
                questions.append(f"How does the system design for '{phrase[:55]}...' function?")

        # 4. Extract conclusions/limitations
        crit_matches = re.findall(r'\b(?:limitation|future work|challenge|bottleneck|drawback|drawbacks|conclusion)\b([^.?!;\n]{15,90})', text, re.IGNORECASE)
        for m in crit_matches[:4]:
            phrase = m.strip()
            if len(phrase) > 15:
                questions.append(f"What is discussed regarding the limitation of '{phrase[:55]}...'?")

        # Deduplicate while preserving order
        unique_q: List[str] = []
        for q in questions:
            if q not in unique_q:
                unique_q.append(q)

        # Baseline contextual questions
        baseline = [
            f"What is the primary contribution and novelty in {clean_name}?",
            f"What experimental baseline methods are compared against {clean_name}?",
            f"What are the practical applications and impact of {clean_name}?",
            f"What are the critical architectural assumptions made in {clean_name}?"
        ]
        for b in baseline:
            if b not in unique_q:
                unique_q.append(b)

        return unique_q
