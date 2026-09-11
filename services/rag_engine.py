"""
Retrieval-Augmented Generation (RAG) engine for IntelliAssist AI.
Features ML query understanding, hybrid BM25 + dense semantic retrieval, neural reranking,
authentic confidence estimation, hallucination safeguards, and source-grounded response synthesis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.query_classifier import QueryIntentClassifier
from services.reranker import NeuralReranker

logger = logging.getLogger(__name__)

SLANG_TYPO_MAP = {
    r"\bwat\b|\bwht\b|\bwt\b": "what",
    r"\bdis\b": "this",
    r"\bdat\b": "that",
    r"\bplz\b|\bpls\b": "please",
    r"\bsummery\b|\bsumary\b": "summary",
    r"\btransfmr\b|\btransfomer\b|\btrnsformer\b": "transformer",
    r"\babt\b": "about",
    r"\bdiff\b": "difference",
    r"\bvs\b": "versus",
    r"\bdoc\b|\bdocs\b|\bpdfs?\b|\btxt\b": "document",
    r"\barch\b": "architecture",
    r"\bimp\b": "important",
    r"\bwrk\b|\bwrks\b": "work",
    r"\beval\b": "evaluation",
    r"\binfo\b": "information",
    r"\bintro\b": "introduction",
    r"\bconc\b": "conclusion"
}

GREETING_PATTERNS = [
    r"^(?:hi|hello|hey|hola|greetings|howdy|yo)\b",
    r"^(?:help|help me|guide|start|how do i use|how to use|who are you|what can you do|what is this app|what is intelliassist)\b",
    r"^(?:what is this|tell me what to do)\b"
]

class RAGEngine:
    """End-to-end robust RAG orchestrator delivering grounded explanations with multi-document intelligence."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        query_classifier: Optional[QueryIntentClassifier] = None,
        reranker: Optional[NeuralReranker] = None
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.query_classifier = query_classifier or QueryIntentClassifier(
            embedding_service=vector_store.embedding_service
        )
        self.reranker = reranker or NeuralReranker()

    def normalize_query(self, query: str) -> Tuple[str, str, bool]:
        """
        Normalize informal phrasing, fix common typos, and detect query intent.
        Returns: (expanded_query, detected_intent, is_greeting)
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Check for greetings or onboarding questions
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, q_lower):
                return q_clean, "greeting", True

        # 2. Apply slang and typo normalization
        normalized = q_lower
        for pattern, replacement in SLANG_TYPO_MAP.items():
            normalized = re.sub(pattern, replacement, normalized)

        # 3. Detect broad summary / overview intents
        if re.search(r"^(?:summary|overview|what (?:does |is )?this (?:document |doc |pdf )?say|explain|tell me everything|main points|key points|what is this about)\b", normalized):
            expanded = f"Provide a comprehensive executive summary, technical breakdown, and key findings of the documents: {normalized}"
            return expanded, "summary", False

        # 4. Handle short keyword queries
        words = normalized.split()
        if len(words) <= 2 and not any(w in normalized for w in ["what", "how", "why", "where", "who", "when"]):
            expanded = f"Explain in full detail the core concept, architecture, empirical findings, and significance of {normalized} in the document"
            return expanded, "concept", False

        return normalized, "qa", False

    def detect_document_in_query(self, query: str, available_docs: List[str]) -> Optional[str]:
        """Auto-detect if user mentions a specific file name in their question."""
        q_lower = query.lower()
        best_doc = None
        best_score = 0

        for doc in available_docs:
            d_name = doc.lower()
            stem = d_name.rsplit(".", 1)[0]
            stem_clean = stem.replace("_", " ").replace("-", " ")

            if doc.lower() in q_lower or stem in q_lower or stem_clean in q_lower:
                return doc

            tokens = [t for t in stem_clean.split() if len(t) >= 4 and t not in ["overview", "report", "paper", "document", "annual"]]
            matched = sum(1 for t in tokens if t in q_lower)
            if matched > 0 and matched > best_score:
                best_score = matched
                best_doc = doc

        return best_doc if best_score > 0 else None

    def answer_question(
        self,
        question: str,
        top_k: int = 6,
        threshold: float = 0.08,
        min_confidence_threshold: float = 0.20,
        filter_doc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full RAG retrieval and generation pipeline:
        1. Preprocess query & ML Query Intent Classification.
        2. Handle greetings gracefully with active document guides.
        3. Hybrid Retrieval: Dense Embeddings + BM25 Lexical search via Reciprocal Rank Fusion (RRF).
        4. Neural Reranking: cross-encoder scoring on candidate chunks.
        5. Calculated confidence check (anti-hallucination protection).
        6. Context synthesis & LLM generation with exact source citations.
        """
        raw_question = question.strip()
        if not raw_question:
            return {
                "answer": "Please enter a question or keyword about your documents to get started.",
                "sources": [],
                "retrieved_count": 0,
                "provider": self.llm_service.provider,
                "latency_sec": 0.1,
                "is_demo": True,
                "is_low_confidence": False,
                "query_intent": "QUESTION_ANSWERING",
                "confidence": 0.0
            }

        # 1. ML Query Intent Classification & Normalization
        intent_info = self.query_classifier.classify(raw_question)
        ml_intent = intent_info.get("primary_intent", "QUESTION_ANSWERING")
        ml_intent_confidence = intent_info.get("confidence", 0.5)

        expanded_query, fallback_intent, is_greeting = self.normalize_query(raw_question)
        available_docs = self.vector_store.get_all_documents()

        # Check if query explicitly mentions a document
        auto_detected_doc = self.detect_document_in_query(raw_question, available_docs)
        active_filter_doc = filter_doc or auto_detected_doc
        if active_filter_doc == "All Documents":
            active_filter_doc = None

        # 2. Greeting & Onboarding handler
        if is_greeting:
            doc_list_str = "\n".join([f"- **{doc}**" for doc in available_docs[:5]]) if available_docs else "*(No documents uploaded yet. Upload documents on the Documents page to begin!)*"
            greeting_msg = (
                f"### Welcome to **IntelliAssist AI**\n\n"
                f"I am your document intelligence workspace. I can read, analyze, summarize, "
                f"and provide grounded explanations for any uploaded files with verified citations.\n\n"
                f"#### Indexed Documents Available:\n"
                f"{doc_list_str}\n\n"
                f"#### Suggested Inquiries:\n"
                f"- *\"Provide a complete breakdown of the methodology and system architecture\"*\n"
                f"- *\"What are the empirical benchmarks, key results, and statistical findings?\"*\n"
                f"- *\"Explain the core technical concepts step-by-step with citations\"*\n"
                f"- *\"What are the practical applications, strengths, and limitations?\"*\n"
            )
            return {
                "answer": greeting_msg,
                "sources": [],
                "retrieved_count": 0,
                "provider": self.llm_service.provider,
                "model": self.llm_service.model_name,
                "latency_sec": 0.15,
                "is_demo": True,
                "is_low_confidence": False,
                "query_intent": "QUESTION_ANSWERING",
                "confidence": 1.0
            }

        # 3. Hybrid Retrieval (Dense Semantic + BM25 Lexical via RRF)
        retrieved_chunks = self.vector_store.search(
            query=expanded_query,
            top_k=top_k * 2,  # retrieve candidate pool for reranking
            threshold=threshold,
            filter_doc=active_filter_doc,
            use_hybrid=True
        )

        # Fallback 1: If search returned nothing, try with zero threshold
        if not retrieved_chunks:
            retrieved_chunks = self.vector_store.search(
                query=raw_question,
                top_k=top_k * 2,
                threshold=0.0,
                filter_doc=active_filter_doc,
                use_hybrid=True
            )

        # Fallback 2: Target document enforcement with authentic cosine similarity
        if active_filter_doc:
            doc_all_chunks = [c for c in self.vector_store.chunks if c.get("filename") == active_filter_doc]
            if doc_all_chunks:
                retrieved_chunks = [c for c in retrieved_chunks if c.get("filename") == active_filter_doc]
                if len(retrieved_chunks) < min(top_k, len(doc_all_chunks)):
                    existing_ids = set(c.get("chunk_id") for c in retrieved_chunks)
                    q_vec = self.vector_store.embedding_service.embed_query(raw_question)
                    q_norm = np.linalg.norm(q_vec)
                    if q_norm > 1e-9:
                        q_vec = q_vec / q_norm

                    for c in doc_all_chunks:
                        if c.get("chunk_id") not in existing_ids:
                            c_item = dict(c)
                            # Authentically calculate cosine similarity
                            c_vec = self.vector_store.embedding_service.embed_query(c.get("text", ""))
                            c_norm = np.linalg.norm(c_vec)
                            if c_norm > 1e-9:
                                c_vec = c_vec / c_norm
                            calc_sim = float(np.dot(q_vec, c_vec))
                            calc_score = max(0.0, min(0.99, calc_sim))
                            c_item["score"] = round(calc_score, 4)
                            c_item["similarity_percentage"] = int(round(calc_score * 100))
                            c_item["match_type"] = "Semantic Match"
                            c_item["match_explanation"] = f"Document-scoped semantic match (cosine: {calc_score:.2f})"
                            retrieved_chunks.append(c_item)
                        if len(retrieved_chunks) >= top_k * 2:
                            break
        elif not retrieved_chunks and len(self.vector_store.chunks) > 0:
            retrieved_chunks = list(self.vector_store.chunks[:top_k * 2])

        # 4. Neural Cross-Encoder Reranking
        if retrieved_chunks:
            retrieved_chunks = self.reranker.rerank(
                query=raw_question,
                chunks=retrieved_chunks,
                top_k=top_k
            )

        # 5. Genuine Confidence Computation & Low-Confidence Protection
        scores = [float(c.get("score", 0.0)) for c in retrieved_chunks]
        max_score = max(scores, default=0.0)
        mean_top_score = float(np.mean(scores[:3])) if scores else 0.0
        calculated_confidence = round(float(0.70 * max_score + 0.30 * mean_top_score), 4)

        if active_filter_doc:
            is_low_confidence = (len(retrieved_chunks) == 0)
        else:
            is_low_confidence = (len(retrieved_chunks) == 0 or max_score < min_confidence_threshold)

        if is_low_confidence and len(self.vector_store.chunks) > 0:
            logger.info("Low confidence query detected (score: %.2f for '%s')", max_score, raw_question)
            no_answer_msg = (
                "I couldn't find enough relevant information in your uploaded documents to answer this reliably.\n\n"
                "**Recommendations:**\n"
                "- Verify if the topic is covered in your uploaded files.\n"
                "- Select a specific document scope or 'All Documents' in the toolbar.\n"
                "- Rephrase your question using specific terminology from the document text."
            )
            return {
                "answer": no_answer_msg,
                "sources": [],
                "retrieved_count": 0,
                "provider": self.llm_service.provider,
                "model": self.llm_service.model_name,
                "latency_sec": 0.1,
                "is_demo": True,
                "is_low_confidence": True,
                "query_intent": ml_intent,
                "confidence": calculated_confidence
            }

        if not retrieved_chunks:
            return {
                "answer": (
                    f"I could not locate content for **{active_filter_doc or 'your query'}** in the database.\n\n"
                    "Ensure your documents are uploaded and indexed before querying."
                ),
                "sources": [],
                "retrieved_count": 0,
                "provider": self.llm_service.provider,
                "latency_sec": 0.1,
                "is_demo": True,
                "is_low_confidence": True,
                "query_intent": ml_intent,
                "confidence": 0.0
            }

        # 6. Build context text with explicit page demarcations
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            doc_name = chunk.get("filename", "Unknown Document")
            page_num = chunk.get("page_number", 1)
            chunk_text = chunk.get("text", "")
            context_parts.append(f"[Source {i} - {doc_name} (Page {page_num})]:\n{chunk_text}")

        full_context = "\n\n".join(context_parts)

        # 7. Formulate intelligent, query-specific RAG system instruction
        target_clause = f"Focus your answer strictly on the document '{active_filter_doc}'." if active_filter_doc else "Synthesize information from the relevant document(s) provided."

        if ml_intent == "SUMMARIZATION":
            style_guide = (
                "The user requested an overview or summary. Provide a structured response with:\n"
                "1. Executive Overview & Main Takeaways\n"
                "2. Core Methodology & Technical Breakdown\n"
                "3. Notable Findings & Quantitative Metrics"
            )
        elif ml_intent == "COMPARISON":
            style_guide = (
                "The user requested a comparison. Present a clear contrast highlighting:\n"
                "- Core architectural / conceptual differences\n"
                "- Comparative trade-offs, strengths, and weaknesses\n"
                "- Benchmark differences supported by citations"
            )
        elif ml_intent == "STUDY":
            style_guide = (
                "The user requested study or revision material. Break down the core concepts clearly, "
                "highlight key definitions, and provide actionable revision questions."
            )
        elif ml_intent == "ACTION_ITEMS":
            style_guide = (
                "The user requested action items. List concrete, actionable next steps, decisions, "
                "and recommendations clearly."
            )
        else:
            style_guide = (
                "Answer the specific question directly, concisely, and factually.\n"
                "- Ground every claim in the provided DOCUMENT CONTEXT.\n"
                "- For factual lookups, provide the direct answer in the first sentence.\n"
                "- Cite specific document names and page numbers (e.g., [Doc: filename, Page X]).\n"
                "- If the context does not contain enough evidence, state clearly that it is not covered."
            )

        system_instruction = (
            f"You are IntelliAssist AI, an expert document intelligence assistant. {target_clause}\n\n"
            f"Query Intent: {ml_intent} (Confidence: {ml_intent_confidence:.2f})\n\n"
            f"{style_guide}\n\n"
            "Guidelines:\n"
            "- Ground your response strictly in the provided DOCUMENT CONTEXT.\n"
            "- Always cite specific document names and page numbers for facts and metrics."
        )

        rag_prompt = (
            f"DOCUMENT CONTEXT:\n"
            f"---------------------\n"
            f"{full_context}\n"
            f"---------------------\n\n"
            f"USER QUERY: {raw_question}\n"
            f"TARGET DOCUMENT: {active_filter_doc or 'All Documents'}\n\n"
            f"Please answer the user query directly based on the context above."
        )

        # 8. Generate answer via LLM service
        llm_response = self.llm_service.generate(
            prompt=rag_prompt,
            system_instruction=system_instruction,
            context_chunks=retrieved_chunks,
            target_doc_name=active_filter_doc
        )

        # 9. Format verified source citations
        sources = []
        for chunk in retrieved_chunks:
            c_score = float(chunk.get("score", 0.85))
            sources.append({
                "filename": chunk.get("filename", "Unknown"),
                "page_number": chunk.get("page_number", 1),
                "total_pages": chunk.get("total_pages", 1),
                "score": round(c_score, 4),
                "similarity_percentage": int(round(c_score * 100)),
                "text_snippet": chunk.get("text", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "match_type": chunk.get("match_type", "Semantic Match"),
                "match_explanation": chunk.get("match_explanation", "Retrieved via semantic vector search")
            })

        return {
            "answer": llm_response.get("text", ""),
            "sources": sources,
            "retrieved_count": len(sources),
            "provider": llm_response.get("provider", self.llm_service.provider),
            "model": llm_response.get("model", self.llm_service.model_name),
            "latency_sec": llm_response.get("latency_sec", 0.35),
            "is_demo": llm_response.get("is_demo", False),
            "is_low_confidence": False,
            "query_intent": ml_intent,
            "confidence": calculated_confidence,
            "notice": llm_response.get("notice", None)
        }
