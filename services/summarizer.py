"""
Document summarization, topic modeling, entity extraction, and cross-document comparison service for IntelliAssist AI.
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """Provides multi-mode document summarization, entity extraction, and cross-document comparative analysis."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def extract_key_topics(self, text: str, max_topics: int = 8) -> List[str]:
        """Extract dominant keywords and key phrases using frequency and stopword filtering."""
        if not text:
            return []

        stopwords = {
            "the", "and", "is", "in", "it", "of", "to", "for", "with", "on", "that", "this",
            "are", "was", "as", "by", "an", "be", "from", "at", "or", "which", "will", "from",
            "can", "has", "have", "been", "were", "their", "such", "other", "into", "more",
            "document", "page", "using", "also", "used", "than", "first", "based", "system",
            "study", "paper", "data", "these", "about", "could", "should", "would", "between",
            "each", "when", "where", "after", "before", "over", "under", "both", "some", "most"
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        bigrams = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 not in stopwords and w2 not in stopwords:
                bigrams.append(f"{w1.title()} {w2.title()}")

        filtered_words = [w.title() for w in words if w not in stopwords and len(w) > 4]

        counts_bi = Counter(bigrams).most_common(max_topics // 2)
        counts_single = Counter(filtered_words).most_common(max_topics // 2)

        topics = [item[0] for item in counts_bi] + [item[0] for item in counts_single]
        unique_topics = []
        for t in topics:
            if t not in unique_topics:
                unique_topics.append(t)

        return unique_topics[:max_topics]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract important entities (Technologies, Organizations, Metrics/Numbers, Dates/Years)."""
        entities = {
            "Technologies & Frameworks": [],
            "Organizations & Roles": [],
            "Metrics & Percentages": [],
            "Dates & Years": []
        }

        metrics = re.findall(r'\b(?:\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:MB|GB|ms|seconds|accuracy|F1|BLEU|parameters|users|requests))\b', text, re.IGNORECASE)
        entities["Metrics & Percentages"] = list(set(metrics))[:6]

        years = re.findall(r'\b(?:19\d\d|20\d\d|January|February|March|April|May|June|July|August|September|October|November|December)\b', text)
        entities["Dates & Years"] = list(set(years))[:5]

        tech_keywords = [
            "Python", "PyTorch", "TensorFlow", "LangChain", "Streamlit", "Transformers", "BERT",
            "FAISS", "ChromaDB", "LLM", "GPT", "Gemini", "RAG", "Vector Database", "NLP", "API",
            "REST", "Docker", "SQL", "Deep Learning", "Machine Learning", "Attention Mechanism",
            "Cosine Similarity", "Neural Network", "CNN", "RNN", "Embedding", "Encoder", "Decoder",
            "Pandas", "Scikit-Learn", "FastAPI", "React", "Node.js", "Kubernetes", "AWS", "Azure"
        ]
        found_tech = [k for k in tech_keywords if re.search(r'\b' + re.escape(k) + r'\b', text, re.IGNORECASE)]
        entities["Technologies & Frameworks"] = found_tech[:8]

        org_keywords = [
            "Google", "OpenAI", "DeepMind", "Microsoft", "Meta", "Amazon", "IEEE", "Stanford",
            "MIT", "Hugging Face", "University", "Research Team", "Enterprise", "Developer", "Client",
            "Author", "Investigator", "Clinician", "Analyst"
        ]
        found_orgs = [k for k in org_keywords if re.search(r'\b' + re.escape(k) + r'\b', text, re.IGNORECASE)]
        entities["Organizations & Roles"] = found_orgs[:6]

        return entities

    def _generate_smart_local_summary(self, text: str, mode: str, doc_name: str) -> str:
        """
        Generate high-quality, mode-specific structured summary locally
        using sentence classification, keyword extraction, and tailored layouts.
        """
        raw_sentences = re.split(r'(?<=[.?!])\s+', text)
        sentences = [s.strip().replace("\n", " ") for s in raw_sentences if len(s.strip()) > 20]

        if not sentences:
            return f"**Summary of {doc_name}**\n\nThe document contains minimal textual content."

        # Deduplicate sentences while preserving chronological sequence
        seen = set()
        clean_sentences = []
        for s in sentences:
            k = s.lower()[:45]
            if k not in seen:
                seen.add(k)
                clean_sentences.append(s)

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)

        # Scored sentences by word importance
        scored_sentences = []
        for i, s in enumerate(clean_sentences):
            s_words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            if not s_words:
                continue
            score = sum(word_freq.get(w, 0) for w in s_words) / len(s_words)
            if i < 4:
                score *= 1.35
            scored_sentences.append((score, i, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_ranked = [item[2] for item in scored_sentences]

        # Categorize sentences into functional roles
        metric_pattern = re.compile(
            r'\b(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:BLEU|F1|accuracy|ms|seconds|GB|MB|parameters|users|requests|GPUs?|days?|times?|higher|lower|improvement))\b|'
            r'\b(?:achiev|outperform|increas|decreas|result|benchmark|state-of-the-art|sota|evaluat)\b',
            re.IGNORECASE
        )
        intro_pattern = re.compile(
            r'\b(?:introduc|propos|present|focus|aim|objectiv|paper|document|overview|background|design|address|we build|we study)\b',
            re.IGNORECASE
        )
        tech_pattern = re.compile(
            r'\b(?:architecture|mechanism|model|algorithm|transformer|attention|neural|network|layer|embedding|encoder|decoder|pipeline|methodology|technique|framework|retriev)\b',
            re.IGNORECASE
        )
        takeaway_pattern = re.compile(
            r'\b(?:conclud|demonstrat|show|indicat|highlight|implication|takeaway|futur|significan|impact|recommend|promis|crucial|superior)\b',
            re.IGNORECASE
        )

        metric_sents = [s for s in clean_sentences if metric_pattern.search(s)]
        intro_sents = [s for s in clean_sentences if intro_pattern.search(s)]
        tech_sents = [s for s in clean_sentences if tech_pattern.search(s)]
        takeaway_sents = [s for s in clean_sentences if takeaway_pattern.search(s)]

        # Fallbacks if specific category lists are sparse
        if not intro_sents:
            intro_sents = clean_sentences[:2]
        if not metric_sents:
            metric_sents = top_ranked[:4]
        if not tech_sents:
            tech_sents = clean_sentences[1:4] if len(clean_sentences) > 3 else clean_sentences
        if not takeaway_sents:
            takeaway_sents = clean_sentences[-3:] if len(clean_sentences) >= 3 else clean_sentences

        topics = self.extract_key_topics(text, max_topics=5)
        topics_str = ", ".join(topics) if topics else "Document Analysis"
        entities = self.extract_entities(text)

        # -------------------------------------------------------------
        # MODE 1: Quick Summary (Ultra-concise TL;DR, <100 words)
        # -------------------------------------------------------------
        if mode == "Quick Summary":
            lead = intro_sents[0] if intro_sents else clean_sentences[0]
            core_finding = metric_sents[0] if metric_sents and metric_sents[0] != lead else (tech_sents[0] if tech_sents else "")
            takeaway = takeaway_sents[0] if takeaway_sents and takeaway_sents[0] not in [lead, core_finding] else (clean_sentences[-1] if len(clean_sentences) > 1 else lead)

            return (
                f"### ⚡ Quick Summary: {doc_name}\n\n"
                f"{lead} {core_finding}\n\n"
                f"> 💡 **Core Takeaway**: {takeaway}\n\n"
                f"*Focus Domain: **{topics_str}***"
            )

        # -------------------------------------------------------------
        # MODE 2: Bullet Points (Categorized, rapid-skimming format)
        # -------------------------------------------------------------
        elif mode == "Bullet Points":
            lines = [
                f"### 📌 Skimmable Bullet Points: {doc_name}\n",
                f"*Key takeaways categorized by topic from automated NLP parsing:*\n",
                "#### 🎯 Core Scope & Objective"
            ]
            for s in intro_sents[:2]:
                lines.append(f"- {s}")

            lines.append("\n#### ⚙️ Technical Approach & Framework")
            for s in tech_sents[:2]:
                if s not in intro_sents[:2]:
                    lines.append(f"- {s}")
            if len(lines) == 5:  # ensure at least one tech bullet
                lines.append(f"- Utilizes advanced methodologies centered on {topics_str}.")

            lines.append("\n#### 📊 Key Results & Empirical Outcomes")
            added_metrics = 0
            for s in metric_sents[:3]:
                if s not in intro_sents[:2] and s not in tech_sents[:2]:
                    lines.append(f"- {s}")
                    added_metrics += 1
                if added_metrics >= 2:
                    break
            if added_metrics == 0 and top_ranked:
                lines.append(f"- {top_ranked[0]}")

            lines.append("\n#### 💡 Practical Implications")
            for s in takeaway_sents[:2]:
                if s not in lines:
                    lines.append(f"- {s}")
                    break
            else:
                lines.append(f"- Establishes critical practical benchmarks in **{topics_str}**.")

            return "\n".join(lines)

        # -------------------------------------------------------------
        # MODE 3: Key Findings (Empirical benchmarks, metrics, breakthroughs)
        # -------------------------------------------------------------
        elif mode == "Key Findings":
            lines = [
                f"### 🎯 Key Empirical Findings & Metrics: {doc_name}\n",
                f"Primary empirical discoveries, benchmark metrics, and findings extracted from **{doc_name}**:\n"
            ]
            findings_pool = []
            for s in metric_sents:
                if s not in findings_pool:
                    findings_pool.append(s)
            for s in top_ranked:
                if s not in findings_pool:
                    findings_pool.append(s)

            for idx, s in enumerate(findings_pool[:5], 1):
                lines.append(f"{idx}. **Empirical Finding {idx}**: {s}")

            metrics_list = entities.get("Metrics & Percentages", [])
            if metrics_list:
                metric_badges = " • ".join([f"`{m}`" for m in metrics_list[:6]])
                lines.append(f"\n> 📈 **Extracted Quantitative Metrics**: {metric_badges}")
            else:
                lines.append(f"\n> 📈 **Core Focus**: High statistical significance across **{topics_str}**.")

            return "\n".join(lines)

        # -------------------------------------------------------------
        # MODE 4: Executive Summary (C-Suite strategic briefing)
        # -------------------------------------------------------------
        elif mode == "Executive Summary":
            lead = intro_sents[0] if intro_sents else clean_sentences[0]
            val_sent = tech_sents[0] if tech_sents else (clean_sentences[1] if len(clean_sentences) > 1 else "")
            res_sent = metric_sents[0] if metric_sents else (top_ranked[1] if len(top_ranked) > 1 else "")
            takeaway = takeaway_sents[0] if takeaway_sents else clean_sentences[-1]

            lines = [
                f"### 📋 Executive Summary: {doc_name}\n",
                "#### 1. Strategic Context & Vision",
                f"{lead} Addressing core challenges in **{topics_str}**, this work outlines strategic workflows and modern capabilities.\n",
                "#### 2. Business & Operational Value",
                f"{val_sent} By standardizing technical components, organizations can optimize throughput and improve operational efficiency.",
                f"{res_sent}\n",
                "#### 3. Strategic Recommendations & Next Steps",
                f"- **Actionable Adoption**: Leverage the framework's core methodologies in **{topics_str}** to streamline technical execution.",
                f"- **Benchmark Verification**: Track continuous improvement metrics against the reported empirical baselines.",
                f"- **Implementation Strategy**: {takeaway}"
            ]
            return "\n".join(lines)

        # -------------------------------------------------------------
        # MODE 5: Detailed Summary (Comprehensive multi-section review)
        # -------------------------------------------------------------
        else:
            lead = intro_sents[0] if intro_sents else clean_sentences[0]
            sec_intro = intro_sents[1] if len(intro_sents) > 1 else (clean_sentences[1] if len(clean_sentences) > 1 else "")
            
            lines = [
                f"### 📑 Comprehensive Technical Summary: {doc_name}\n",
                "#### 1. 📖 Background & Problem Statement",
                f"{lead} {sec_intro}\n",
                "#### 2. 🔍 System Architecture & Technical Methodology"
            ]
            for s in tech_sents[:3]:
                if s != lead and s != sec_intro:
                    lines.append(f"- **Mechanics**: {s}")
            if len(lines) == 4 and clean_sentences:
                lines.append(f"- **Methodology**: {clean_sentences[1] if len(clean_sentences) > 1 else clean_sentences[0]}")

            lines.append("\n#### 3. 📊 Empirical Experiments, Benchmarks & Evidence")
            for idx, s in enumerate(metric_sents[:3], 1):
                lines.append(f"{idx}. {s}")
            if len(metric_sents) == 0 and top_ranked:
                lines.append(f"1. {top_ranked[0]}")

            lines.append("\n#### 4. 💡 Limitations, Critical Analysis & Future Directions")
            conc = takeaway_sents[0] if takeaway_sents else clean_sentences[-1]
            lines.append(f"{conc} Continued research in **{topics_str}** will focus on scaling, parameter efficiency, and expanded domain adaptation.")

            return "\n".join(lines)

    def summarize(self, text: str, mode: str = "Detailed Summary", doc_name: str = "Document") -> Dict[str, Any]:
        """Generate structured summary based on selected mode."""
        if not text or not text.strip():
            return {
                "mode": mode,
                "doc_name": doc_name,
                "summary": f"Document '{doc_name}' contains no readable text to summarize.",
                "topics": [],
                "entities": {},
                "takeaways": [],
                "is_demo": True,
                "provider": "IntelliAssist Local Engine"
            }

        topics = self.extract_key_topics(text)
        entities = self.extract_entities(text)

        # Mode-specific prompt specifications for external LLMs
        MODE_PROMPTS = {
            "Quick Summary": (
                f"Generate an ultra-concise 'Quick Summary' (2-3 sentences max) for the document named '{doc_name}'. "
                f"Explain what this document is about, its primary innovation or result, and why it matters. "
                f"End with a single '> 💡 **Core Takeaway**:' callout. Keep total length under 120 words. Do NOT generate long bullet lists."
            ),
            "Bullet Points": (
                f"Generate a structured 'Bullet Points' summary for the document named '{doc_name}'. "
                f"Group strictly into categorized, skimmable bullet points:\n"
                f"- **Core Scope & Objective** (2-3 crisp bullets)\n"
                f"- **Technical Approach & Framework** (2-3 crisp bullets)\n"
                f"- **Key Results & Metrics** (2-3 crisp bullets)\n"
                f"- **Actionable Takeaways** (1-2 crisp bullets)\n"
                f"Do NOT write narrative paragraphs. Keep each bullet point punchy and direct."
            ),
            "Key Findings": (
                f"Generate a 'Key Findings' summary report for the document named '{doc_name}' focused strictly on empirical results, discoveries, and metrics.\n"
                f"Format as:\n"
                f"1. A numbered list of 4-6 primary empirical findings (each with a bold title, exact numbers/percentages, and citation).\n"
                f"2. A '> 📊 **Quantitative Highlights**:' callout summarizing key percentages, speedups, or benchmark scores.\n"
                f"Focus strictly on results and outcomes rather than general background."
            ),
            "Executive Summary": (
                f"Generate a strategic 'Executive Summary' for '{doc_name}' tailored for executive leadership and C-suite decision-makers.\n"
                f"Format with:\n"
                f"#### 1. Strategic Context & Vision (concise context and core purpose)\n"
                f"#### 2. Business & Operational Value (high-level impact, efficiency gains, strategic benefits)\n"
                f"#### 3. Strategic Recommendations (3 actionable takeaways for implementation)\n"
                f"Avoid low-level code; focus on high-level strategy, practical impact, and next steps."
            ),
            "Detailed Summary": (
                f"Generate an exhaustive, in-depth 'Detailed Summary' for '{doc_name}'.\n"
                f"Format with:\n"
                f"#### 1. 📖 Background & Problem Statement\n"
                f"#### 2. 🔍 System Architecture & Technical Methodology\n"
                f"#### 3. 📊 Empirical Experiments, Benchmarks & Evidence\n"
                f"#### 4. 💡 Limitations, Critical Analysis & Future Directions\n"
                f"Provide thorough explanations, mechanisms, and concrete data points."
            )
        }

        # Check if LLM provider has an API key configured
        has_llm_key = bool(self.llm_service.api_key and any(p in self.llm_service.provider for p in ["Gemini", "OpenAI", "NVIDIA"]))

        if has_llm_key:
            try:
                truncated_text = text[:6500]
                mode_instruction = MODE_PROMPTS.get(mode, MODE_PROMPTS["Detailed Summary"])
                prompt = (
                    f"{mode_instruction}\n\n"
                    f"DOCUMENT CONTENT:\n{truncated_text}"
                )
                system_instruction = (
                    f"You are an expert document intelligence researcher. Your task is to generate a distinct '{mode}' summary. "
                    f"Adhere strictly to the requested structure, tone, and length for this specific style mode."
                )
                llm_resp = self.llm_service.generate(prompt=prompt, system_instruction=system_instruction, target_doc_name=doc_name)
                is_demo = llm_resp.get("is_demo", False)
                if is_demo:
                    summary_text = self._generate_smart_local_summary(text, mode, doc_name)
                    provider = "IntelliAssist Smart NLP Engine (Demo Mode)"
                else:
                    summary_text = llm_resp.get("text", "")
                    provider = llm_resp.get("provider", "LLM")
                    if not summary_text or len(summary_text) < 30:
                        raise ValueError("LLM returned empty summary text.")
            except Exception as e:
                logger.warning("LLM summarization failed (%s), using local fallback", e)
                summary_text = self._generate_smart_local_summary(text, mode, doc_name)
                is_demo = True
                provider = "IntelliAssist Smart NLP Engine (Demo Mode)"
        else:
            summary_text = self._generate_smart_local_summary(text, mode, doc_name)
            is_demo = True
            provider = "IntelliAssist Smart NLP Engine (Demo Mode)"

        sentences = [s.strip().replace("\n", " ") for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 35]
        takeaways = []
        for s in sentences:
            if any(w in s.lower() for w in ["conclude", "result", "achieve", "propose", "demonstrate", "show", "finding", "important", "improve", "significantly"]):
                takeaways.append(s)
            if len(takeaways) >= 4:
                break

        if len(takeaways) < 3 and len(sentences) >= 3:
            takeaways = sentences[:4]

        return {
            "mode": mode,
            "doc_name": doc_name,
            "summary": summary_text,
            "topics": topics,
            "entities": entities,
            "takeaways": takeaways,
            "is_demo": is_demo,
            "provider": provider
        }

    def compare_documents(
        self,
        doc_a_name: str,
        doc_a_text: str,
        doc_b_name: str,
        doc_b_text: str
    ) -> Dict[str, Any]:
        """
        Perform in-depth cross-document comparison between Document A and Document B.
        Generates structured comparative matrix (Markdown table) and synthesized evaluation.
        """
        topics_a = self.extract_key_topics(doc_a_text, max_topics=5)
        topics_b = self.extract_key_topics(doc_b_text, max_topics=5)
        entities_a = self.extract_entities(doc_a_text)
        entities_b = self.extract_entities(doc_b_text)

        # Build concise excerpt profiles
        sentences_a = [s.strip().replace("\n", " ") for s in re.split(r'(?<=[.?!])\s+', doc_a_text) if len(s.strip()) > 25]
        sentences_b = [s.strip().replace("\n", " ") for s in re.split(r'(?<=[.?!])\s+', doc_b_text) if len(s.strip()) > 25]

        lead_a = sentences_a[0] if sentences_a else "Overview of Document A"
        lead_b = sentences_b[0] if sentences_b else "Overview of Document B"
        method_a = sentences_a[1] if len(sentences_a) > 1 else "Empirical analysis"
        method_b = sentences_b[1] if len(sentences_b) > 1 else "Empirical analysis"
        metrics_a = ", ".join(entities_a.get("Metrics & Percentages", [])) or "Standard evaluation metrics"
        metrics_b = ", ".join(entities_b.get("Metrics & Percentages", [])) or "Standard evaluation metrics"
        tech_a = ", ".join(entities_a.get("Technologies & Frameworks", [])) or "NLP / AI Pipelines"
        tech_b = ", ".join(entities_b.get("Technologies & Frameworks", [])) or "NLP / AI Pipelines"

        # Construct Markdown Comparison Table
        comparison_table = (
            f"| Comparative Dimension | 📄 {doc_a_name} | 📄 {doc_b_name} |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Primary Topic & Domain** | {', '.join(topics_a)} | {', '.join(topics_b)} |\n"
            f"| **Core Objective** | {lead_a[:140]}... | {lead_b[:140]}... |\n"
            f"| **Methodology & Architecture** | {method_a[:140]}... | {method_b[:140]}... |\n"
            f"| **Technologies & Frameworks** | {tech_a} | {tech_b} |\n"
            f"| **Key Metrics / Benchmarks** | {metrics_a} | {metrics_b} |\n"
            f"| **Document Length** | {len(doc_a_text.split()):,} words | {len(doc_b_text.split()):,} words |\n"
        )

        # Executive Comparative Synthesis
        executive_synthesis = (
            f"### ⚖️ Executive Cross-Document Comparison\n\n"
            f"#### 1. Strategic Thematic Contrast\n"
            f"- **{doc_a_name}** centers around **{', '.join(topics_a[:3])}**, emphasizing {lead_a[:120]}.\n"
            f"- **{doc_b_name}** focuses on **{', '.join(topics_b[:3])}**, presenting {lead_b[:120]}.\n\n"
            f"#### 2. Methodological & Technological Alignment\n"
            f"- **Technologies in {doc_a_name}**: `{tech_a}`\n"
            f"- **Technologies in {doc_b_name}**: `{tech_b}`\n\n"
            f"#### 3. Synergy & Research Takeaways\n"
            f"Comparing both documents reveals complementary perspectives: while **{doc_a_name}** addresses core foundational mechanics, "
            f"**{doc_b_name}** provides experimental depth and application benchmarks. Cross-referencing both creates a unified intelligence baseline."
        )

        # If LLM key is present, enhance synthesis
        if self.llm_service.api_key and any(p in self.llm_service.provider for p in ["Gemini", "OpenAI", "NVIDIA"]):
            try:
                prompt = (
                    f"Perform a professional academic comparison between Document A ('{doc_a_name}') and Document B ('{doc_b_name}').\n\n"
                    f"DOCUMENT A EXCERPT:\n{doc_a_text[:3000]}\n\n"
                    f"DOCUMENT B EXCERPT:\n{doc_b_text[:3000]}\n\n"
                    f"Generate a side-by-side comparison table and executive synthesis explaining similarities, differences, and key takeaways."
                )
                system_instruction = "You are an expert document intelligence researcher. Compare two documents thoroughly with clear markdown tables and synthesis."
                llm_resp = self.llm_service.generate(prompt=prompt, system_instruction=system_instruction)
                if len(llm_resp.get("text", "")) > 100:
                    executive_synthesis = llm_resp.get("text", "")
            except Exception as e:
                logger.warning("LLM comparison synthesis failed (%s), using deterministic matrix", e)

        full_report = f"# Comparative Analysis: {doc_a_name} vs {doc_b_name}\n\n## Summary Comparison Matrix\n\n{comparison_table}\n\n{executive_synthesis}"

        return {
            "doc_a": doc_a_name,
            "doc_b": doc_b_name,
            "topics_a": topics_a,
            "topics_b": topics_b,
            "comparison_table": comparison_table,
            "executive_synthesis": executive_synthesis,
            "full_report": full_report
        }
