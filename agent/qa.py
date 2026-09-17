"""PDF Grounded Question Answering Engine with section and page citations."""
from typing import Dict, Any, List, Optional
import re

from .extractor import ExtractedDocument, PageContent
from .llm import GeminiLLMClient


class PDFQuestionAnswerer:
    """Answers user queries grounded strictly in the PDF document."""

    def __init__(self, llm_client: GeminiLLMClient):
        self.llm = llm_client

    def answer_question(self, doc: ExtractedDocument, query: str) -> str:
        """Answer a specific question referencing exact pages and sections."""
        # 1. Retrieve the most relevant pages
        relevant_pages = self._find_relevant_pages(doc, query)

        # 2. Build context
        context_parts = []
        for p in relevant_pages:
            table_info = ""
            if p.tables:
                for t in p.tables:
                    table_info += f"\n[Table on Page {t.page_num}]:\n{t.markdown}\n"

            context_parts.append(
                f"--- PAGE {p.page_number} ---\n{p.raw_text.strip()}\n{table_info}"
            )
        context_text = "\n\n".join(context_parts)

        # If LLM is available, generate grounded response
        if self.llm.is_available():
            system_prompt = (
                "You are an expert Q&A agent grounded strictly in the provided PDF document.\n"
                "Rules:\n"
                "1. Answer the question directly and concisely.\n"
                "2. Provide supporting context from the document.\n"
                "3. Cite the exact page number(s) (e.g., 'According to page 4...').\n"
                "4. Clearly distinguish information directly stated in the PDF from any necessary explanatory context.\n"
                "5. If the answer cannot be found in the document, explicitly say:\n"
                "   'The document does not provide enough information to determine this.'\n"
                "6. NEVER hallucinate or assume facts not present in the text."
            )

            prompt = f"""
Document Title: {doc.inferred_title}
Total Pages: {doc.total_pages}

Document Excerpts from Relevant Pages:
{context_text}

User Question:
{query}
"""
            return self.llm.generate(prompt, system_instruction=system_prompt)

        # Fallback keyword-based response
        return self._heuristic_answer(relevant_pages, query)

    def _find_relevant_pages(self, doc: ExtractedDocument, query: str, top_k: int = 4) -> List[PageContent]:
        query_words = set(re.findall(r'\w+', query.lower()))
        # Filter stopwords
        stopwords = {"what", "is", "are", "the", "a", "an", "in", "of", "to", "for", "and", "or", "how", "why", "does"}
        keywords = query_words - stopwords
        if not keywords:
            keywords = query_words

        scored_pages = []
        for page in doc.pages:
            text_lower = page.raw_text.lower()
            score = 0
            for kw in keywords:
                matches = len(re.findall(rf'\b{re.escape(kw)}\b', text_lower))
                score += matches * 2
            
            # Additional score if keyword in headings
            for h in page.headings:
                h_lower = h["text"].lower()
                for kw in keywords:
                    if kw in h_lower:
                        score += 5

            scored_pages.append((score, page))

        scored_pages.sort(key=lambda x: x[0], reverse=True)
        # Return top pages that have score > 0, or first 3 pages if no matches
        selected = [p for s, p in scored_pages if s > 0][:top_k]
        if not selected:
            selected = doc.pages[:top_k]
        return selected

    def _heuristic_answer(self, relevant_pages: List[PageContent], query: str) -> str:
        return (
            f"> [!NOTE]\n"
            f"> Offline mode active. Showing relevant text matches for '{query}':\n\n"
            + "\n\n".join([f"**Page {p.page_number}:**\n> {p.raw_text[:300]}..." for p in relevant_pages])
            + "\n\n*Configure `GEMINI_API_KEY` to enable deep question answering.*"
        )
