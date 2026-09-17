"""Document Analyzer executing the 10-Step Analysis Workflow."""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .extractor import ExtractedDocument
from .llm import GeminiLLMClient


@dataclass
class AnalysisResult:
    # Step 1: Document Understanding
    title: str
    doc_type: str
    main_topic: str
    purpose: str
    intended_audience: str
    major_sections: List[str]
    complexity: str

    # Step 2 & 3: Extracted Information & Hierarchy
    central_message: str
    key_points: List[str]
    section_analyses: List[Dict[str, Any]]

    # Step 5 - 8: Technical, Tables, Figures, Equations
    technical_details: Dict[str, Any]
    tables_summary: List[Dict[str, Any]]
    equations_summary: List[Dict[str, Any]]

    # Step 9 & 10: Claims, Results, Limitations & Uncertainty
    key_findings: List[str]
    limitations: List[str]
    uncertainties: List[str]
    raw_llm_analysis: Optional[str] = None


class DocumentAnalyzer:
    """Executes the 10-Step Analysis Workflow on an ExtractedDocument."""

    def __init__(self, llm_client: GeminiLLMClient):
        self.llm = llm_client

    def analyze(self, doc: ExtractedDocument) -> AnalysisResult:
        """Run the 10-step analysis using Gemini or local heuristics."""
        if self.llm.is_available():
            return self._llm_analyze(doc)
        return self._heuristic_analyze(doc)

    def _llm_analyze(self, doc: ExtractedDocument) -> AnalysisResult:
        # Prepare context with page markers
        text_preview = doc.get_text_with_page_markers()
        # Cap text if extremely large (e.g. > 150k characters for a single prompt)
        if len(text_preview) > 120000:
            text_preview = text_preview[:120000] + "\n\n... [Content truncated for analysis pass] ..."

        tables_context = ""
        for p in doc.pages:
            for t in p.tables:
                tables_context += f"\nTable on Page {t.page_num}:\n{t.markdown}\n"

        system_prompt = (
            "You are an expert PDF document analysis engine. You strictly follow a 10-step analytical workflow:\n"
            "1. Understand the document (Title, Type, Main Topic, Purpose, Audience, Complexity, Sections)\n"
            "2. Extract core information (Ideas, findings, concepts, processes, methods, limitations)\n"
            "3. Determine hierarchy (Doc -> Sections -> Subsections -> Concepts -> Evidence -> Conclusions)\n"
            "4. Identify central message: What is this fundamentally communicating?\n"
            "5. Preserve technical details (architectures, algorithms, metrics, datasets, setups)\n"
            "6. Interpret tables accurately without hallucinating numbers\n"
            "7. Note figures and conceptual flows\n"
            "8. Preserve equations and variable explanations\n"
            "9. Detect claims: distinguish facts/findings from author interpretations and assumptions\n"
            "10. Detect uncertainty: explicitly mark missing or ambiguous points.\n\n"
            "CRITICAL: Never invent information. The PDF is the sole source of truth."
        )

        user_prompt = f"""Analyze the following document extracted from PDF: '{doc.filename}'.
Tables extracted:
{tables_context if tables_context else 'No explicit tables detected.'}

Document Text with Page Numbers:
{text_preview}

Output your analysis as valid JSON matching this exact structure:
{{
  "title": "Title of the document",
  "doc_type": "Research Paper / Technical Whitepaper / Manual / Financial Report / etc.",
  "main_topic": "Concise main topic",
  "purpose": "Primary purpose of the document",
  "intended_audience": "Target audience",
  "major_sections": ["Section 1", "Section 2", "Section 3"],
  "complexity": "Introductory / Intermediate / Advanced / Highly Technical",
  "central_message": "Single cohesive statement of what this document fundamentally communicates",
  "key_points": ["Point 1 (p. X)", "Point 2 (p. Y)"],
  "section_analyses": [
    {{
      "section_name": "Name of section",
      "pages": "e.g. p. 1-3",
      "summary": "Core discussion, concepts, and relationships in this section"
    }}
  ],
  "technical_details": {{
    "architectures_models": ["Model or architecture details"],
    "algorithms_methods": ["Methods, algorithms, or protocols"],
    "datasets_metrics": ["Metrics, benchmarks, datasets mentioned"],
    "equations": ["Equation formulas and variable meanings"]
  }},
  "tables_summary": [
    {{
      "table_description": "What the table compares or shows",
      "key_insights": "Key numerical comparisons or results"
    }}
  ],
  "key_findings": ["Finding 1 with exact numbers if present (p. X)", "Finding 2 (p. Y)"],
  "limitations": ["Limitation 1 explicitly stated by authors (p. X)"],
  "uncertainties": ["Any information not clearly specified or ambiguous in the document"]
}}
Return ONLY the JSON object, wrapped in ```json ... ```.
"""
        response_text = self.llm.generate(user_prompt, system_instruction=system_prompt)
        
        # Parse JSON from response
        try:
            cleaned = response_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            data = json.loads(cleaned)
            return AnalysisResult(
                title=data.get("title", doc.inferred_title),
                doc_type=data.get("doc_type", "Technical Document"),
                main_topic=data.get("main_topic", "Analysis of PDF Document"),
                purpose=data.get("purpose", "Informational"),
                intended_audience=data.get("intended_audience", "General / Technical"),
                major_sections=data.get("major_sections", [h["text"] for p in doc.pages for h in p.headings][:8]),
                complexity=data.get("complexity", "Intermediate"),
                central_message=data.get("central_message", ""),
                key_points=data.get("key_points", []),
                section_analyses=data.get("section_analyses", []),
                technical_details=data.get("technical_details", {}),
                tables_summary=data.get("tables_summary", []),
                equations_summary=data.get("equations", []),
                key_findings=data.get("key_findings", []),
                limitations=data.get("limitations", ["No explicit limitations were identified in the document."]),
                uncertainties=data.get("uncertainties", []),
                raw_llm_analysis=response_text
            )
        except Exception:
            # Fallback to heuristic parser if JSON parsing failed
            res = self._heuristic_analyze(doc)
            res.raw_llm_analysis = response_text
            return res

    def _heuristic_analyze(self, doc: ExtractedDocument) -> AnalysisResult:
        """Local structural heuristics when LLM is offline or unconfigured."""
        all_headings = []
        for p in doc.pages:
            for h in p.headings:
                all_headings.append(f"{h['text']} (p. {h['page']})")

        # Collect key sentences based on headings & patterns
        key_points = []
        findings = []
        limitations = []

        for p in doc.pages:
            lines = p.raw_text.split("\n")
            for line in lines:
                l = line.strip()
                if not l or len(l) < 25:
                    continue
                lower = l.lower()
                if any(w in lower for w in ["we propose", "we present", "in this paper", "this document describes", "this report presents"]):
                    key_points.append(f"{l} (p. {p.page_number})")
                elif any(w in lower for w in ["achieved", "results show", "outperforms", "significant increase", "demonstrates"]):
                    findings.append(f"{l} (p. {p.page_number})")
                elif any(w in lower for w in ["limitation", "drawback", "future work", "fails when", "constrained by"]):
                    limitations.append(f"{l} (p. {p.page_number})")

        if not limitations:
            limitations = ["No explicit limitations were identified in the document."]

        sections_list = [h.split(" (p.")[0] for h in all_headings[:8]] if all_headings else ["Main Body"]

        return AnalysisResult(
            title=doc.inferred_title or doc.filename,
            doc_type="Extracted Document",
            main_topic=f"Subject matter of {doc.inferred_title or doc.filename}",
            purpose="Extracted from document contents and section structure.",
            intended_audience="Technical / Domain Practitioners",
            major_sections=sections_list,
            complexity="Technical",
            central_message=key_points[0] if key_points else f"Document detailing {doc.inferred_title}",
            key_points=key_points[:8] if key_points else [f"Extracted {doc.total_pages} pages from {doc.filename}"],
            section_analyses=[
                {"section_name": s, "pages": f"p. 1-{doc.total_pages}", "summary": "Section extracted from document."}
                for s in sections_list[:5]
            ],
            technical_details={"notes": "Extracted via structural parser"},
            tables_summary=[{"table_description": f"Table on page {t.page_num}", "key_insights": t.markdown} for p in doc.pages for t in p.tables],
            equations_summary=[{"equation": eq} for p in doc.pages for eq in p.equations],
            key_findings=findings[:6] if findings else ["Refer to document text for full quantitative metrics."],
            limitations=limitations[:4],
            uncertainties=["Analysis generated via local structural heuristics; configure Gemini API for deep semantic reasoning."]
        )
