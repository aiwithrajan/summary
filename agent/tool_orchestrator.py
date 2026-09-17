"""Tool Orchestrator executing the 4-tool deterministic pipeline."""
import json
import os
from typing import Dict, Any, Optional

from tools.json_formatter_tool import PDFToStructuredJSONTool
from tools.noise_cleaner_tool import NoiseCleanerTool
from tools.table_analytics_tool import TableAnalyticsTool
from tools.fact_checker_tool import FactAndCitationCheckerTool
from .llm import GeminiLLMClient


class ToolDrivenOrchestrator:
    """Coordinates the 4 deterministic tools to produce faithful, verified summaries."""

    def __init__(self, llm_client: GeminiLLMClient):
        self.tool1 = PDFToStructuredJSONTool()
        self.tool2 = NoiseCleanerTool()
        self.tool3 = TableAnalyticsTool()
        self.tool4 = FactAndCitationCheckerTool()
        self.llm = llm_client

    def process_pdf(
        self,
        pdf_path: str,
        level: str = "executive",
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the 4 tools sequentially to generate the verified answer."""
        tool_traces = []

        # --- STEP 1: Tool 1 (PDF to Structured JSON) ---
        raw_json = self.tool1.execute(pdf_path)
        tool_traces.append({
            "step": 1,
            "tool": self.tool1.name,
            "status": "COMPLETED",
            "message": f"Extracted {raw_json['document_metadata']['total_pages']} pages, {raw_json['document_metadata']['total_tables']} tables, and {raw_json['document_metadata']['total_equations']} equations into JSON format."
        })

        # --- STEP 2: Tool 2 (Noise Cleaner) ---
        cleaned_json = self.tool2.execute(raw_json)
        headers_removed = len(cleaned_json.get("sanitization_meta", {}).get("boilerplate_headers_removed", []))
        tool_traces.append({
            "step": 2,
            "tool": self.tool2.name,
            "status": "COMPLETED",
            "message": f"Sanitized document text, normalized unicode ligatures, and stripped {headers_removed} boilerplate headers/footers."
        })

        # --- STEP 3: Tool 3 (Table Analytics) ---
        analyzed_json = self.tool3.execute(cleaned_json)
        tables_count = analyzed_json.get("table_analytics", {}).get("total_tables_analyzed", 0)
        tool_traces.append({
            "step": 3,
            "tool": self.tool3.name,
            "status": "COMPLETED",
            "message": f"Computed deterministic mathematical comparisons and percentage deltas across {tables_count} tables."
        })

        # --- STEP 4: Synthesis from Clean Structured JSON ---
        synthesis_markdown = self._synthesize_from_tools_json(analyzed_json, level, custom_instructions)
        tool_traces.append({
            "step": 4,
            "tool": "LanguageSynthesizer",
            "status": "COMPLETED",
            "message": "Synthesized structured answer strictly from the 3 tools' verified JSON schema."
        })

        # --- STEP 5: Tool 4 (Fact & Citation Checker) ---
        verification_result = self.tool4.execute(synthesis_markdown, analyzed_json)
        tool_traces.append({
            "step": 5,
            "tool": self.tool4.name,
            "status": "COMPLETED",
            "message": f"Verified citations ({verification_result['valid_citations']} valid) & statistics. Faithfulness Score: {verification_result['fact_check_score']}%."
        })

        return {
            "title": analyzed_json["document_metadata"]["inferred_title"],
            "structured_json": analyzed_json,
            "summary": verification_result["annotated_summary"],
            "raw_summary": synthesis_markdown,
            "tool_traces": tool_traces,
            "table_analytics": analyzed_json.get("table_analytics", {}),
            "verification": verification_result
        }

    def _synthesize_from_tools_json(
        self,
        doc_json: Dict[str, Any],
        level: str,
        custom_instructions: Optional[str]
    ) -> str:
        """Feed the clean, validated JSON into Gemini to synthesize language."""
        if not self.llm.is_available():
            return self._heuristic_fallback(doc_json)

        # Build high-density context from JSON
        meta = doc_json.get("document_metadata", {})
        hierarchy = doc_json.get("section_hierarchy", [])
        table_analytics = doc_json.get("table_analytics", {}).get("analyses", [])

        # Extract mathematical table insights
        math_insights = []
        for ta in table_analytics:
            for mi in ta.get("mathematical_insights", []):
                math_insights.append(f"- [Table p. {ta.get('page')}] {mi}")

        # Page by page clean text
        pages_text_list = []
        for p in doc_json.get("pages", []):
            p_num = p.get("page_number")
            c_text = p.get("clean_text", "")
            eqs = p.get("equations", [])
            eq_str = f"\nEquations: {', '.join(eqs)}" if eqs else ""
            pages_text_list.append(f"--- PAGE {p_num} ---\n{c_text}{eq_str}")

        clean_pages_text = "\n\n".join(pages_text_list)
        if len(clean_pages_text) > 100000:
            clean_pages_text = clean_pages_text[:100000] + "\n...[truncated]..."

        system_instruction = (
            "You are a strict technical summarizer. You do NOT read raw PDFs.\n"
            "You are provided with pre-processed, verified STRUCTURED JSON produced by:\n"
            "- Tool 1 (PDFToStructuredJSONTool)\n"
            "- Tool 2 (NoiseCleanerTool)\n"
            "- Tool 3 (TableAnalyticsTool)\n\n"
            "STRICT RULES:\n"
            "1. Rely ONLY on the facts, numbers, equations, and mathematical table insights in this JSON.\n"
            "2. Cite exact pages using (p. X) or (Section X, p. Y).\n"
            "3. Use the exact mathematical differences and numbers computed by Tool 3.\n"
            "4. Never invent details outside this structured document.\n"
            "5. If information is missing, explicitly state: 'Not clearly specified in the document.'"
        )

        prompt = f"""
Document Title: {meta.get('inferred_title')}
Filename: {meta.get('filename')}
Total Pages: {meta.get('total_pages')}

Document Sections Hierarchy (from Tool 1):
{json.dumps(hierarchy[:15], indent=2)}

Mathematical Table Analytics (Calculated deterministically by Tool 3):
{chr(10).join(math_insights) if math_insights else 'No numerical comparison tables detected.'}

Sanitized Text (from Tool 2):
{clean_pages_text}

---
Task: Generate an EXECUTIVE SUMMARY focusing on:
# Executive Summary: {meta.get('inferred_title')}

### 1. Problem Statement
Explain the operational/research challenge addressed (with page citations).

### 2. Significance & Relevance
Why this work matters and its core motivation.

### 3. Proposals & Methodology
Specific architectures, algorithms, equations, and approaches proposed.

### 4. Major Findings & Quantitative Results
Empirical outcomes and numbers. Include the exact mathematical comparisons computed by Tool 3.

### 5. Implications & Strategic Impact
Practical and system-level impact.

### 6. Limitations & Open Challenges
Explicitly stated constraints (or state none if not mentioned).
"""
        return self.llm.generate(prompt, system_instruction=system_instruction)

    def _heuristic_fallback(self, doc_json: Dict[str, Any]) -> str:
        meta = doc_json.get("document_metadata", {})
        hierarchy = [h["heading"] for h in doc_json.get("section_hierarchy", [])]
        table_analytics = doc_json.get("table_analytics", {}).get("analyses", [])

        lines = [
            f"# Executive Summary: {meta.get('inferred_title')}",
            "",
            "> [!NOTE]",
            "> Generated using the 4-Tool Pipeline in deterministic mode.",
            "",
            "### 1. Problem Statement & Overview",
            f"The document '{meta.get('filename')}' spans {meta.get('total_pages')} pages.",
            "",
            "### 2. Major Sections Detected",
        ]
        for h in hierarchy[:8]:
            lines.append(f"- {h}")

        if table_analytics:
            lines.extend(["", "### 3. Deterministic Table Analytics (Computed by Tool 3)"])
            for ta in table_analytics:
                for mi in ta.get("mathematical_insights", []):
                    lines.append(f"- {mi}")

        lines.extend([
            "",
            "### 4. Mathematical Equations (Extracted by Tool 1)",
        ])
        for p in doc_json.get("pages", []):
            for eq in p.get("equations", []):
                lines.append(f"- Page {p['page_number']}: `{eq}`")

        return "\n".join(lines)
