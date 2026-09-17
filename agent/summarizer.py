"""Multi-level PDF Summarizer adhering to strict faithfulness and structure."""
from enum import Enum
from typing import Optional, Dict, Any, List

from .extractor import ExtractedDocument
from .analyzer import AnalysisResult, DocumentAnalyzer
from .llm import GeminiLLMClient


class SummaryLevel(str, Enum):
    DEFAULT = "default"
    EXECUTIVE = "executive"
    DETAILED = "detailed"
    MEDIUM = "medium"
    SHORT = "short"
    CUSTOM = "custom"


class PDFSummarizer:
    """Generates structured, traceable, and faithful summaries across multiple formats."""

    def __init__(self, llm_client: GeminiLLMClient):
        self.llm = llm_client
        self.analyzer = DocumentAnalyzer(llm_client)

    def summarize(
        self,
        doc: ExtractedDocument,
        level: SummaryLevel = SummaryLevel.EXECUTIVE,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Analyze the document and produce the requested summary."""
        # 1. Execute the 10-Step Analysis
        analysis = self.analyzer.analyze(doc)

        # 2. If Gemini is available, synthesize according to requested level
        if self.llm.is_available():
            return self._llm_synthesize(doc, analysis, level, custom_instructions)

        # 3. Fallback heuristic summary
        return self._heuristic_synthesize(doc, analysis, level)

    def _llm_synthesize(
        self,
        doc: ExtractedDocument,
        analysis: AnalysisResult,
        level: SummaryLevel,
        custom_instructions: Optional[str] = None
    ) -> str:
        system_prompt = (
            "You are a rigorous, intelligent PDF Summarization Agent.\n"
            "Priorities: 1. Accuracy 2. Faithful representation 3. Clear structure 4. Technical detail preservation 5. Zero hallucination.\n"
            "Rules:\n"
            "- Never invent information that is not present in the PDF.\n"
            "- Always cite pages where appropriate, e.g., (p. 3) or (Section 2, p. 5).\n"
            "- Preserve exact numbers, formulas, and metric values.\n"
            "- Distinguish facts from author interpretations.\n"
            "- If information is missing or ambiguous, state: 'Not clearly specified in the document.'\n"
            "- Format beautifully in GitHub-flavored Markdown."
        )

        doc_context = (
            f"Filename: {doc.filename}\n"
            f"Inferred Title: {analysis.title}\n"
            f"Type: {analysis.doc_type}\n"
            f"Main Topic: {analysis.main_topic}\n"
            f"Purpose: {analysis.purpose}\n"
            f"Central Message: {analysis.central_message}\n"
            f"Pages: {doc.total_pages}\n"
            f"Major Sections: {', '.join(analysis.major_sections)}\n"
        )

        level_prompts = {
            SummaryLevel.EXECUTIVE: """
Create an EXECUTIVE SUMMARY focusing strictly on these 6 core pillars:
# Executive Summary: {title}

### 1. Problem Statement
Describe the core problem, operational challenge, or research gap this document addresses (with page citations).

### 2. Significance & Relevance
Why does this work matter? What is the background context and motivation?

### 3. Proposals & Methodology
What specific solutions, architectures, algorithms, or approaches does the document propose?

### 4. Major Findings & Quantitative Results
What are the primary outcomes and experimental/empirical results? (Preserve exact statistics and numbers).

### 5. Implications & Strategic Impact
What are the practical, technical, or business implications of these findings?

### 6. Limitations & Open Challenges
List all explicit limitations, constraints, or boundary conditions mentioned by the authors. (If none stated, note: "No explicit limitations were identified in the document.")
""",
            SummaryLevel.SHORT: """
Create a SHORT SUMMARY:
# Summary: {title}

### Main Purpose
(A concise explanation of what the document accomplishes)

### Key Points
(5–10 high-impact bullet points with page references)

### Main Conclusion
(The single takeaway or concluding determination)
""",
            SummaryLevel.MEDIUM: """
Create a MEDIUM SUMMARY:
# Summary: {title}

### Overview
(Clear introduction to the document, audience, and scope)

### Major Sections & Architecture
(Overview of the primary sections and their relationships)

### Important Concepts
(Core definitions, mechanisms, and theories)

### Key Findings & Results
(Primary outcomes with exact numbers)

### Conclusion
(Final synthesized wrap-up)
""",
            SummaryLevel.DETAILED: """
Create a DETAILED SECTION-BY-SECTION SUMMARY:
# Comprehensive Technical Summary: {title}

### 1. Document Overview
Title, Type, Main Topic, Purpose, Intended Audience, Complexity.

### 2. Section-by-Section Analysis
For each major section, explain:
- What the section discusses
- Important concepts & technical mechanisms
- Evidence, findings, and relationships between sections
(Include page references for each section).

### 3. Important Technical Details
Algorithms, frameworks, mathematical formulas, datasets, evaluation metrics, and system components.

### 4. Key Findings & Empirical Results
Quantitative results, benchmarks, and comparison tables.

### 5. Explicit Limitations & Boundaries
Directly acknowledged limitations and assumptions.

### 6. Conclusions & Critical Takeaways
Detailed conclusions and future directions.
""",
            SummaryLevel.DEFAULT: """
Create a standard structured summary matching this exact format:
# Summary: {title}

## 1. Document Overview
- **Title:** {title}
- **Type:** {doc_type}
- **Main Topic:** {main_topic}
- **Purpose:** {purpose}

## 2. One-Paragraph Summary
Concise explanation of the entire document.

## 3. Key Points
5-10 prioritized bullet points with page citations.

## 4. Section-by-Section Summary
Breakdown of major sections with concepts and relationships.

## 5. Important Technical Details
Technologies, algorithms, architectures, formulas, and metrics.

## 6. Key Findings / Results
Major outcomes with preserved numerical values.

## 7. Limitations / Problems
Explicit limitations stated in the document.

## 8. Final Takeaway
Central message in direct, accessible language.
"""
        }

        if level == SummaryLevel.CUSTOM and custom_instructions:
            prompt_template = f"""
Produce a custom analysis of '{doc.filename}' according to these specific user instructions:
{custom_instructions}

Ensure all facts are sourced from the document and cite pages where applicable.
"""
        else:
            template = level_prompts.get(level, level_prompts[SummaryLevel.DEFAULT])
            prompt_template = template.format(
                title=analysis.title,
                doc_type=analysis.doc_type,
                main_topic=analysis.main_topic,
                purpose=analysis.purpose,
            )

        full_prompt = f"""
Document Metadata & Analysis:
{doc_context}

Key Points Identified:
{chr(10).join(['- ' + kp for kp in analysis.key_points])}

Major Findings:
{chr(10).join(['- ' + kf for kf in analysis.key_findings])}

Limitations:
{chr(10).join(['- ' + l for l in analysis.limitations])}

Document Content (Page-delimited):
{doc.get_text_with_page_markers()[:100000]}

---
{prompt_template}
"""
        return self.llm.generate(full_prompt, system_instruction=system_prompt)

    def _heuristic_synthesize(
        self,
        doc: ExtractedDocument,
        analysis: AnalysisResult,
        level: SummaryLevel
    ) -> str:
        """Heuristic generator for offline mode."""
        lines = [
            f"# Summary: {analysis.title}",
            "",
            "> [!NOTE]",
            "> Generated in local offline mode. To enable full generative AI synthesis with Google Gemini, set `GEMINI_API_KEY`.",
            "",
            "## Document Overview",
            f"- **Filename:** {doc.filename}",
            f"- **Total Pages:** {doc.total_pages}",
            f"- **Inferred Title:** {analysis.title}",
            f"- **Document Type:** {analysis.doc_type}",
            f"- **Main Topic:** {analysis.main_topic}",
            "",
            "## Central Message",
            analysis.central_message or "Extracted from technical document contents.",
            "",
            "## Key Extracted Points",
        ]
        for pt in analysis.key_points:
            lines.append(f"- {pt}")

        lines.extend([
            "",
            "## Major Sections Detected",
        ])
        for sec in analysis.major_sections:
            lines.append(f"- {sec}")

        lines.extend([
            "",
            "## Key Findings / Results",
        ])
        for f in analysis.key_findings:
            lines.append(f"- {f}")

        lines.extend([
            "",
            "## Explicit Limitations",
        ])
        for lim in analysis.limitations:
            lines.append(f"- {lim}")

        if doc.pages and any(p.tables for p in doc.pages):
            lines.extend(["", "## Detected Tables"])
            for p in doc.pages:
                for t in p.tables:
                    lines.append(f"\n#### Table on Page {t.page_num}\n{t.markdown}")

        return "\n".join(lines)
