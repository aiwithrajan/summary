"""Tool 1: PDF to Structured JSON Formatter Tool.
Extracts document metadata, hierarchical sections, page-by-page clean blocks,
structured tables (with columns/rows), equations, and statistical markers.
Zero LLM involved — pure deterministic Python.
"""
import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


class PDFToStructuredJSONTool:
    """Deterministic tool that parses raw PDF into a strict, high-density JSON schema."""

    def __init__(self):
        self.name = "PDFToStructuredJSONTool"
        self.description = "Extracts structure, hierarchy, tables, and equations into standard JSON format."

    def execute(self, pdf_path: str) -> Dict[str, Any]:
        """Execute the structural extraction on a PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        pages_data = []
        toc_entries = []
        metadata = {}
        inferred_title = ""

        # 1. PyMuPDF analysis (fonts, headings, equations)
        if fitz:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            toc_entries = [{"level": item[0], "title": item[1], "page": item[2]} for item in doc.get_toc()]

            for idx in range(len(doc)):
                page = doc[idx]
                page_num = idx + 1
                text = page.get_text()

                # Heading detection via block font sizes
                blocks = page.get_text("dict").get("blocks", [])
                font_sizes = []
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            for s in l["spans"]:
                                font_sizes.append(s["size"])

                median_font = 10.0
                if font_sizes:
                    font_sizes.sort()
                    median_font = font_sizes[len(font_sizes) // 2]

                headings = []
                for b in blocks:
                    if "lines" in b:
                        block_text = ""
                        max_sz = 0.0
                        is_bold = False
                        for l in b["lines"]:
                            for s in l["spans"]:
                                block_text += s["text"] + " "
                                if s["size"] > max_sz:
                                    max_sz = s["size"]
                                if "bold" in s["font"].lower() or "black" in s["font"].lower():
                                    is_bold = True
                        clean = block_text.strip()
                        if clean and max_sz >= median_font * 1.25 and len(clean) < 130:
                            headings.append({
                                "text": clean,
                                "font_size": round(max_sz, 1),
                                "is_bold": is_bold,
                                "page": page_num
                            })

                # Equations regex extraction
                equations = []
                eq_patterns = [
                    r'\$\$.*?\$\$',
                    r'\\begin\{equation\}.*?\\end\{equation\}',
                    r'[A-Za-z]\([a-z]\)\s*=\s*[^;\n]+',
                    r'\\alpha|\\beta|\\gamma|\\lambda|\\theta|\\sigma|\\sum|\\int'
                ]
                for pat in eq_patterns:
                    matches = re.findall(pat, text, re.DOTALL)
                    for m in matches:
                        m_clean = m.strip()
                        if m_clean and m_clean not in equations and len(m_clean) < 200:
                            equations.append(m_clean)

                # Numbers & percentage statistics detection
                stats = re.findall(r'\b\d+(?:\.\d+)?(?:%|ms|MB|GB|KB|s|x)?\b', text)
                filtered_stats = [s for s in stats if any(c.isalpha() or c == '%' for c in s)]

                pages_data.append({
                    "page_number": page_num,
                    "headings": headings,
                    "equations": equations,
                    "statistics_markers": list(set(filtered_stats)),
                    "tables": [],
                    "raw_text": text,
                    "char_count": len(text),
                    "images_count": len(page.get_images())
                })

            doc.close()

        # 2. Table extraction via pdfplumber
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as plumber_doc:
                    for i, page in enumerate(plumber_doc.pages):
                        if i < len(pages_data):
                            raw_tables = page.extract_tables()
                            for t_idx, t in enumerate(raw_tables):
                                if t and len(t) > 1:
                                    headers = [str(c).strip() if c is not None else f"Col_{j}" for j, c in enumerate(t[0])]
                                    rows = []
                                    for r in t[1:]:
                                        cleaned_row = [str(c).strip() if c is not None else "" for c in r]
                                        if any(cleaned_row):
                                            rows.append(cleaned_row)
                                    if rows:
                                        pages_data[i]["tables"].append({
                                            "table_id": f"table_p{i+1}_{t_idx+1}",
                                            "page": i + 1,
                                            "headers": headers,
                                            "rows": rows,
                                            "row_count": len(rows),
                                            "col_count": len(headers)
                                        })
            except Exception:
                pass

        # Inferred Title
        inferred_title = metadata.get("title") or ""
        if not inferred_title and pages_data and pages_data[0]["headings"]:
            inferred_title = pages_data[0]["headings"][0]["text"]
        elif not inferred_title and pages_data:
            first_lines = [l.strip() for l in pages_data[0]["raw_text"].split("\n") if l.strip()]
            inferred_title = first_lines[0] if first_lines else os.path.basename(pdf_path)

        all_headings = []
        for p in pages_data:
            for h in p["headings"]:
                all_headings.append({
                    "heading": h["text"],
                    "page": p["page_number"],
                    "level": 1 if h["font_size"] >= 14.0 else 2
                })

        total_tables = sum(len(p["tables"]) for p in pages_data)
        total_equations = sum(len(p["equations"]) for p in pages_data)

        structured_json = {
            "tool": self.name,
            "document_metadata": {
                "filename": os.path.basename(pdf_path),
                "inferred_title": inferred_title,
                "total_pages": len(pages_data),
                "total_tables": total_tables,
                "total_equations": total_equations,
                "pdf_metadata": metadata
            },
            "table_of_contents": toc_entries,
            "section_hierarchy": all_headings,
            "pages": pages_data
        }

        return structured_json
