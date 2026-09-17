"""Tool 2: Noise Cleaner Tool (Document Sanitizer).
Strips out repetitive running headers, footers, page number watermarks,
normalizes unicode ligatures, fixes hyphenated line breaks, and produces clean_text.
Zero LLM involved — pure deterministic Python.
"""
import re
from typing import Dict, Any, List


class NoiseCleanerTool:
    """Sanitizes text by removing headers, footers, and formatting noise."""

    def __init__(self):
        self.name = "NoiseCleanerTool"
        self.description = "Cleans document text, removing repetitive headers/footers and formatting noise."

    def execute(self, structured_json: Dict[str, Any]) -> Dict[str, Any]:
        pages = structured_json.get("pages", [])
        if not pages:
            return structured_json

        # 1. Identify repetitive lines (running headers/footers appearing on >= 50% of pages)
        line_occurrences: Dict[str, int] = {}
        for p in pages:
            lines = [l.strip() for l in p.get("raw_text", "").split("\n") if l.strip()]
            # Only consider top 2 lines (header) and bottom 2 lines (footer)
            candidates = lines[:2] + lines[-2:]
            for c in set(candidates):
                if len(c) < 100:  # headers/footers are usually short
                    line_occurrences[c] = line_occurrences.get(c, 0) + 1

        total_pages = len(pages)
        boilerplate_lines = {line for line, count in line_occurrences.items() if count >= max(2, total_pages * 0.4)}

        # 2. Process and clean each page
        for p in pages:
            raw = p.get("raw_text", "")
            lines = raw.split("\n")
            cleaned_lines = []

            for l in lines:
                l_strip = l.strip()
                # Skip identified boilerplate
                if l_strip in boilerplate_lines:
                    continue
                # Skip standalone page numbers like "1", "- 2 -", "Page 3 of 10"
                if re.match(r'^(?:page\s*)?\d+(?:\s*of\s*\d+)?$', l_strip, re.IGNORECASE):
                    continue
                cleaned_lines.append(l)

            cleaned_text = "\n".join(cleaned_lines)

            # Fix hyphenated line wraps (e.g., "implemen-\ntation" -> "implementation")
            cleaned_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned_text)

            # Normalize common ligatures & quotes
            replacements = {
                "\ufb01": "fi",
                "\ufb02": "fl",
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "--",
                "\xa0": " ",
            }
            for orig, rep in replacements.items():
                cleaned_text = cleaned_text.replace(orig, rep)

            # Collapse multiple blank lines
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

            p["clean_text"] = cleaned_text
            p["noise_reduction_ratio"] = round(1.0 - (len(cleaned_text) / (len(raw) + 1e-5)), 3)

        structured_json["sanitization_meta"] = {
            "tool": self.name,
            "boilerplate_headers_removed": list(boilerplate_lines),
            "total_pages_cleaned": total_pages
        }

        return structured_json
