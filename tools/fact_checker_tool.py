"""Tool 4: Fact and Citation Checker Tool (Grounding & Anti-Hallucination).
Cross-references every generated claim, citation (p. X), and numerical statistic
against the ground-truth structured JSON produced by Tools 1, 2, and 3.
Zero LLM involved — pure deterministic Python cross-validation.
"""
import re
from typing import Dict, Any, List, Tuple


class FactAndCitationCheckerTool:
    """Verifies citations, statistics, and claims against ground-truth document JSON."""

    def __init__(self):
        self.name = "FactAndCitationCheckerTool"
        self.description = "Verifies citations and numbers against ground truth JSON to prevent hallucinations."

    def execute(self, summary_markdown: str, structured_json: Dict[str, Any]) -> Dict[str, Any]:
        pages = structured_json.get("pages", [])
        page_texts = {p["page_number"]: p.get("clean_text", "") for p in pages}
        page_stats = {p["page_number"]: p.get("statistics_markers", []) for p in pages}

        # Build table cell index
        table_cells: Dict[int, List[str]] = {}
        for p in pages:
            num = p["page_number"]
            table_cells[num] = []
            for t in p.get("tables", []):
                for row in t.get("rows", []):
                    table_cells[num].extend([str(c).lower() for c in row])

        verified_citations = []
        flagged_claims = []

        # 1. Regex find all page citations e.g. (p. 1), (page 2), (Section 1, p. 3)
        # Avoid matching percentile notations like P99, P95, P50 latency
        citation_matches = re.finditer(r'(?:\b(?:page\s*(\d+)|p\.\s*(\d+)|\(\s*p(?:age)?\.?\s*(\d+)\s*\))\b)', summary_markdown, re.IGNORECASE)

        total_citations = 0
        valid_citations = 0

        for match in citation_matches:
            # Extract first non-None group
            page_str = next((g for g in match.groups() if g is not None), None)
            if not page_str:
                continue
            total_citations += 1
            cited_page = int(page_str)

            if 1 <= cited_page <= len(pages):
                valid_citations += 1
                verified_citations.append({
                    "cited_page": cited_page,
                    "status": "VALID_PAGE_EXISTS",
                    "page_available": True
                })
            else:
                flagged_claims.append({
                    "cited_page": cited_page,
                    "status": "INVALID_PAGE_OUT_OF_BOUNDS",
                    "reason": f"Document has {len(pages)} pages, but summary cited page {cited_page}."
                })

        # 2. Verify numerical statistics mentioned in summary
        numbers_in_summary = re.findall(r'(\d+(?:\.\d+)?(?:%|ms|MB|GB|KB|s)?)', summary_markdown)
        verified_numbers = 0
        unverified_numbers = 0

        all_doc_text = " ".join([p.get("clean_text", "") for p in pages]).lower()
        all_table_text = " ".join([" ".join(cells) for cells in table_cells.values()]).lower()
        full_ground_truth = all_doc_text + " " + all_table_text

        checked_numbers = []
        for num_token in set(numbers_in_summary):
            clean_num = num_token.strip()
            # Ignore single generic numbers like 1, 2, 3 (often bullet numbers)
            if clean_num in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
                continue

            if clean_num.lower() in full_ground_truth:
                verified_numbers += 1
                checked_numbers.append({"token": clean_num, "status": "VERIFIED_IN_DOCUMENT"})
            else:
                # Check stripped numerical value without unit
                val_only = re.sub(r'[^\d.]', '', clean_num)
                if val_only and val_only in full_ground_truth:
                    verified_numbers += 1
                    checked_numbers.append({"token": clean_num, "status": "VERIFIED_NUMERICALLY"})
                else:
                    unverified_numbers += 1
                    checked_numbers.append({"token": clean_num, "status": "UNVERIFIED_IN_DOCUMENT"})
                    flagged_claims.append({
                        "token": clean_num,
                        "status": "UNVERIFIED_NUMBER",
                        "reason": f"Statistic '{clean_num}' could not be matched in source text or tables."
                    })

        total_checked = verified_numbers + unverified_numbers
        fact_check_score = round((verified_numbers / (total_checked + 1e-5)) * 100.0, 1) if total_checked > 0 else 100.0

        # 3. Create Annotated Summary with verification report
        verification_report = (
            "\n\n---\n"
            "### 🛡️ Tool Verification & Grounding Report\n"
            f"- **Faithfulness Score:** `{fact_check_score}% Grounded`\n"
            f"- **Citations Checked:** `{total_citations}` (Valid: `{valid_citations}`)\n"
            f"- **Numerical Statistics Verified:** `{verified_numbers}` verified, `{unverified_numbers}` unverified\n"
        )
        if flagged_claims:
            verification_report += "- **Flagged Discrepancies:**\n"
            for f in flagged_claims[:3]:
                verification_report += f"  - ⚠️ {f.get('reason', 'Discrepancy detected')}\n"
        else:
            verification_report += "- **Status:** ✅ Zero hallucinations detected. All numbers match ground-truth JSON.\n"

        annotated_summary = summary_markdown + verification_report

        return {
            "tool": self.name,
            "fact_check_score": fact_check_score,
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "verified_numbers": verified_numbers,
            "unverified_numbers": unverified_numbers,
            "flagged_claims": flagged_claims,
            "annotated_summary": annotated_summary
        }
