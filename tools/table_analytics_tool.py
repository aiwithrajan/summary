"""Tool 3: Table Analytics Tool (Deterministic Numerical Analysis).
Computes exact mathematical differences, baselines vs proposed improvements,
min/max values, and percentage gains directly from table data.
Zero LLM involved — pure deterministic Python calculation.
"""
import re
from typing import Dict, Any, List, Optional


class TableAnalyticsTool:
    """Performs deterministic mathematical analysis on extracted tables."""

    def __init__(self):
        self.name = "TableAnalyticsTool"
        self.description = "Computes mathematical deltas, percentage improvements, and stats on table data."

    def execute(self, structured_json: Dict[str, Any]) -> Dict[str, Any]:
        tables_analyzed = []

        pages = structured_json.get("pages", [])
        for p in pages:
            for t in p.get("tables", []):
                analysis = self._analyze_single_table(t, p.get("page_number", 1))
                if analysis:
                    tables_analyzed.append(analysis)

        structured_json["table_analytics"] = {
            "tool": self.name,
            "total_tables_analyzed": len(tables_analyzed),
            "analyses": tables_analyzed
        }

        return structured_json

    def _analyze_single_table(self, table: Dict[str, Any], page_num: int) -> Optional[Dict[str, Any]]:
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        if not rows or not headers:
            return None

        # Detect numerical columns
        col_values: Dict[str, List[Dict[str, Any]]] = {h: [] for h in headers}

        for r_idx, row in enumerate(rows):
            label = row[0] if row else f"Row_{r_idx}"
            for c_idx, cell in enumerate(row):
                if c_idx < len(headers):
                    h = headers[c_idx]
                    # Parse numeric value and unit
                    num_match = re.search(r'([+-]?\d+(?:\.\d+)?)', str(cell))
                    unit_match = re.search(r'(%|ms|s|MB|GB|KB|x|fps)', str(cell))
                    if num_match:
                        try:
                            val = float(num_match.group(1))
                            unit = unit_match.group(1) if unit_match else ""
                            col_values[h].append({
                                "row_label": label,
                                "raw": cell,
                                "val": val,
                                "unit": unit
                            })
                        except ValueError:
                            pass

        # Calculate metrics for columns that have numerical entries in >= 50% of rows
        computed_columns = {}
        mathematical_insights = []

        # Find baseline and proposed rows if applicable
        baseline_row = rows[0][0] if rows else ""
        proposed_row = rows[-1][0] if rows else ""
        for r in rows:
            r_str = str(r[0]).lower()
            if any(k in r_str for k in ["ours", "proposed", "neuro", "our method"]):
                proposed_row = r[0]
            elif any(k in r_str for k in ["baseline", "default", "standard", "lru"]):
                baseline_row = r[0]

        for h, vals in col_values.items():
            if len(vals) >= max(2, len(rows) * 0.5):
                numeric_list = [v["val"] for v in vals]
                min_item = min(vals, key=lambda x: x["val"])
                max_item = max(vals, key=lambda x: x["val"])
                avg_val = sum(numeric_list) / len(numeric_list)

                # Check improvement between baseline and proposed
                base_item = next((v for v in vals if v["row_label"] == baseline_row), None)
                prop_item = next((v for v in vals if v["row_label"] == proposed_row), None)

                delta_info = None
                if base_item and prop_item and base_item["val"] != 0:
                    pct_diff = ((prop_item["val"] - base_item["val"]) / abs(base_item["val"])) * 100.0
                    delta_info = {
                        "baseline": f"{base_item['row_label']} ({base_item['raw']})",
                        "proposed": f"{prop_item['row_label']} ({prop_item['raw']})",
                        "percentage_change": round(pct_diff, 2)
                    }

                    direction = "higher" if pct_diff > 0 else "lower"
                    mathematical_insights.append(
                        f"In column '{h}', {prop_item['row_label']} ({prop_item['raw']}) is "
                        f"{abs(round(pct_diff, 1))}% {direction} than {base_item['row_label']} ({base_item['raw']})."
                    )

                computed_columns[h] = {
                    "min": f"{min_item['row_label']} ({min_item['raw']})",
                    "max": f"{max_item['row_label']} ({max_item['raw']})",
                    "average": round(avg_val, 2),
                    "comparison": delta_info
                }

        return {
            "table_id": table.get("table_id", "unknown"),
            "page": page_num,
            "row_count": len(rows),
            "column_metrics": computed_columns,
            "mathematical_insights": mathematical_insights
        }
