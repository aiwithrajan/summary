"""Generates a sample technical PDF for testing the PDF Summarization Agent."""
import fitz  # PyMuPDF


def create_sample_pdf(output_path: str = "sample_research_paper.pdf"):
    doc = fitz.open()

    # --- PAGE 1: Title, Abstract, Introduction ---
    page1 = doc.new_page(width=595, height=842)  # A4 standard
    
    # Title
    page1.insert_text((50, 70), "Adaptive Neural Caching for Low-Latency Query Execution", fontsize=16, fontname="helv", fontfile=None)
    page1.insert_text((50, 95), "Author: Dr. Alex Vance, Systems Research Lab, MIT", fontsize=10, fontname="helv")
    page1.insert_text((50, 110), "Date: September 2026", fontsize=9, fontname="helv")

    # Abstract
    page1.insert_text((50, 140), "Abstract", fontsize=13, fontname="helv")
    abstract_text = (
        "Modern cloud databases suffer from high tail latencies due to inefficient query result caching.\n"
        "In this paper, we propose NeuroCache, an adaptive neural caching framework that predicts query\n"
        "re-use using lightweight embedding representations. Unlike traditional LRU and LFU heuristics,\n"
        "NeuroCache dynamically optimizes eviction policies using reinforcement learning. Our evaluation\n"
        "demonstrates an 18.4% improvement in cache hit ratio and a 34.2% reduction in P99 query latency\n"
        "on the TPC-H 100GB benchmark suite."
    )
    page1.insert_text((50, 160), abstract_text, fontsize=10, fontname="helv")

    # Section 1: Introduction
    page1.insert_text((50, 260), "1. Introduction", fontsize=13, fontname="helv")
    intro_text = (
        "Data-intensive enterprise applications demand sub-millisecond response times. Traditional cache eviction\n"
        "algorithms like Least Recently Used (LRU) fail to account for query computational cost and temporal query skew.\n"
        "When an expensive aggregation query is evicted prematurely, the database must recompute results from scratch,\n"
        "incurring severe CPU and I/O bottlenecks.\n\n"
        "To solve this problem, we present a novel caching mechanism based on neural predictive scoring.\n"
        "Our primary contributions are:\n"
        "• A compact neural feature representation for SQL query ASTs.\n"
        "• An online actor-critic policy that balances hit rate against computation cost.\n"
        "• Extensive empirical evaluation across synthetic and real-world database traces."
    )
    page1.insert_text((50, 280), intro_text, fontsize=10, fontname="helv")

    # --- PAGE 2: Architecture, Mathematical Formulation, Table ---
    page2 = doc.new_page(width=595, height=842)

    page2.insert_text((50, 60), "2. NeuroCache System Architecture", fontsize=13, fontname="helv")
    arch_text = (
        "The NeuroCache pipeline consists of three core components: the Query Parser, the Embedder,\n"
        "and the Neural Eviction Engine. When a SQL query is received, its Abstract Syntax Tree (AST)\n"
        "is converted into a 64-dimensional dense vector representation."
    )
    page2.insert_text((50, 80), arch_text, fontsize=10, fontname="helv")

    page2.insert_text((50, 140), "2.1 Mathematical Formulation", fontsize=12, fontname="helv")
    math_text = (
        "The utility score U(q) of query q is calculated using the trade-off formula:\n\n"
        "$$ U(q) = \\alpha \\cdot C(q) + (1 - \\alpha) \\cdot P_{hit}(q) - \\lambda \\cdot S(q) $$\n\n"
        "Where C(q) is the computational cost, P_hit(q) is predicted recurrence probability, S(q) is byte size,\n"
        "and \\alpha is the weighting factor bounded by [0, 1]."
    )
    page2.insert_text((50, 160), math_text, fontsize=10, fontname="helv")

    page2.insert_text((50, 260), "3. Experimental Evaluation & Results", fontsize=13, fontname="helv")
    page2.insert_text((50, 280), "Table 1: Performance comparison across 100,000 queries on TPC-H 100GB", fontsize=10, fontname="helv")

    # Draw Table
    # Headers
    y = 310
    page2.draw_rect(fitz.Rect(50, y, 545, y + 22), color=(0, 0, 0), fill=(0.85, 0.85, 0.85))
    page2.insert_text((55, y + 15), "Method", fontsize=10, fontname="helv")
    page2.insert_text((160, y + 15), "Hit Rate (%)", fontsize=10, fontname="helv")
    page2.insert_text((280, y + 15), "Avg Latency (ms)", fontsize=10, fontname="helv")
    page2.insert_text((420, y + 15), "P99 Latency (ms)", fontsize=10, fontname="helv")

    rows = [
        ("LRU Baseline", "54.2%", "14.8 ms", "128.5 ms"),
        ("LFU Baseline", "58.1%", "12.3 ms", "112.0 ms"),
        ("ARC (Adaptive)", "63.7%", "9.7 ms", "89.4 ms"),
        ("NeuroCache (Ours)", "75.4%", "5.1 ms", "42.6 ms")
    ]
    
    for r in rows:
        y += 24
        page2.draw_rect(fitz.Rect(50, y, 545, y + 24), color=(0.8, 0.8, 0.8))
        page2.insert_text((55, y + 16), r[0], fontsize=10, fontname="helv")
        page2.insert_text((160, y + 16), r[1], fontsize=10, fontname="helv")
        page2.insert_text((280, y + 16), r[2], fontsize=10, fontname="helv")
        page2.insert_text((420, y + 16), r[3], fontsize=10, fontname="helv")

    # --- PAGE 3: Limitations, Future Work, Conclusion ---
    page3 = doc.new_page(width=595, height=842)

    page3.insert_text((50, 60), "4. Limitations and Known Constraints", fontsize=13, fontname="helv")
    lim_text = (
        "While NeuroCache delivers notable gains, several explicit limitations exist:\n"
        "1. Cold-Start Overhead: During the first 500 queries, model predictions are unstable\n"
        "   and fall back to standard LRU.\n"
        "2. Memory Overhead: The neural model consumes an additional 128 MB of host RAM for storing embeddings.\n"
        "3. Distributed Multi-Region Consistency: The current prototype is constrained to single-node deployments\n"
        "   and does not handle cross-datacenter cache invalidations."
    )
    page3.insert_text((50, 85), lim_text, fontsize=10, fontname="helv")

    page3.insert_text((50, 200), "5. Conclusion", fontsize=13, fontname="helv")
    concl_text = (
        "NeuroCache demonstrates that neural reinforcement policies can outperform conventional rule-based\n"
        "cache eviction algorithms in enterprise databases. Future work will investigate distributed consensus\n"
        "protocols and hardware-accelerated inference."
    )
    page3.insert_text((50, 220), concl_text, fontsize=10, fontname="helv")

    doc.save(output_path)
    doc.close()
    print(f"Sample PDF created successfully at: {output_path}")


if __name__ == "__main__":
    create_sample_pdf()
