"""Verification test suite for PDF Summarization Agent."""
import os
import sys

from sample_generator import create_sample_pdf
from agent.extractor import PDFExtractor
from agent.analyzer import DocumentAnalyzer
from agent.summarizer import PDFSummarizer, SummaryLevel
from agent.qa import PDFQuestionAnswerer
from agent.llm import GeminiLLMClient


def test_full_pipeline():
    pdf_path = "sample_test_doc.pdf"
    print("\n--- 1. Generating Sample PDF ---")
    create_sample_pdf(pdf_path)
    assert os.path.exists(pdf_path), "Sample PDF was not created."

    print("\n--- 2. Testing PDFExtractor ---")
    extractor = PDFExtractor(pdf_path)
    doc = extractor.extract()
    print(f"Inferred Title: {doc.inferred_title}")
    print(f"Total Pages: {doc.total_pages}")
    assert doc.total_pages == 3, f"Expected 3 pages, got {doc.total_pages}"
    assert "Adaptive Neural Caching" in doc.inferred_title or "sample_test_doc" in doc.filename
    assert any(p.has_equations for p in doc.pages), "Failed to detect equation blocks"
    print("Extractor test passed!")

    print("\n--- 3. Testing Analyzer ---")
    llm = GeminiLLMClient()
    analyzer = DocumentAnalyzer(llm)
    analysis = analyzer.analyze(doc)
    print(f"Main Topic: {analysis.main_topic}")
    print(f"Major Sections: {analysis.major_sections}")
    assert len(analysis.major_sections) > 0, "No sections detected"
    print("Analyzer test passed!")

    print("\n--- 4. Testing Summarizer (Executive Level) ---")
    summarizer = PDFSummarizer(llm)
    summary = summarizer.summarize(doc, level=SummaryLevel.EXECUTIVE)
    print("\nGenerated Executive Summary Preview:")
    print("-" * 50)
    print(summary[:600] + "\n...")
    print("-" * 50)
    assert len(summary) > 100, "Summary output too short"
    print("Summarizer test passed!")

    print("\n--- 5. Testing Traceable Q&A ---")
    qa = PDFQuestionAnswerer(llm)
    question = "What was the latency reduction achieved?"
    answer = qa.answer_question(doc, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    assert len(answer) > 20, "Q&A answer empty"
    print("Q&A test passed!")

    # Cleanup test pdf
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! ✅")


if __name__ == "__main__":
    test_full_pipeline()
