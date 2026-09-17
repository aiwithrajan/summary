#!/usr/bin/env python3
"""CLI interface for the PDF Summarization Agent."""
import argparse
import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from agent.extractor import PDFExtractor
from agent.analyzer import DocumentAnalyzer
from agent.summarizer import PDFSummarizer, SummaryLevel
from agent.qa import PDFQuestionAnswerer
from agent.llm import GeminiLLMClient


console = Console()


def handle_inspect(args):
    console.print(f"[bold blue]Inspecting PDF:[/] {args.pdf_path}")
    extractor = PDFExtractor(args.pdf_path)
    doc = extractor.extract()

    console.print(Panel.fit(
        f"[bold]Inferred Title:[/] {doc.inferred_title}\n"
        f"[bold]Total Pages:[/] {doc.total_pages}\n"
        f"[bold]Metadata:[/] {doc.metadata}",
        title="Document Profile"
    ))

    # Pages breakdown
    table = Table(title="Extracted Document Structure")
    table.add_column("Page", justify="center", style="cyan")
    table.add_column("Headings Detected", style="green")
    table.add_column("Tables", justify="center", style="magenta")
    table.add_column("Equations", justify="center", style="yellow")

    for p in doc.pages:
        h_names = ", ".join([h["text"][:30] for h in p.headings]) or "(None)"
        table.add_row(
            str(p.page_number),
            h_names,
            str(len(p.tables)),
            "Yes" if p.has_equations else "No"
        )
    console.print(table)


def handle_summarize(args):
    from agent.tool_orchestrator import ToolDrivenOrchestrator
    console.print(f"[bold green]Starting 4-Tool Pipeline Analysis for:[/] {args.pdf_path}")
    llm = GeminiLLMClient(api_key=args.api_key, model_name=args.model)
    if llm.is_available():
        console.print(f"[bold cyan]Connected to Google Gemini ({llm.model_name})[/]")
    else:
        console.print("[yellow]Gemini API key not detected. Running in heuristic mode.[/]")

    with console.status("[bold green]Executing 4-Tool Pipeline (JSON Formatter -> Sanitizer -> Analytics -> Grounding)..."):
        orchestrator = ToolDrivenOrchestrator(llm)
        res = orchestrator.process_pdf(args.pdf_path, level=args.level.lower(), custom_instructions=args.custom_prompt)
        summary_text = res["summary"]

    # Print Tool Traces
    console.print("\n[bold yellow]══ 4-Tool Execution Traces ══[/]")
    for t in res.get("tool_traces", []):
        console.print(f"  [bold green]✓ Step {t['step']}[/] [cyan]{t['tool']}[/]: {t['message']}")

    # Print Table Analytics
    ta = res.get("table_analytics", {}).get("analyses", [])
    if ta:
        console.print("\n[bold magenta]══ Tool 3: Mathematical Table Analytics ══[/]")
        for t in ta:
            for mi in t.get("mathematical_insights", []):
                console.print(f"  • {mi}")

    console.print("\n" + "=" * 60 + "\n")
    console.print(Markdown(summary_text))
    console.print("\n" + "=" * 60 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary_text)
        console.print(f"[bold green]Summary saved to:[/] {args.output}")


def handle_qa(args):
    console.print(f"[bold blue]PDF Q&A Mode:[/] {args.pdf_path}")
    console.print(f"[bold cyan]Question:[/] {args.query}\n")

    llm = GeminiLLMClient(api_key=args.api_key, model_name=args.model)
    extractor = PDFExtractor(args.pdf_path)
    doc = extractor.extract()
    qa_engine = PDFQuestionAnswerer(llm)

    with console.status("[bold blue]Locating relevant pages and synthesizing answer..."):
        answer = qa_engine.answer_question(doc, args.query)

    console.print(Panel(Markdown(answer), title="Agent Answer", subtitle=f"Source: {doc.filename}"))


def main():
    parser = argparse.ArgumentParser(description="PDF Summarization and Analysis Agent (Google Gemini Powered)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect structural hierarchy, tables, and pages of a PDF")
    inspect_parser.add_argument("pdf_path", help="Path to PDF file")

    # Summarize command
    sum_parser = subparsers.add_parser("summarize", help="Generate structured summary of PDF")
    sum_parser.add_argument("pdf_path", help="Path to PDF file")
    sum_parser.add_argument(
        "--level",
        choices=["executive", "detailed", "medium", "short", "default", "custom"],
        default="executive",
        help="Summary level (default: executive)"
    )
    sum_parser.add_argument("--custom-prompt", help="Custom prompt when level is 'custom'")
    sum_parser.add_argument("--output", "-o", help="Save summary output to markdown file")
    sum_parser.add_argument("--api-key", help="Google Gemini API Key (or set GEMINI_API_KEY env var)")
    sum_parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model (default: gemini-1.5-flash)")

    # QA command
    qa_parser = subparsers.add_parser("qa", help="Ask grounded questions about the PDF with page citations")
    qa_parser.add_argument("pdf_path", help="Path to PDF file")
    qa_parser.add_argument("query", help="Question to ask about the PDF content")
    qa_parser.add_argument("--api-key", help="Google Gemini API Key")
    qa_parser.add_argument("--model", default="gemini-1.5-flash")

    args = parser.parse_args()

    if args.command == "inspect":
        handle_inspect(args)
    elif args.command == "summarize":
        handle_summarize(args)
    elif args.command == "qa":
        handle_qa(args)


if __name__ == "__main__":
    main()
