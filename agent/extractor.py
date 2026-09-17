"""PDF Document Extractor using PyMuPDF (fitz) and pdfplumber."""
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


@dataclass
class ExtractedTable:
    page_num: int
    data: List[List[str]]
    markdown: str


@dataclass
class PageContent:
    page_number: int
    raw_text: str
    headings: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    has_equations: bool = False
    equations: List[str] = field(default_factory=list)
    images_count: int = 0


@dataclass
class ExtractedDocument:
    filename: str
    total_pages: int
    metadata: Dict[str, Any]
    pages: List[PageContent]
    full_text: str
    inferred_title: str
    table_of_contents: List[Dict[str, Any]] = field(default_factory=list)

    def get_page(self, page_num: int) -> Optional[PageContent]:
        if 1 <= page_num <= len(self.pages):
            return self.pages[page_num - 1]
        return None

    def get_text_with_page_markers(self) -> str:
        chunks = []
        for page in self.pages:
            chunks.append(f"--- PAGE {page.page_number} ---\n{page.raw_text.strip()}\n")
        return "\n".join(chunks)


class PDFExtractor:
    """Extracts structured text, headings, tables, and equations from PDF documents."""

    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def extract(self) -> ExtractedDocument:
        pages: List[PageContent] = []
        doc_metadata: Dict[str, Any] = {}
        toc: List[Dict[str, Any]] = []

        # 1. Use PyMuPDF (fitz) for high-speed font, block, and heading extraction
        if fitz:
            doc = fitz.open(self.pdf_path)
            doc_metadata = doc.metadata or {}
            toc = [{"level": item[0], "title": item[1], "page": item[2]} for item in doc.get_toc()]
            
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                page_text = page.get_text()
                
                # Analyze text blocks for headings based on font size
                headings = []
                equations = []
                images_count = len(page.get_images())
                
                blocks = page.get_text("dict").get("blocks", [])
                font_sizes = []
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            for s in l["spans"]:
                                font_sizes.append(s["size"])

                median_font_size = 10.0
                if font_sizes:
                    font_sizes.sort()
                    median_font_size = font_sizes[len(font_sizes) // 2]

                for b in blocks:
                    if "lines" in b:
                        block_text = ""
                        max_size = 0.0
                        is_bold = False
                        for l in b["lines"]:
                            for s in l["spans"]:
                                block_text += s["text"] + " "
                                if s["size"] > max_size:
                                    max_size = s["size"]
                                if "bold" in s["font"].lower() or "black" in s["font"].lower():
                                    is_bold = True
                        
                        clean_block = block_text.strip()
                        if clean_block and max_size >= median_font_size * 1.25 and len(clean_block) < 120:
                            headings.append({
                                "text": clean_block,
                                "size": max_size,
                                "is_bold": is_bold,
                                "page": page_num
                            })

                # Check for equations (LaTeX or symbols)
                math_patterns = [
                    r'\$\$.*?\$\$',
                    r'\$.*?\$',
                    r'\\begin\{equation\}.*?\\end\{equation\}',
                    r'\\sum|\\int|\\prod|\\partial|\\alpha|\\beta|\\gamma|\\theta|\\sigma|\\in|\\approx|\\leq|\\geq'
                ]
                for pattern in math_patterns:
                    matches = re.findall(pattern, page_text, re.DOTALL)
                    if matches:
                        equations.extend(matches)

                pages.append(PageContent(
                    page_number=page_num,
                    raw_text=page_text,
                    headings=headings,
                    tables=[],
                    has_equations=bool(equations),
                    equations=equations,
                    images_count=images_count
                ))
            doc.close()

        # 2. Use pdfplumber for high-precision table extraction
        if pdfplumber:
            try:
                with pdfplumber.open(self.pdf_path) as plumber_pdf:
                    for i, p_page in enumerate(plumber_pdf.pages):
                        if i < len(pages):
                            extracted_tables = p_page.extract_tables()
                            for t in extracted_tables:
                                if t and len(t) > 1:
                                    # Convert to markdown
                                    cleaned_table = []
                                    for row in t:
                                        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                                        if any(cleaned_row):
                                            cleaned_table.append(cleaned_row)
                                    
                                    if len(cleaned_table) > 1:
                                        header = cleaned_table[0]
                                        col_count = len(header)
                                        md = "| " + " | ".join(header) + " |\n"
                                        md += "| " + " | ".join(["---"] * col_count) + " |\n"
                                        for row in cleaned_table[1:]:
                                            padded_row = row + [""] * (col_count - len(row))
                                            md += "| " + " | ".join(padded_row[:col_count]) + " |\n"
                                        
                                        pages[i].tables.append(ExtractedTable(
                                            page_num=i+1,
                                            data=cleaned_table,
                                            markdown=md
                                        ))
            except Exception:
                pass

        # Inferred Title
        inferred_title = doc_metadata.get("title") or ""
        if not inferred_title and pages and pages[0].headings:
            inferred_title = pages[0].headings[0]["text"]
        elif not inferred_title and pages:
            first_lines = [l.strip() for l in pages[0].raw_text.split("\n") if l.strip()]
            inferred_title = first_lines[0] if first_lines else os.path.basename(self.pdf_path)

        full_text = "\n\n".join([f"[Page {p.page_number}]\n" + p.raw_text for p in pages])

        return ExtractedDocument(
            filename=os.path.basename(self.pdf_path),
            total_pages=len(pages),
            metadata=doc_metadata,
            pages=pages,
            full_text=full_text,
            inferred_title=inferred_title,
            table_of_contents=toc
        )
