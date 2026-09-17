<div align="center">

# 📑 Intelligent PDF Summarization Agent
### *Deterministic 4-Tool Pipeline • Ground-Truth JSON Schema • Zero-Hallucination Summaries*

[![Live Demo on Vercel](https://img.shields.io/badge/Vercel-Live%20Demo-black?style=for-the-badge&logo=vercel)](https://agent-pdf-summary.vercel.app)
[![Powered by Gemini](https://img.shields.io/badge/Google%20Gemini-2.5--Flash-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>A production-grade PDF analysis system that eliminates LLM hallucinations by enforcing a deterministic Python pre-processing pipeline before passing verified structured data to Google Gemini.</b>
</p>

[🌐 **Live Web App**](https://agent-pdf-summary.vercel.app) • [📖 **10-Step Workflow**](#-10-step-cognitive-analysis-workflow) • [🛠️ **Tools Pipeline**](#-deterministic-4-tool-architecture) • [🚀 **Quickstart**](#-quickstart-guide) • [📡 **API Docs**](#-api-endpoints)

</div>

---

## 🎯 Overview

Most standard AI PDF summarizers pass raw, noisy OCR or extracted text directly into an LLM context. This frequently leads to **hallucinated citations, garbled tables, missed math equations, and fabricated statistics**.

This **Intelligent PDF Summarization Agent** eliminates that risk through a dual-stage architecture:
1. It parses the document through **3 deterministic, code-level pre-processing tools** (hierarchical JSON parsing, header/footer noise cleaning, and mathematical table analytics).
2. It constructs an authoritative **Ground-Truth JSON Schema**.
3. Google Gemini synthesizes the summary strictly grounded in the verified JSON context.
4. Finally, **Tool 4 (Fact & Citation Checker)** cross-examines the LLM's output against the raw ground truth, verifying all `(p. X)` citations and metrics to produce a verifiable Grounding Score.

---

## 🧭 Deterministic 4-Tool Architecture

The web interface features an interactive left sidebar visualizing the sequential execution pipeline:

```
[ PDF Document Upload ]
         │
         ▼
┌──────────────────────────────────────────────┐
│  TOOL 1: JSON Formatter Tool                 │
│  • Extracts font hierarchy, headings & pages │
│  • Converts layout into standard JSON schema │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  TOOL 2: Noise Sanitizer Tool                │
│  • Strips running headers & page footers     │
│  • Normalizes ligatures & line-wraps         │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  TOOL 3: Table Analytics Tool                │
│  • Calculates column min/max/average         │
│  • Computes deltas & benchmark % gains       │
└──────────────────────────────────────────────┘
         │
         ▼
════════════════════════════════════════════════
  ⭐ NOW READY TO GO FOR LLM
  Verified Ground-Truth Structured JSON Context
════════════════════════════════════════════════
         │
         ▼
┌──────────────────────────────────────────────┐
│  GOOGLE GEMINI 2.5 FLASH SYNTHESIS           │
│  Multi-Level Summarization & Grounded QA     │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│  TOOL 4: Fact & Citation Checker             │
│  • Cross-checks all (p. X) page citations    │
│  • Verifies claimed statistics vs JSON       │
│  • Generates Grounding & Fact-Check Score    │
└──────────────────────────────────────────────┘
```

### 1. `PDFToStructuredJSONTool` (`tools/json_formatter_tool.py`)
- Employs `PyMuPDF` (`fitz`) and dual-stage structural parsing.
- Extracts document metadata, font-size hierarchy, page blocks, sections, equations, and tables.
- Standardizes raw PDFs into a predictable JSON schema.

### 2. `NoiseCleanerTool` (`tools/noise_cleaner_tool.py`)
- Programmatically identifies repeating running headers and footers across pages.
- Normalizes broken hyphenated words across line breaks.
- Fixes unicode ligatures (`fi`, `fl`, `ff`, `ffi`) that confuse tokenizers.

### 3. `TableAnalyticsTool` (`tools/table_analytics_tool.py`)
- Parses raw tabular numbers directly with Python.
- Computes baseline vs. proposed percentage gains, delta improvements, and column statistics.
- Feeds verified mathematical facts to the LLM so it never guesses percentages or numbers.

### 4. `FactAndCitationCheckerTool` (`tools/fact_checker_tool.py`)
- Post-synthesis verification tool.
- Regex audits all `(p. X)` citations against existing pages in the JSON ground truth.
- Cross-verifies extracted percentages, latencies, and metrics.
- Outputs an objective Grounding Score (e.g., `100% Grounded`).

---

## 🧠 10-Step Cognitive Analysis Workflow

The agent internally runs a 10-step analytical workflow before generating summaries:

| Step | Focus Area | Description |
| :--- | :--- | :--- |
| **01** | **Document Understanding** | Infers document type (research paper, report, manual), intended audience, and topic complexity. |
| **02** | **Core Idea Extraction** | Extracts the primary thesis, problem statement, and novel contributions. |
| **03** | **Hierarchy Mapping** | Maps logical section flow (Abstract → Methodology → Experiments → Limitations). |
| **04** | **Central Message** | Distills the paper into a single authoritative core message. |
| **05** | **Technical Mechanics** | Catalogs explicit algorithms, architectures, theorems, and formulas. |
| **06** | **Quantitative Data** | Extracts tables and benchmarks with exact numerical values. |
| **07** | **Visual Concepts** | Identifies figures, diagrams, and conceptual architectures. |
| **08** | **Claims vs Assumptions** | Distinguishes empirically validated claims from assumptions. |
| **09** | **Key Findings** | Synthesizes verified results with traceable page numbers. |
| **10** | **Limitations & Uncertainty** | Explicitly catalogs stated caveats, failure modes, and open questions. |

---

## 📊 Multi-Level Summary Formats

Select from 5 distinct summary formats tailored to your reading workflow:

1. **Executive Summary**:
   - Problem Statement & Significance
   - Key Innovation / Proposal
   - Empirical Results & Gains
   - Stated Limitations & Operational Impact
2. **Detailed Analysis**:
   - In-depth, section-by-section breakdown.
   - Comprehensive technical mechanics and equations.
3. **Medium Digest**:
   - High-level overview, core concepts, and key findings.
4. **Short Digest**:
   - Quick executive overview, 5–8 bullet points, and conclusion.
5. **Custom Instructions**:
   - Free-form instructions (e.g. *"Create revision notes for an exam"*, *"Generate interview questions from this paper"*).

---

## 🚀 Quickstart Guide

### Option A: Use the Live Web App
Navigate directly to **[https://agent-pdf-summary.vercel.app](https://agent-pdf-summary.vercel.app)** — pre-configured with Google Gemini!

---

### Option B: Run Locally

#### 1. Clone the repository
```bash
git clone https://github.com/aiwithrajan/summary.git
cd summary
```

#### 2. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Set up Environment Variables
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

#### 4. Run the FastAPI Server
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```
Open **`http://localhost:8080`** in your browser.

---

### Option C: Terminal CLI Usage

You can also run the agent completely from your terminal without opening a browser:

```bash
# 1. Inspect structure, headings, equations, and tables
python3 cli.py inspect sample_research_paper.pdf

# 2. Generate an Executive Summary
python3 cli.py summarize sample_research_paper.pdf --level executive

# 3. Generate a Short Digest
python3 cli.py summarize sample_research_paper.pdf --level short

# 4. Ask grounded, page-traceable questions
python3 cli.py qa sample_research_paper.pdf "What was the P99 latency reduction?"
```

---

## 🌐 Deploying to Vercel

The project is pre-configured for Vercel Serverless deployment with `@vercel/python`.

1. Fork or push this repo to your GitHub account (`aiwithrajan/summary`).
2. Go to **[Vercel Dashboard](https://vercel.com/new)** and import the project.
3. Add the following **Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API Key.
   - `GEMINI_MODEL`: `gemini-2.5-flash`.
4. Click **Deploy**. Vercel will build the serverless function and serve the dark web interface!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the dark-themed web application. |
| `GET` | `/api/config` | Returns pre-configured model and configuration state. |
| `GET` | `/api/models` | Lists available Google Gemini models for the provided key. |
| `POST` | `/api/upload` | Uploads PDF and executes Tools 1, 2, and 3 (JSON parsing & table analytics). |
| `POST` | `/api/summarize` | Executes LLM synthesis and Tool 4 Fact-Checking on the structured JSON. |
| `POST` | `/api/qa` | Answers page-traceable questions grounded strictly in the PDF text. |
| `GET` | `/api/structured-json` | Returns the raw deterministic Ground-Truth JSON Schema. |
| `GET` | `/api/inspect` | Returns document hierarchy, extracted headings, and pages. |

---

## 📂 Project Structure

```
├── agent/
│   ├── analyzer.py            # 10-Step cognitive analysis engine
│   ├── extractor.py           # Dual-engine PDF structural parser
│   ├── llm.py                 # Google Gemini 2.5 Flash client & offline fallback
│   ├── qa.py                  # Page-traceable grounded Q&A engine
│   ├── summarizer.py          # Multi-level summary generator
│   └── tool_orchestrator.py   # Orchestrates deterministic 4-tool execution
├── tools/
│   ├── __init__.py            # Tool exports
│   ├── json_formatter_tool.py # Tool 1: PDF to standard JSON schema
│   ├── noise_cleaner_tool.py  # Tool 2: Header/footer & ligature cleaner
│   ├── table_analytics_tool.py# Tool 3: Mathematical table & delta analytics
│   └── fact_checker_tool.py   # Tool 4: Citation & fact validation checker
├── api/
│   └── index.py               # Vercel serverless entry point
├── static/
│   └── index.html             # Responsive dark UI with Tools Sidebar pipeline
├── server.py                  # FastAPI web server
├── cli.py                     # Command-line interface
├── vercel.json                # Vercel serverless build configuration
├── requirements.txt           # Production Python dependencies
└── README.md                  # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/aiwithrajan">Rajan Mishra</a></sub>
</div>
