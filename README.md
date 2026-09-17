# Intelligent PDF Summarization Agent

An intelligent PDF analysis and summarization agent featuring deep structural document analysis, hallucination prevention, multi-level summarization (Executive, Detailed, Short, Medium, Custom), grounded Q&A with page citations, and a modern responsive dark web interface.

Powered by **Google Gemini API** (`gemini-2.5-flash`).

![Web UI Preview](https://raw.githubusercontent.com/aiwithrajan/summary/main/preview.png)

---

## 🌟 Key Features

- **10-Step Internal Cognitive Workflow**: Parses document structure, analyzes hierarchy, identifies central messages, validates tables and formulas, and explicitly flags uncertainties.
- **Strict Hallucination Prevention**: Never assumes missing information.
- **Executive Summary Engine**: Focuses on Problem Statement, Significance, Proposals, Results, Implications, and Limitations.
- **Page-Traceable Q&A**: Answers queries with direct references (e.g., `(p. 3)` or `(Section 2.1)`).
- **FastAPI & Modern Web UI**: Sleek dark theme (`#0e1117`, `#FF4B4B` coral red accents) with instant responses, drag-and-drop file upload, markdown rendering, and copy/download controls.
- **Vercel Serverless Ready**: Configured with `vercel.json` and `api/index.py` for one-click deployment.

---

## 🚀 Live Vercel Deployment

### Deploy with Vercel:
1. Import this repository into **[Vercel](https://vercel.com)**:
   - Repository: `aiwithrajan/summary`
2. In the **Environment Variables** section on Vercel, add:
   - `GEMINI_API_KEY`: `your_gemini_api_key_here`
   - `GEMINI_MODEL`: `gemini-2.5-flash`
3. Click **Deploy**.

---

## 💻 Local Setup & Execution

### 1. Clone & Install
```bash
git clone https://github.com/aiwithrajan/summary.git
cd summary
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```bash
cp .env.example .env
```
Add your Google Gemini API key:
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 3. Run the Web Interface
```bash
python3 -m uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```
Open **`http://localhost:8080`** in your browser.

### 4. Run via CLI
```bash
# Inspect structure & tables
python3 cli.py inspect sample_research_paper.pdf

# Generate an Executive Summary
python3 cli.py summarize sample_research_paper.pdf --level executive

# Ask grounded questions
python3 cli.py qa sample_research_paper.pdf "What was the latency reduction achieved?"
```

---

## 🏗️ Architecture

```
├── api/
│   └── index.py            # Vercel serverless entry point
├── agent/
│   ├── extractor.py        # PyMuPDF/pdfplumber parser (headings, tables, equations)
│   ├── analyzer.py         # 10-step analytical workflow
│   ├── summarizer.py       # Multi-level summary generator
│   ├── qa.py               # Grounded Q&A engine
│   └── llm.py              # Google Gemini client with dynamic fallback
├── static/
│   └── index.html          # Modern dark-themed frontend
├── server.py               # FastAPI application backend
├── cli.py                  # Terminal CLI interface
├── vercel.json             # Vercel deployment configuration
└── requirements.txt        # Dependencies
```
