"""FastAPI Web Server for the PDF Summarization Agent powered by the 4-Tool Deterministic Pipeline."""
import os
import shutil
import tempfile
import json
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.extractor import PDFExtractor, ExtractedDocument
from agent.analyzer import DocumentAnalyzer
from agent.summarizer import PDFSummarizer, SummaryLevel
from agent.qa import PDFQuestionAnswerer
from agent.llm import GeminiLLMClient, get_available_gemini_models
from agent.tool_orchestrator import ToolDrivenOrchestrator
from tools import PDFToStructuredJSONTool, NoiseCleanerTool, TableAnalyticsTool, FactAndCitationCheckerTool

app = FastAPI(title="PDF Summarization Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document session cache
CURRENT_DOC: Optional[ExtractedDocument] = None
CURRENT_PDF_PATH: Optional[str] = None
CURRENT_STRUCTURED_JSON: Optional[Dict[str, Any]] = None
CURRENT_ORCHESTRATOR_RESULT: Optional[Dict[str, Any]] = None

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "pdf_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class SummarizeRequest(BaseModel):
    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    level: str = "executive"
    custom_prompt: Optional[str] = None


class QARequest(BaseModel):
    query: str
    api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2


@app.get("/api/config")
async def get_config():
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    env_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return {
        "api_key": env_key,
        "default_model": env_model,
        "is_configured": bool(env_key)
    }


@app.get("/api/models")
async def list_models(api_key: Optional[str] = None):
    models = get_available_gemini_models(api_key)
    return {"models": models}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global CURRENT_DOC, CURRENT_PDF_PATH, CURRENT_STRUCTURED_JSON
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        CURRENT_PDF_PATH = file_path
        extractor = PDFExtractor(file_path)
        doc = extractor.extract()
        doc.filename = file.filename
        CURRENT_DOC = doc

        # Execute Tool 1 (Structured JSON), Tool 2 (Noise Cleaner), Tool 3 (Table Analytics)
        t1 = PDFToStructuredJSONTool()
        t2 = NoiseCleanerTool()
        t3 = TableAnalyticsTool()

        raw_json = t1.execute(file_path)
        cleaned_json = t2.execute(raw_json)
        analyzed_json = t3.execute(cleaned_json)
        CURRENT_STRUCTURED_JSON = analyzed_json

        meta = analyzed_json["document_metadata"]

        return {
            "filename": doc.filename,
            "title": meta.get("inferred_title") or doc.inferred_title,
            "total_pages": meta.get("total_pages", doc.total_pages),
            "tables_detected": meta.get("total_tables", 0),
            "contains_math": meta.get("total_equations", 0) > 0,
            "equations_count": meta.get("total_equations", 0),
            "images_count": sum(p.images_count for p in doc.pages),
            "metadata": doc.metadata,
            "table_analytics": analyzed_json.get("table_analytics", {}),
            "sections_count": len(analyzed_json.get("section_hierarchy", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/api/summarize")
async def summarize_pdf(req: SummarizeRequest):
    global CURRENT_PDF_PATH, CURRENT_STRUCTURED_JSON, CURRENT_ORCHESTRATOR_RESULT
    if not CURRENT_PDF_PATH or not os.path.exists(CURRENT_PDF_PATH):
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet.")

    try:
        api_key = req.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        llm = GeminiLLMClient(
            api_key=api_key,
            model_name=req.model,
            temperature=req.temperature
        )

        # Run 4-tool pipeline orchestrator
        orchestrator = ToolDrivenOrchestrator(llm)
        res = orchestrator.process_pdf(
            CURRENT_PDF_PATH,
            level=req.level,
            custom_instructions=req.custom_prompt
        )

        CURRENT_ORCHESTRATOR_RESULT = res
        CURRENT_STRUCTURED_JSON = res["structured_json"]

        return {
            "summary": res["summary"],
            "raw_summary": res["raw_summary"],
            "level": req.level,
            "title": res["title"],
            "tool_traces": res["tool_traces"],
            "table_analytics": res["table_analytics"],
            "verification": res["verification"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@app.get("/api/structured-json")
async def get_structured_json():
    global CURRENT_STRUCTURED_JSON
    if not CURRENT_STRUCTURED_JSON:
        raise HTTPException(status_code=400, detail="No structured JSON available yet. Please upload a PDF.")
    return CURRENT_STRUCTURED_JSON


@app.post("/api/qa")
async def answer_question(req: QARequest):
    global CURRENT_DOC
    if not CURRENT_DOC:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    try:
        api_key = req.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        llm = GeminiLLMClient(
            api_key=api_key,
            model_name=req.model,
            temperature=req.temperature
        )
        qa = PDFQuestionAnswerer(llm)
        answer = qa.answer_question(CURRENT_DOC, req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A error: {str(e)}")


@app.get("/api/inspect")
async def inspect_document():
    global CURRENT_STRUCTURED_JSON, CURRENT_DOC
    if not CURRENT_STRUCTURED_JSON:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    return CURRENT_STRUCTURED_JSON


# Serve static web interface
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>PDF Agent Server Running</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
