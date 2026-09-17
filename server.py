"""FastAPI Web Server for the PDF Summarization Agent with Custom Modern Dark UI."""
import os
import shutil
import tempfile
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
    global CURRENT_DOC
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extractor = PDFExtractor(file_path)
        doc = extractor.extract()
        doc.filename = file.filename
        CURRENT_DOC = doc

        total_tables = sum(len(p.tables) for p in doc.pages)
        total_equations = sum(len(p.equations) for p in doc.pages)
        total_images = sum(p.images_count for p in doc.pages)

        return {
            "filename": doc.filename,
            "title": doc.inferred_title,
            "total_pages": doc.total_pages,
            "tables_detected": total_tables,
            "contains_math": total_equations > 0 or any(p.has_equations for p in doc.pages),
            "images_count": total_images,
            "metadata": doc.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/api/summarize")
async def summarize_pdf(req: SummarizeRequest):
    global CURRENT_DOC
    if not CURRENT_DOC:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet.")

    try:
        llm = GeminiLLMClient(
            api_key=req.api_key,
            model_name=req.model,
            temperature=req.temperature
        )
        summarizer = PDFSummarizer(llm)
        level_enum = SummaryLevel(req.level.lower())
        
        summary_markdown = summarizer.summarize(
            CURRENT_DOC,
            level=level_enum,
            custom_instructions=req.custom_prompt
        )

        return {
            "summary": summary_markdown,
            "level": req.level,
            "title": CURRENT_DOC.inferred_title
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@app.post("/api/qa")
async def answer_question(req: QARequest):
    global CURRENT_DOC
    if not CURRENT_DOC:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    try:
        llm = GeminiLLMClient(
            api_key=req.api_key,
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
    global CURRENT_DOC
    if not CURRENT_DOC:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    pages_data = []
    for p in CURRENT_DOC.pages:
        tables_data = [{"markdown": t.markdown, "page": t.page_num} for t in p.tables]
        pages_data.append({
            "page_number": p.page_number,
            "headings": p.headings,
            "tables": tables_data,
            "has_equations": p.has_equations,
            "raw_text_preview": p.raw_text[:1200]
        })

    return {
        "filename": CURRENT_DOC.filename,
        "title": CURRENT_DOC.inferred_title,
        "total_pages": CURRENT_DOC.total_pages,
        "pages": pages_data
    }


@app.get("/api/analysis-steps")
async def get_analysis_steps(api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
    global CURRENT_DOC
    if not CURRENT_DOC:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    llm = GeminiLLMClient(api_key=api_key, model_name=model)
    analyzer = DocumentAnalyzer(llm)
    analysis = analyzer.analyze(CURRENT_DOC)

    return {
        "title": analysis.title,
        "doc_type": analysis.doc_type,
        "main_topic": analysis.main_topic,
        "purpose": analysis.purpose,
        "intended_audience": analysis.intended_audience,
        "complexity": analysis.complexity,
        "central_message": analysis.central_message,
        "key_points": analysis.key_points,
        "major_sections": analysis.major_sections,
        "key_findings": analysis.key_findings,
        "limitations": analysis.limitations,
        "uncertainties": analysis.uncertainties
    }


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
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
