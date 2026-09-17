"""Streamlit Web Application for the PDF Summarization Agent."""
import os
import tempfile
import streamlit as st

from agent.extractor import PDFExtractor, ExtractedDocument
from agent.analyzer import DocumentAnalyzer
from agent.summarizer import PDFSummarizer, SummaryLevel
from agent.qa import PDFQuestionAnswerer
from agent.llm import GeminiLLMClient, get_available_gemini_models

# Page configuration
st.set_page_config(
    page_title="PDF Summarization Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A73E8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5F6368;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1A73E8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: Configurations
st.sidebar.title("⚙️ Agent Settings")

api_key_env = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=api_key_env,
    type="password",
    help="Enter your Google Gemini API key. You can get one from Google AI Studio (aistudio.google.com)"
)

available_models = get_available_gemini_models(api_key if api_key else None)
model_name = st.sidebar.selectbox(
    "Gemini Model",
    options=available_models,
    index=0
)

temperature = st.sidebar.slider(
    "Temperature (Precision vs Creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

st.sidebar.divider()
st.sidebar.subheader("Summary Configuration")
summary_level_choice = st.sidebar.selectbox(
    "Summary Level",
    options=[
        "Executive (Problem, Significance, Results, Limitations)",
        "Detailed (Section-by-section, In-depth)",
        "Medium (Overview, Concepts, Findings)",
        "Short (Purpose, 5-10 Points, Conclusion)",
        "Default Structured",
        "Custom Instructions"
    ],
    index=0
)

level_mapping = {
    "Executive (Problem, Significance, Results, Limitations)": SummaryLevel.EXECUTIVE,
    "Detailed (Section-by-section, In-depth)": SummaryLevel.DETAILED,
    "Medium (Overview, Concepts, Findings)": SummaryLevel.MEDIUM,
    "Short (Purpose, 5-10 Points, Conclusion)": SummaryLevel.SHORT,
    "Default Structured": SummaryLevel.DEFAULT,
    "Custom Instructions": SummaryLevel.CUSTOM,
}
selected_level = level_mapping[summary_level_choice]

custom_instructions = None
if selected_level == SummaryLevel.CUSTOM:
    custom_instructions = st.sidebar.text_area(
        "Custom Prompt Instructions",
        placeholder="E.g., Generate bullet-point revision notes for a computer science exam..."
    )

st.markdown('<div class="main-header">📄 Intelligent PDF Summarization Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep structural analysis, faithful extraction, zero-hallucination guarantees, and page traceability. Powered by Google Gemini.</div>', unsafe_allow_html=True)

# LLM Client Initialization
llm_client = GeminiLLMClient(api_key=api_key, model_name=model_name, temperature=temperature)

if not api_key:
    st.info("💡 **Tip:** Add your Google Gemini API Key in the left sidebar to enable full AI synthesis. Without an API key, the agent will operate in local structural fallback mode.")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document to analyze", type=["pdf"])

if uploaded_file is not None:
    # Cache extraction so it doesn't re-parse on every widget interaction
    @st.cache_data(show_spinner="Extracting document structure & tables...")
    def parse_uploaded_pdf(file_bytes, filename):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            extractor = PDFExtractor(tmp_path)
            doc = extractor.extract()
            doc.filename = filename
            return doc
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    doc: ExtractedDocument = parse_uploaded_pdf(uploaded_file.getvalue(), uploaded_file.name)

    # Document Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", doc.total_pages)
    with col2:
        total_tables = sum(len(p.tables) for p in doc.pages)
        st.metric("Tables Detected", total_tables)
    with col3:
        has_eqs = any(p.has_equations for p in doc.pages)
        st.metric("Contains Math / Equations", "Yes" if has_eqs else "No")
    with col4:
        total_images = sum(p.images_count for p in doc.pages)
        st.metric("Images / Figures", total_images)

    st.markdown(f"**Inferred Document Title:** `{doc.inferred_title}`")

    # Tabs for Summary, Q&A, Inspection
    tab_summary, tab_qa, tab_inspector, tab_steps = st.tabs([
        "📋 Document Summary",
        "💬 Interactive Q&A (Page Traceable)",
        "🔍 Document Structure & Tables",
        "⚙️ 10-Step Workflow Analysis"
    ])

    # --- TAB 1: SUMMARY ---
    with tab_summary:
        col_gen, col_download = st.columns([3, 1])
        with col_gen:
            generate_btn = st.button("🚀 Generate / Refresh Summary", type="primary", use_container_width=True)

        if "current_summary" not in st.session_state or generate_btn:
            if generate_btn or "current_summary" not in st.session_state:
                with st.spinner("Analyzing document hierarchy and synthesizing summary..."):
                    summarizer = PDFSummarizer(llm_client)
                    summary_text = summarizer.summarize(
                        doc,
                        level=selected_level,
                        custom_instructions=custom_instructions
                    )
                    st.session_state["current_summary"] = summary_text

        st.markdown(st.session_state.get("current_summary", ""))

        st.download_button(
            label="📥 Download Summary as Markdown",
            data=st.session_state.get("current_summary", ""),
            file_name=f"{uploaded_file.name}_summary.md",
            mime="text/markdown"
        )

    # --- TAB 2: Q&A ---
    with tab_qa:
        st.subheader("Ask Grounded Questions About This PDF")
        st.caption("Answers cite specific page numbers and will never invent information outside the PDF.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a question (e.g., 'What was the performance improvement?' or 'What are the stated limitations?')")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching document pages and formulating answer..."):
                    qa_engine = PDFQuestionAnswerer(llm_client)
                    answer = qa_engine.answer_question(doc, user_query)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- TAB 3: INSPECTION ---
    with tab_inspector:
        st.subheader("Document Hierarchy & Tables")
        for p in doc.pages:
            with st.expander(f"📄 Page {p.page_number} ({len(p.headings)} Headings, {len(p.tables)} Tables)"):
                if p.headings:
                    st.markdown("**Headings:**")
                    for h in p.headings:
                        st.markdown(f"- `{h['text']}` (Size: {h['size']:.1f}, Bold: {h['is_bold']})")
                
                if p.tables:
                    st.markdown("**Extracted Tables:**")
                    for t in p.tables:
                        st.markdown(t.markdown)

                st.markdown("**Raw Page Text:**")
                st.text(p.raw_text[:800] + ("..." if len(p.raw_text) > 800 else ""))

    # --- TAB 4: 10-STEP ANALYSIS ---
    with tab_steps:
        st.subheader("10-Step Document Understanding Workflow")
        st.caption("Visualizing the internal cognitive workflow performed before final summarization.")

        analyzer = DocumentAnalyzer(llm_client)
        analysis = analyzer.analyze(doc)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Step 1: Document Type & Topic**\n- Type: `{analysis.doc_type}`\n- Topic: `{analysis.main_topic}`\n- Audience: `{analysis.intended_audience}`\n- Complexity: `{analysis.complexity}`")
            st.markdown(f"**Step 4: Central Message**\n- {analysis.central_message}")
            st.markdown("**Step 2 & 3: Key Points**")
            for kp in analysis.key_points:
                st.markdown(f"- {kp}")

        with col_b:
            st.markdown("**Step 5: Major Sections Identified**")
            for s in analysis.major_sections:
                st.markdown(f"- {s}")
            st.markdown("**Step 9: Key Findings & Claims**")
            for f in analysis.key_findings:
                st.markdown(f"- {f}")
            st.markdown("**Step 10: Explicit Limitations & Uncertainty**")
            for l in analysis.limitations:
                st.markdown(f"- {l}")
