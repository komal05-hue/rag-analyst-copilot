import os
import json
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import gc

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(current_dir, "..", "vectorstore")
data_folder = os.path.join(current_dir, "..", "data")
analytics_path = os.path.join(current_dir, "..", "analytics.json")
os.makedirs(data_folder, exist_ok=True)

def get_stored_path(filename):
    return os.path.normpath(os.path.join(data_folder, filename))

st.set_page_config(page_title="Analyst Co-Pilot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
  .stApp { background-color: #0f1117; }
  .header-banner { background: linear-gradient(135deg, #1a1f2e, #2d3561); border: 1px solid #3d4fd1; border-radius: 16px; padding: 24px 32px; margin-bottom: 20px; }
  .header-title { font-size: 28px; font-weight: 700; color: #ffffff; margin: 0; }
  .header-subtitle { font-size: 14px; color: #8b95b5; margin: 4px 0 0 0; }
  .stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
  .stat-card { background: #1a1f2e; border: 1px solid #2d3561; border-radius: 12px; padding: 14px 18px; flex: 1; text-align: center; }
  .stat-number { font-size: 26px; font-weight: 700; color: #6c7ee1; }
  .stat-label { font-size: 11px; color: #8b95b5; text-transform: uppercase; }
  .sidebar-title { font-size: 13px; font-weight: 600; color: #6c7ee1; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
  [data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1f2537 !important; }
  .stButton button { background: linear-gradient(135deg, #3d4fd1, #6c7ee1) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
  .stChatMessage { background: #1a1f2e !important; border: 1px solid #2a2f45 !important; border-radius: 14px !important; margin-bottom: 12px !important; }
  [data-testid="stChatMessageContent"] p { color: #e8eaf6 !important; font-size: 15px !important; line-height: 1.7 !important; }
  .stChatInput { background: #1a1f2e !important; border: 2px solid #3d4fd1 !important; border-radius: 12px !important; padding: 4px 8px !important; }
  .stChatInput textarea { background: #1a1f2e !important; border: none !important; color: #ffffff !important; font-size: 15px !important; min-height: 44px !important; max-height: 44px !important; padding: 10px !important; resize: none !important; }
  .stChatInput button { background: #3d4fd1 !important; border-radius: 8px !important; }
  .block-container { padding-bottom: 120px !important; }
  .fq-btn button { background: #1a1f2e !important; border: 1px solid #3d4fd1 !important; color: #8b95b5 !important; font-size: 12px !important; font-weight: 400 !important; border-radius: 20px !important; white-space: normal !important; height: auto !important; line-height: 1.4 !important; }
  .metric-card { background: #1a1f2e; border: 1px solid #2d3561; border-radius: 12px; padding: 20px; text-align: center; }
  .metric-num { font-size: 32px; font-weight: 700; color: #6c7ee1; }
  .metric-label { font-size: 12px; color: #8b95b5; text-transform: uppercase; margin-top: 4px; }
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
for key, val in [
    ("messages", []), ("pending_question", None),
    ("uploader_key", 0), ("followups", []),
    ("compare_request", None), ("page", "💬 Chat"),
    ("doc_filter", "📂 All Documents")
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_ingested_files():
    return [f for f in os.listdir(data_folder) if f.endswith(".pdf")]

def get_chunk_count():
    try:
        index_file = os.path.join(vectorstore_path, "index.faiss")
        if os.path.exists(index_file):
            vs = FAISS.load_local(vectorstore_path, get_embeddings(), allow_dangerous_deserialization=True)
            return vs.index.ntotal
        return 0
    except:
        return 0

def ingest_pdf(uploaded_file):
    save_path = os.path.join(data_folder, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(save_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    os.makedirs(vectorstore_path, exist_ok=True)
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if os.path.exists(index_file):
        vs = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(vectorstore_path)
    return len(documents), len(chunks)

def delete_document(filename):
    try:
        file_path = os.path.join(data_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        # Rebuild vectorstore without deleted file
        import shutil
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path, ignore_errors=True)
        os.makedirs(vectorstore_path, exist_ok=True)
        remaining_files = get_ingested_files()
        if remaining_files:
            embeddings = get_embeddings()
            all_chunks = []
            for f in remaining_files:
                loader = PyPDFLoader(os.path.join(data_folder, f))
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            if all_chunks:
                vs = FAISS.from_documents(all_chunks, embeddings)
                vs.save_local(vectorstore_path)
        return True
    except Exception as e:
        return False
    
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", ""),
        temperature=0.3
    )

def get_retriever_for_file(filename):
    embeddings = get_embeddings()
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_file):
        return None
    vs = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    stored_path = get_stored_path(filename)
    return vs.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"source": stored_path}
        }
    )

def get_retriever_all_docs():
    embeddings = get_embeddings()
    index_file = os.path.join(vectorstore_path, "index.faiss")
    if not os.path.exists(index_file):
        return None
    vs = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 8})

def build_chain(retriever):
    llm = get_llm()
    template = """You are a professional analyst assistant.
Use ONLY the context below to answer the question.
The context may come from multiple different documents.
If the answer is not in the context, say "I could not find this information in the documents."

FORMATTING RULES:
- If answer spans multiple documents, clearly label each:
  📄 From [filename]:
  [answer from that document]

- Multiple points in one document:
- Point one

- Point two

- Steps: numbered list
- Single fact: one clear sentence
- Always cite: *Source: [exact filename], page [number]*
- NEVER write "filename" literally

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    def format_docs(docs):
        # Group by source file
        from collections import defaultdict
        grouped = defaultdict(list)
        for doc in docs:
            src = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', '?')
            grouped[src].append(f"[Page {page}]: {doc.page_content}")
        
        formatted = []
        for src, contents in grouped.items():
            formatted.append(f"=== From {src} ===\n" + "\n".join(contents))
        return "\n\n".join(formatted)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

def suggest_followups(question, answer):
    llm = get_llm()
    prompt = f"""Suggest exactly 3 smart follow-up questions based on this Q&A.
Question: {question}
Answer: {answer}
Rules: Each under 12 words. One per line. No numbering, no bullets, no extra text."""
    result = llm.invoke(prompt)
    lines = [l.strip() for l in result.content.strip().split('\n') if l.strip()]
    return lines[:3]

def compare_documents(question, file1, file2):
    llm = get_llm()
    retriever1 = get_retriever_for_file(file1)
    chain1 = build_chain(retriever1)
    ans1 = chain1.invoke(question)
    retriever2 = get_retriever_for_file(file2)
    chain2 = build_chain(retriever2)
    ans2 = chain2.invoke(question)
    compare_prompt = f"""Compare two documents on: {question}

Document 1 ({file1}): {ans1}
Document 2 ({file2}): {ans2}

Use this EXACT format:

**Overview**
Brief 1-2 line summary.

**Document 1 — {file1}**

- Key point 1

- Key point 2

**Document 2 — {file2}**

- Key point 1

- Key point 2

**Key Similarities**

- Similarity 1

**Key Differences**

- Difference 1

**Conclusion**
One sentence verdict.
"""
    result = llm.invoke(compare_prompt)
    return result.content, ans1, ans2

def detect_hallucination(answer, source_docs):
    if not source_docs:
        return None, None, None
    llm = get_llm()
    context = "\n\n".join([doc.page_content for doc in source_docs])
    prompt = f"""You are a fact-checking assistant.
Check if the answer is fully supported by the context.

Context: {context}
Answer: {answer}

Reply ONLY in this exact format:
SCORE: [0-100]
VERDICT: [TRUSTWORTHY / PARTIALLY SUPPORTED / POSSIBLE HALLUCINATION]
REASON: [one sentence]"""
    result = llm.invoke(prompt)
    lines = result.content.strip().split('\n')
    score, verdict, reason = 75, "PARTIALLY SUPPORTED", "Could not verify"
    for line in lines:
        if line.startswith("SCORE:"):
            try: score = int(line.replace("SCORE:", "").strip())
            except: pass
        elif line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    return score, verdict, reason

def show_hallucination_badge(score, verdict, reason):
    if score >= 80:
        color, bg, icon = "#0F6E56", "#E1F5EE", "✅"
    elif score >= 50:
        color, bg, icon = "#633806", "#FAEEDA", "⚠️"
    else:
        color, bg, icon = "#712B13", "#FAECE7", "🚨"
    st.markdown(f"""
    <div style='background:{bg};border:1px solid {color};border-radius:10px;
    padding:10px 16px;margin-top:10px;display:flex;align-items:center;gap:12px;'>
        <div style='font-size:20px;'>{icon}</div>
        <div>
            <div style='font-size:13px;font-weight:600;color:{color};'>
            {verdict} — Trust Score: {score}/100</div>
            <div style='font-size:12px;color:{color};opacity:0.8;margin-top:2px;'>
            {reason}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def save_analytics(question, doc_label, response_time, is_comparison=False, hallucination_score=None):
    if os.path.exists(analytics_path):
        with open(analytics_path, "r") as f:
            data = json.load(f)
    else:
        data = {"questions": [], "response_times": [], "docs_queried": [],
                "comparison_count": 0, "total_questions": 0, "hallucination_scores": []}
    data["questions"].append(question)
    data["response_times"].append(round(response_time, 2))
    data["docs_queried"].append(os.path.basename(doc_label).strip())
    data["total_questions"] += 1
    if is_comparison:
        data["comparison_count"] += 1
    if hallucination_score is not None:
        if "hallucination_scores" not in data:
            data["hallucination_scores"] = []
        data["hallucination_scores"].append(hallucination_score)
    with open(analytics_path, "w") as f:
        json.dump(data, f)

def load_analytics():
    if os.path.exists(analytics_path):
        with open(analytics_path, "r") as f:
            return json.load(f)
    return {"questions": [], "response_times": [], "docs_queried": [],
            "comparison_count": 0, "total_questions": 0, "hallucination_scores": []}

def process_question(question, doc_filter):
    start = time.time()
    files = get_ingested_files()

    if doc_filter == "📂 All Documents":
        retriever = get_retriever_all_docs()
        if retriever is None:
            return "Please upload documents first!", []
        chain = build_chain(retriever)
        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)
        doc_label = "All Documents"
    else:
        retriever = get_retriever_for_file(doc_filter)
        if retriever is None:
            return "Please upload documents first!", []
        chain = build_chain(retriever)
        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)
        doc_label = doc_filter

    elapsed = round(time.time() - start, 2)
    src_label = os.path.basename(
        source_docs[0].metadata.get('source', doc_label)
    ) if source_docs else doc_label
    save_analytics(question, src_label, elapsed)
    return answer, source_docs

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 16px;'>
        <div style='font-size:36px;'>🤖</div>
        <div style='font-size:17px;font-weight:700;color:#fff;'>Analyst Co-Pilot</div>
        <div style='font-size:11px;color:#8b95b5;'>Powered by Groq + LangChain</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation — clean 3 buttons only
    st.markdown("<div class='sidebar-title'>📌 Navigation</div>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("💬\nChat", use_container_width=True,
                    type="primary" if st.session_state.page == "💬 Chat" else "secondary"):
            st.session_state.page = "💬 Chat"
            st.rerun()
    with nav_col2:
        if st.button("📊\nBoard", use_container_width=True,
                    type="primary" if st.session_state.page == "📊 Dashboard" else "secondary"):
            st.session_state.page = "📊 Dashboard"
            st.rerun()
    with nav_col3:
        if st.button("🔬\nStudy", use_container_width=True,
                    type="primary" if st.session_state.page == "🔬 Ablation" else "secondary"):
            st.session_state.page = "🔬 Ablation"
            st.rerun()

    st.divider()

    files = get_ingested_files()
    chunks = get_chunk_count()

    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-card'><div class='stat-number'>{len(files)}</div><div class='stat-label'>Docs</div></div>
        <div class='stat-card'><div class='stat-number'>{chunks}</div><div class='stat-label'>Chunks</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='sidebar-title'>➕ Upload PDF</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here (max 3)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_files:
        # Limit to 3
        if len(uploaded_files) > 3:
            st.warning("Max 3 files at once!")
            uploaded_files = uploaded_files[:3]
        
        new_files = [f for f in uploaded_files if f.name not in files]
        existing = [f for f in uploaded_files if f.name in files]
        
        if existing:
            st.warning(f"Already exists: {', '.join([f.name for f in existing])}")
        
        if new_files:
            if st.button(f"⚡ Ingest {len(new_files)} PDF(s)", use_container_width=True):
                total_pages = total_chunks = 0
                for uf in new_files:
                    with st.spinner(f"Processing {uf.name}..."):
                        p, c = ingest_pdf(uf)
                        total_pages += p
                        total_chunks += c
                st.session_state.uploader_key += 1
                st.success(f"✅ {len(new_files)} files · {total_pages} pages · {total_chunks} chunks!")
                st.rerun()

    st.divider()

    # Compare section
    st.markdown("<div class='sidebar-title'>⚖️ Compare Docs</div>", unsafe_allow_html=True)
    if len(files) >= 2:
        comp_doc1 = st.selectbox("Document 1", files, key="comp1")
        comp_doc2 = st.selectbox("Document 2", [f for f in files if f != comp_doc1], key="comp2")
        comp_question = st.text_input("Topic to compare", placeholder="e.g. regulations...", key="comp_q")
        if st.button("⚖️ Compare", use_container_width=True):
            if comp_question:
                st.session_state.compare_request = {
                    "file1": comp_doc1, "file2": comp_doc2, "question": comp_question
                }
                st.session_state.page = "💬 Chat"
                st.session_state.followups = []
                st.rerun()
            else:
                st.warning("Enter a topic!")
    else:
        st.markdown("<div style='font-size:12px;color:#8b95b5;'>Upload 2+ docs to compare</div>", unsafe_allow_html=True)

    st.divider()

    st.divider()

    # Document list with delete
    st.markdown("<div class='sidebar-title'>📚 Documents</div>", unsafe_allow_html=True)
    if files:
        for f in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<div style='font-size:13px;color:#c5cae9;padding:6px 0;'>📄 {f}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{f}"):
                    delete_document(f)
                    st.rerun()
    else:
        st.markdown("<div style='font-size:13px;color:#8b95b5;'>No documents yet!</div>", unsafe_allow_html=True)

    st.divider()

    col_clr1, col_clr2 = st.columns(2)
    with col_clr1:
        if st.button("🧹 Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.followups = []
            st.session_state.pending_question = None
            st.rerun()
    with col_clr2:
        if st.button("🗑️ Docs", use_container_width=True):
            import shutil, gc
            for f in get_ingested_files():
                try:
                    os.remove(os.path.join(data_folder, f))
                except:
                    pass
            gc.collect()
            time.sleep(1)
            if os.path.exists(vectorstore_path):
                shutil.rmtree(vectorstore_path, ignore_errors=True)
            st.session_state.messages = []
            st.session_state.followups = []
            st.rerun()

    # Full Reset Button
    st.divider()
    if st.button("🔄 Full Reset Everything", use_container_width=True):
        import shutil, gc
        # Clear all documents
        for f in get_ingested_files():
            try:
                os.remove(os.path.join(data_folder, f))
            except:
                pass
        gc.collect()
        time.sleep(1)
        # Clear vectorstore
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path, ignore_errors=True)
        # Clear analytics
        if os.path.exists(analytics_path):
            os.remove(analytics_path)
        # Clear ablation results
        ablation_results = os.path.join(current_dir, "..", "ablation_results.json")
        if os.path.exists(ablation_results):
            os.remove(ablation_results)
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Everything cleared! Fresh start!")
        time.sleep(1)
        st.rerun()

    st.markdown("""
    <div style='margin-top:20px;padding-top:14px;border-top:0.5px solid #1f2537;
    font-size:11px;color:#3d4555;text-align:center;'>
        Built by Komal Sawant · MSc Data Science
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ════════════════════════════════════════════════════════════════════════
if st.session_state.page == "💬 Chat":

    st.markdown("""
    <div class='header-banner'>
        <p class='header-title'>🤖 Analyst Co-Pilot</p>
        <p class='header-subtitle'>Ask questions about your documents · Get cited AI answers instantly</p>
    </div>
    """, unsafe_allow_html=True)

    if not get_ingested_files():
        st.warning("⬅️ Upload a PDF from the sidebar first!")
        st.stop()

    if not st.session_state.messages:
        st.markdown("""
        <div style='background:#1a1f2e;border:1px solid #2a2f45;border-radius:14px;
        padding:20px 24px;margin-bottom:20px;'>
            <div style='font-size:18px;font-weight:600;color:#fff;margin-bottom:8px;'>
            👋 Hello, Komal!</div>
            <div style='font-size:14px;color:#8b95b5;line-height:1.7;'>
                Select a document from sidebar and ask any question!<br>
                Try: <span style='color:#6c7ee1;'>"What is this document about?"</span>
                or <span style='color:#6c7ee1;'>"What are the key topics?"</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View Sources"):
                    for i, src in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** · `{src['file']}` · Page `{src['page']}`")
                        st.caption(src['preview'])
                        if i < len(message["sources"]) - 1:
                            st.divider()

    # Follow-up buttons
    if (st.session_state.followups and st.session_state.messages
            and st.session_state.messages[-1]["role"] == "assistant"):
        st.markdown("<div style='font-size:12px;color:#8b95b5;margin:8px 0 6px;'>💡 You might also want to ask:</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, fq in enumerate(st.session_state.followups):
            with cols[i]:
                st.markdown('<div class="fq-btn">', unsafe_allow_html=True)
                if st.button(fq, key=f"fq_{i}", use_container_width=True):
                    st.session_state.pending_question = fq
                    st.session_state.followups = []
                st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    typed = st.chat_input("Ask anything about your documents...")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    elif typed:
        question = typed
    else:
        question = None

    # Handle comparison
    if st.session_state.compare_request:
        req = st.session_state.compare_request
        st.session_state.compare_request = None
        with st.chat_message("user"):
            st.markdown(f"⚖️ **Compare:** {req['question']} — `{req['file1']}` vs `{req['file2']}`")
        with st.chat_message("assistant"):
            with st.spinner("⚖️ Comparing documents..."):
                comparison, ans1, ans2 = compare_documents(req['question'], req['file1'], req['file2'])
            placeholder = st.empty()
            displayed = ""
            for char in comparison:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.005)
            placeholder.markdown(displayed)
            with st.expander("📄 Raw answers from each document"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📄 {req['file1']}**")
                    st.markdown(ans1)
                with col2:
                    st.markdown(f"**📄 {req['file2']}**")
                    st.markdown(ans2)
        st.session_state.messages.append({"role": "user", "content": f"⚖️ Compare: {req['question']}"})
        st.session_state.messages.append({"role": "assistant", "content": comparison, "sources": []})
        save_analytics(f"COMPARE: {req['question']}", f"{req['file1']} vs {req['file2']}", 0, is_comparison=True)
        st.session_state.followups = suggest_followups(req['question'], comparison)
        st.rerun()

    # Process question
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                answer, source_docs = process_question(
                    question,
                    st.session_state.get("doc_filter", "📂 All Documents")
                )
            placeholder = st.empty()
            displayed = ""
            for char in answer:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.008)
            placeholder.markdown(displayed)

            not_found = "could not find" in answer.lower()
            sources_data = []
            if source_docs and not not_found:
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(source_docs[:3]):
                        src_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        src_page = doc.metadata.get('page', 'Unknown')
                        preview = doc.page_content[:200] + "..."
                        sources_data.append({"file": src_file, "page": src_page, "preview": preview})
                        st.markdown(f"**Source {i+1}** · `{src_file}` · Page `{src_page}`")
                        st.caption(preview)
                        if i < min(2, len(source_docs)-1):
                            st.divider()

                # Hallucination check
                with st.spinner("🔍 Verifying accuracy..."):
                    score, verdict, reason = detect_hallucination(answer, source_docs)
                if score is not None:
                    show_hallucination_badge(score, verdict, reason)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_data})
        if not not_found:
            st.session_state.followups = suggest_followups(question, answer)
        else:
            st.session_state.followups = []
        st.rerun()

# ════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 Dashboard":

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1f2e,#2d3561);border:1px solid #3d4fd1;
    border-radius:16px;padding:24px 32px;margin-bottom:24px;'>
        <p style='font-size:28px;font-weight:700;color:#fff;margin:0;'>📊 Analytics Dashboard</p>
        <p style='font-size:14px;color:#8b95b5;margin:4px 0 0;'>Visual insights about your usage</p>
    </div>
    """, unsafe_allow_html=True)

    analytics = load_analytics()
    files = get_ingested_files()
    rt_list = analytics.get("response_times", [])
    docs_queried = analytics.get("docs_queried", [])
    questions_list = analytics.get("questions", [])
    h_scores = analytics.get("hallucination_scores", [])
    avg_rt = round(sum(rt_list)/len(rt_list), 1) if rt_list else 0
    avg_trust = round(sum(h_scores)/len(h_scores), 1) if h_scores else 0

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, num, label in [
        (c1, len(files), "Documents"),
        (c2, analytics.get("total_questions", 0), "Questions"),
        (c3, analytics.get("comparison_count", 0), "Comparisons"),
        (c4, f"{avg_rt}s", "Avg Response"),
        (c5, f"{avg_trust}/100", "Avg Trust"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-num'>{num}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>📄 Questions per Document</div>", unsafe_allow_html=True)
        if docs_queried:
            counts = Counter(docs_queried)
            fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                        labels={"x": "Document", "y": "Questions"},
                        color=list(counts.values()), color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                             font_color="#8b95b5", showlegend=False,
                             coloraxis_showscale=False, margin=dict(l=20,r=20,t=20,b=60))
            fig.update_xaxes(tickangle=-20, tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ask questions to see this chart!")

    with col_right:
        st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>⏱️ Response Time Trend</div>", unsafe_allow_html=True)
        if len(rt_list) > 1:
            fig2 = px.line(x=list(range(1, len(rt_list)+1)), y=rt_list,
                          labels={"x": "Question No.", "y": "Time (s)"}, markers=True)
            fig2.update_traces(line_color="#6c7ee1", marker_color="#6c7ee1")
            fig2.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                              font_color="#8b95b5", margin=dict(l=20,r=20,t=20,b=40))
            fig2.add_hline(y=avg_rt, line_dash="dash", line_color="#f0997b",
                          annotation_text=f"Avg:{avg_rt}s")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Ask more questions to see trends!")

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>☁️ Topics Word Cloud</div>", unsafe_allow_html=True)
        if questions_list:
            all_text = " ".join(questions_list)
            stop_words = {"what","how","why","when","where","is","are","the","a","an",
                         "in","of","to","and","or","this","that","was","were","be",
                         "been","about","tell","me","my","document","from","which",
                         "do","does","did","can","could","would","should","i"}
            wc = WordCloud(width=600, height=300, background_color="#1a1f2e",
                          colormap="cool", stopwords=stop_words, max_words=50).generate(all_text)
            fig_wc, ax = plt.subplots(figsize=(6,3))
            fig_wc.patch.set_facecolor("#1a1f2e")
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
            plt.close()
        else:
            st.info("Ask questions to generate word cloud!")

    with col_right2:
        st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>🍩 Document Coverage</div>", unsafe_allow_html=True)
        if docs_queried:
            counts = Counter(docs_queried)
            fig3 = px.pie(values=list(counts.values()), names=list(counts.keys()),
                         hole=0.5, color_discrete_sequence=["#6c7ee1","#5dcaa5","#f0997b","#fac775"])
            fig3.update_layout(paper_bgcolor="#1a1f2e", font_color="#8b95b5",
                              legend=dict(font=dict(color="#8b95b5")),
                              margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Ask questions to see coverage!")

    # Trust score trend
    if h_scores:
        st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin:16px 0 8px;'>🔍 Answer Trust Score Trend</div>", unsafe_allow_html=True)
        fig_h = px.line(x=list(range(1, len(h_scores)+1)), y=h_scores,
                       labels={"x": "Question No.", "y": "Trust Score"}, markers=True)
        fig_h.update_traces(line_color="#5dcaa5", marker_color="#5dcaa5")
        fig_h.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                            font_color="#8b95b5", margin=dict(l=20,r=20,t=20,b=40),
                            yaxis=dict(range=[0,100]))
        fig_h.add_hline(y=80, line_dash="dash", line_color="#5dcaa5",
                       annotation_text="Trustworthy")
        fig_h.add_hline(y=50, line_dash="dash", line_color="#fac775",
                       annotation_text="Warning")
        st.plotly_chart(fig_h, use_container_width=True)

    st.divider()

    # Recent questions
    st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>🕐 Recent Questions</div>", unsafe_allow_html=True)
    if questions_list:
        recent = list(zip(questions_list[-10:][::-1],
                         docs_queried[-10:][::-1],
                         rt_list[-10:][::-1]))
        h1, h2, h3 = st.columns([4,2,1])
        h1.markdown("<div style='font-size:11px;color:#8b95b5;font-weight:600;'>QUESTION</div>", unsafe_allow_html=True)
        h2.markdown("<div style='font-size:11px;color:#8b95b5;font-weight:600;'>DOCUMENT</div>", unsafe_allow_html=True)
        h3.markdown("<div style='font-size:11px;color:#8b95b5;font-weight:600;'>TIME</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:0.5px solid #1f2537;margin:4px 0 8px;'>", unsafe_allow_html=True)
        for q, doc, rt in recent:
            c1, c2, c3 = st.columns([4,2,1])
            c1.markdown(f"<div style='font-size:13px;color:#c5cae9;padding:5px 0;'>❓ {q[:60]}{'...' if len(q)>60 else ''}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div style='font-size:12px;color:#8b95b5;padding:5px 0;'>📄 {doc[:20]}</div>", unsafe_allow_html=True)
            color = "#5dcaa5" if rt < 5 else "#fac775" if rt < 10 else "#f0997b"
            c3.markdown(f"<div style='font-size:12px;color:{color};padding:5px 0;'>{rt}s</div>", unsafe_allow_html=True)
            st.markdown("<hr style='border:0.5px solid #1f2537;margin:2px 0;'>", unsafe_allow_html=True)
    else:
        st.info("No questions asked yet!")

    st.divider()

    col_ref, col_rst = st.columns(2)
    with col_ref:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_rst:
        if st.button("🗑️ Reset Analytics", use_container_width=True):
            if os.path.exists(analytics_path):
                os.remove(analytics_path)
            st.success("✅ Analytics cleared!")
            st.rerun()

    st.markdown("<div style='text-align:center;font-size:11px;color:#3d4555;margin-top:20px;'>Built by Komal Sawant · MSc Data Science</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE: ABLATION STUDY
# ════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🔬 Ablation":

    import sys
    sys.path.insert(0, current_dir)
    from ablation import run_ablation_study, results_path, CHUNK_SIZES, TEST_QUESTIONS

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1f2e,#2d3561);border:1px solid #3d4fd1;
    border-radius:16px;padding:24px 32px;margin-bottom:24px;'>
        <p style='font-size:28px;font-weight:700;color:#fff;margin:0;'>🔬 Chunk Size Ablation Study</p>
        <p style='font-size:14px;color:#8b95b5;margin:4px 0 0;'>
        Compare RAG performance across chunk sizes 250 · 500 · 1000</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#1a1f2e;border:1px solid #2a2f45;border-radius:12px;
    padding:16px 20px;margin-bottom:20px;'>
        <div style='font-size:14px;font-weight:600;color:#fff;margin-bottom:8px;'>
        📌 What this study does</div>
        <div style='font-size:13px;color:#8b95b5;line-height:1.8;'>
        • Tests 3 chunk sizes: <b style='color:#6c7ee1;'>250</b>,
          <b style='color:#6c7ee1;'>500</b>,
          <b style='color:#6c7ee1;'>1000</b> characters<br>
        • Runs 5 standard questions on each chunk size<br>
        • Scores each answer on Relevance, Faithfulness, Completeness<br>
        • Declares a winner with evidence — great for dissertation! 📄
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 View test questions"):
        for i, q in enumerate(TEST_QUESTIONS):
            st.markdown(f"**{i+1}.** {q}")

    st.divider()

    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_study = st.button("🚀 Run Ablation Study", use_container_width=True)
    with col_info:
        st.markdown("<div style='font-size:13px;color:#8b95b5;padding:8px 0;'>⏱️ Takes 5-8 minutes · Results saved automatically</div>", unsafe_allow_html=True)

    if run_study:
        if not get_ingested_files():
            st.error("❌ Upload documents first!")
        else:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total_steps = len(CHUNK_SIZES) * len(TEST_QUESTIONS)
            step = [0]
            def update_progress(msg):
                progress_text.markdown(f"<div style='font-size:13px;color:#8b95b5;'>{msg}</div>", unsafe_allow_html=True)
                step[0] += 1
                progress_bar.progress(min(step[0]/total_steps, 1.0))
            with st.spinner("🔬 Running ablation study..."):
                results = run_ablation_study(progress_callback=update_progress)
            progress_bar.progress(1.0)
            progress_text.markdown("<div style='font-size:13px;color:#5dcaa5;'>✅ Study complete!</div>", unsafe_allow_html=True)
            st.rerun()

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

        st.markdown("<div style='font-size:18px;font-weight:600;color:#fff;margin:16px 0 12px;'>📊 Results</div>", unsafe_allow_html=True)

        cols = st.columns(3)
        winner_size = max(results.keys(), key=lambda k: results[k]["avg_overall"])

        for i, (chunk_size, data) in enumerate(results.items()):
            with cols[i]:
                is_winner = chunk_size == winner_size
                border_color = "#5dcaa5" if is_winner else "#2a2f45"
                st.markdown(f"""
                <div style='background:#1a1f2e;border:2px solid {border_color};
                border-radius:12px;padding:16px;text-align:center;'>
                    <div style='font-size:22px;font-weight:700;color:#6c7ee1;'>Chunk {chunk_size}</div>
                    <div style='font-size:11px;color:#8b95b5;margin-bottom:12px;'>
                    {data["num_chunks"]} chunks · overlap {data["overlap"]}</div>
                    <div style='font-size:28px;font-weight:700;color:{"#5dcaa5" if is_winner else "#fff"};'>
                    {data["avg_overall"]}/5</div>
                    <div style='font-size:11px;color:#8b95b5;'>Overall Score</div>
                    {"<div style='margin-top:8px;font-size:12px;font-weight:600;color:#5dcaa5;'>🏆 BEST</div>" if is_winner else ""}
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>📊 Overall Score</div>", unsafe_allow_html=True)
            sizes = [int(k) for k in results.keys()]
            overall_scores = [results[k]["avg_overall"] for k in results.keys()]
            colors = ["#5dcaa5" if str(s) == winner_size else "#6c7ee1" for s in sizes]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"Chunk {s}" for s in sizes], y=overall_scores,
                                marker_color=colors, text=overall_scores, textposition='auto'))
            fig.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                             font_color="#8b95b5", showlegend=False,
                             yaxis=dict(range=[0,5]), margin=dict(l=20,r=20,t=20,b=40))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("<div style='font-size:15px;font-weight:600;color:#fff;margin-bottom:8px;'>🎯 Metrics Comparison</div>", unsafe_allow_html=True)
            fig2 = go.Figure()
            colors_list = ["#6c7ee1","#5dcaa5","#f0997b"]
            for i, (size_key, data) in enumerate(results.items()):
                fig2.add_trace(go.Bar(name=f"Chunk {size_key}",
                                     x=["Relevance","Faithfulness","Completeness"],
                                     y=[data["avg_relevance"],data["avg_faithfulness"],data["avg_completeness"]],
                                     marker_color=colors_list[i]))
            fig2.update_layout(barmode='group', paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
                              font_color="#8b95b5", yaxis=dict(range=[0,5]),
                              margin=dict(l=20,r=20,t=20,b=40))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        winner_data = results[winner_size]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0F2A1F,#0F3320);
        border:2px solid #5dcaa5;border-radius:16px;padding:24px;text-align:center;'>
            <div style='font-size:36px;margin-bottom:8px;'>🏆</div>
            <div style='font-size:22px;font-weight:700;color:#5dcaa5;margin-bottom:8px;'>
            Best Chunk Size: {winner_size} characters</div>
            <div style='font-size:14px;color:#8b95b5;line-height:1.8;'>
            Overall: <b style='color:#5dcaa5;'>{winner_data["avg_overall"]}/5</b> ·
            Relevance: <b style='color:#fff;'>{winner_data["avg_relevance"]}/5</b> ·
            Faithfulness: <b style='color:#fff;'>{winner_data["avg_faithfulness"]}/5</b> ·
            Completeness: <b style='color:#fff;'>{winner_data["avg_completeness"]}/5</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.download_button(
            label="⬇️ Download Results JSON",
            data=json.dumps(results, indent=2),
            file_name="ablation_results.json",
            mime="application/json"
        )
    else:
        st.info("👆 Click 'Run Ablation Study' to start!")