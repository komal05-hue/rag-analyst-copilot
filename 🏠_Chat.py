import os
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from helpers import (
    get_ingested_files, get_chunk_count, ingest_pdf, delete_document,
    get_retriever_for_file, build_chain, answer_across_all_files,
    suggest_followups, compare_documents, process_question, save_analytics,
    DATA_FOLDER
)

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
  .fq-btn button:hover { background: #2d3561 !important; color: #ffffff !important; }
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
for key, val in [
    ("messages", []), ("pending_question", None),
    ("uploader_key", 0), ("followups", []),
    ("compare_request", None), ("doc_filter", "📂 All Documents")
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px;'>
        <div style='font-size:36px;'>🤖</div>
        <div style='font-size:17px;font-weight:700;color:#fff;'>Analyst Co-Pilot</div>
        <div style='font-size:11px;color:#8b95b5;'>Powered by Groq + LangChain</div>
    </div>
    """, unsafe_allow_html=True)

    files = get_ingested_files()
    chunks = get_chunk_count()

    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-card'><div class='stat-number'>{len(files)}</div><div class='stat-label'>Documents</div></div>
        <div class='stat-card'><div class='stat-number'>{chunks}</div><div class='stat-label'>Chunks</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='sidebar-title'>➕ Upload PDF</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop PDF", type=["pdf"], label_visibility="collapsed", key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_file:
        if uploaded_file.name in files:
            st.warning(f"'{uploaded_file.name}' already exists!")
        else:
            if st.button("⚡ Ingest PDF", use_container_width=True):
                with st.spinner("Processing..."):
                    pages, chunks_added = ingest_pdf(uploaded_file)
                st.session_state.uploader_key += 1
                st.success(f"✅ {pages} pages · {chunks_added} chunks!")
                st.rerun()

    st.divider()
    st.markdown("<div class='sidebar-title'>⚖️ Compare Documents</div>", unsafe_allow_html=True)
    if len(files) >= 2:
        comp_doc1 = st.selectbox("Document 1", files, key="comp1")
        comp_doc2 = st.selectbox("Document 2", [f for f in files if f != comp_doc1], key="comp2")
        comp_question = st.text_input("What to compare?", placeholder="e.g. regulations, topics...", key="comp_q")
        if st.button("⚖️ Compare Now", use_container_width=True):
            if comp_question:
                st.session_state.compare_request = {"file1": comp_doc1, "file2": comp_doc2, "question": comp_question}
                st.session_state.followups = []
                st.rerun()
            else:
                st.warning("Enter a comparison topic!")
    else:
        st.markdown("<div style='font-size:12px;color:#8b95b5;'>Upload at least 2 documents to compare</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='sidebar-title'>🔍 Search In</div>", unsafe_allow_html=True)
    filter_options = ["📂 All Documents"] + files
    selected = st.radio("", filter_options, label_visibility="collapsed")
    st.session_state["doc_filter"] = selected

    st.divider()
    st.markdown("<div class='sidebar-title'>📚 Documents</div>", unsafe_allow_html=True)
    if files:
        for f in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<div style='font-size:13px;color:#c5cae9;padding:6px 0;'>📄 {f}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{f}"):
                    with st.spinner("Deleting..."):
                        delete_document(f)
                    st.rerun()
    else:
        st.markdown("<div style='font-size:13px;color:#8b95b5;'>No documents yet!</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.followups = []
        st.session_state.pending_question = None
        st.rerun()

    if st.button("🗑️ Clear All Documents", use_container_width=True):
        import shutil
        from helpers import VECTORSTORE_PATH
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        for f in get_ingested_files():
            os.remove(os.path.join(DATA_FOLDER, f))
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style='margin-top:20px;padding-top:14px;border-top:0.5px solid #1f2537;font-size:11px;color:#3d4555;text-align:center;'>
        Built by Komal Sawant · MSc Data Science
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────
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
    <div style='background:#1a1f2e;border:1px solid #2a2f45;border-radius:14px;padding:20px 24px;margin-bottom:20px;'>
        <div style='font-size:18px;font-weight:600;color:#fff;margin-bottom:8px;'>👋 Hello, Komal!</div>
        <div style='font-size:14px;color:#8b95b5;line-height:1.7;'>
            I've loaded your documents and I'm ready!<br>
            Try: <span style='color:#6c7ee1;'>"What is this document about?"</span>
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
if st.session_state.followups and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
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
        with st.spinner(f"⚖️ Comparing documents..."):
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
            answer, source_docs = process_question(question, st.session_state.get("doc_filter", "📂 All Documents"))
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
                for i, doc in enumerate(source_docs):
                    src_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    src_page = doc.metadata.get('page', 'Unknown')
                    preview = doc.page_content[:200] + "..."
                    sources_data.append({"file": src_file, "page": src_page, "preview": preview})
                    st.markdown(f"**Source {i+1}** · `{src_file}` · Page `{src_page}`")
                    st.caption(preview)
                    if i < len(source_docs) - 1:
                        st.divider()
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_data})
    if not not_found:
        st.session_state.followups = suggest_followups(question, answer)
    else:
        st.session_state.followups = []
    st.rerun()