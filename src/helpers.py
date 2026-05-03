import os
import json
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")
DATA_FOLDER = os.path.join(ROOT_DIR, "data")
ANALYTICS_PATH = os.path.join(ROOT_DIR, "analytics.json")

os.makedirs(DATA_FOLDER, exist_ok=True)

def get_stored_path(filename):
    return os.path.join(ROOT_DIR, "src", "..", "data", filename)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_ingested_files():
    if not os.path.exists(DATA_FOLDER):
        return []
    return [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

def get_chunk_count():
    try:
        vs = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=get_embeddings())
        return vs._collection.count()
    except:
        return 0

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

def ingest_pdf(uploaded_file):
    save_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(save_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    return len(documents), len(chunks)

def delete_document(filename):
    try:
        embeddings = get_embeddings()
        vs = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
        stored_path = get_stored_path(filename)
        all_data = vs.get()
        ids_to_delete = [
            all_data['ids'][i]
            for i, meta in enumerate(all_data['metadatas'])
            if meta.get('source') == stored_path
        ]
        if ids_to_delete:
            vs.delete(ids=ids_to_delete)
        file_path = os.path.join(DATA_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except:
        return False

def build_chain(retriever):
    llm = get_llm()
    template = """
You are a professional analyst assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I could not find this information in the documents."

FORMATTING RULES:
- Multiple points → each bullet on its own line with blank line between:

- Point one

- Point two

- Steps → numbered list with blank line between each
- Single fact → one clear sentence
- End with: *Source: filename, page X*
- Never write bullets on same line

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate.from_template(template)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

def get_retriever_for_file(filename):
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
    stored_path = get_stored_path(filename)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "filter": {"source": stored_path}}
    )

def answer_across_all_files(question, files):
    all_answers = []
    for filename in files:
        retriever = get_retriever_for_file(filename)
        chain = build_chain(retriever)
        try:
            ans = chain.invoke(question)
            if "could not find" not in ans.lower():
                all_answers.append(f"**📄 From {filename}:**\n\n{ans}")
        except:
            pass
    if not all_answers:
        return "I could not find this information across any of the documents."
    return "\n\n---\n\n".join(all_answers)

def suggest_followups(question, answer):
    llm = get_llm()
    prompt = f"""Suggest exactly 3 smart follow-up questions based on this Q&A.

Question: {question}
Answer: {answer}

Rules:
- Each under 12 words
- One per line
- No numbering, no bullets, no extra text
"""
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

    compare_prompt = f"""
You are an expert analyst. Compare two documents on the topic below.

Topic: {question}

Document 1 ({file1}): {ans1}
Document 2 ({file2}): {ans2}

Use this EXACT format:

**Overview**
Brief summary of how documents relate to this topic.

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

def save_analytics(question, doc_label, response_time, is_comparison=False):
    if os.path.exists(ANALYTICS_PATH):
        with open(ANALYTICS_PATH, "r") as f:
            data = json.load(f)
    else:
        data = {
            "questions": [], "response_times": [],
            "docs_queried": [], "comparison_count": 0,
            "total_questions": 0
        }
    data["questions"].append(question)
    data["response_times"].append(round(response_time, 2))
    data["docs_queried"].append(doc_label)
    data["total_questions"] += 1
    if is_comparison:
        data["comparison_count"] += 1
    with open(ANALYTICS_PATH, "w") as f:
        json.dump(data, f)

def load_analytics():
    if os.path.exists(ANALYTICS_PATH):
        with open(ANALYTICS_PATH, "r") as f:
            return json.load(f)
    return {
        "questions": [], "response_times": [],
        "docs_queried": [], "comparison_count": 0,
        "total_questions": 0
    }

def process_question(question, doc_filter):
    start = time.time()
    if doc_filter == "📂 All Documents":
        answer = answer_across_all_files(question, get_ingested_files())
        source_docs = []
        doc_label = "All Documents"
    else:
        retriever = get_retriever_for_file(doc_filter)
        chain = build_chain(retriever)
        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)
        doc_label = doc_filter
    elapsed = round(time.time() - start, 2)
    save_analytics(question, doc_label, elapsed)
    return answer, source_docs