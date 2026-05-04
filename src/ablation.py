# ablation.py - Chunk Size Ablation Study for RAG System

import os
import json
import time
import shutil
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
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "..", "data")
results_path = os.path.join(current_dir, "..", "ablation_results.json")

CHUNK_SIZES = [250, 500, 1000]
CHUNK_OVERLAPS = {250: 25, 500: 50, 1000: 100}

TEST_QUESTIONS = [
    "What is this document about?",
    "What are the main topics covered?",
    "What are the key findings or conclusions?",
    "What definitions are provided?",
    "What recommendations are mentioned?"
]

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

def build_temp_vectorstore(chunk_size, overlap, vs_path):
    embeddings = get_embeddings()
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
    vectorstore = FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )
    vectorstore.save_local(vs_path)
    return vectorstore, len(all_chunks)

def score_answer(question, answer, context, llm):
    """Score answer on relevance, faithfulness and completeness."""
    prompt = f"""Rate this Q&A on three metrics. Reply ONLY in this exact format:
RELEVANCE: [1-5]
FAITHFULNESS: [1-5]
COMPLETENESS: [1-5]

Question: {question}
Context: {context[:500]}
Answer: {answer[:300]}

Just the three lines above. Nothing else."""
    
    result = llm.invoke(prompt)
    lines = result.content.strip().split('\n')
    scores = {"relevance": 3, "faithfulness": 3, "completeness": 3}
    
    for line in lines:
        if "RELEVANCE:" in line:
            try:
                scores["relevance"] = float(line.split(":")[1].strip())
            except:
                pass
        elif "FAITHFULNESS:" in line:
            try:
                scores["faithfulness"] = float(line.split(":")[1].strip())
            except:
                pass
        elif "COMPLETENESS:" in line:
            try:
                scores["completeness"] = float(line.split(":")[1].strip())
            except:
                pass
    
    return scores

def run_ablation_study(progress_callback=None):
    """Run the full ablation study across all chunk sizes."""
    llm = get_llm()
    all_results = {}
    
    for chunk_size in CHUNK_SIZES:
        overlap = CHUNK_OVERLAPS[chunk_size]
        vs_path = os.path.join(current_dir, "..", f"ablation_vs_{chunk_size}")
        
        if progress_callback:
            progress_callback(f"⚙️ Building vectorstore with chunk size {chunk_size}...")
        
        # Build temp vectorstore
        if os.path.exists(vs_path):
            shutil.rmtree(vs_path, ignore_errors=True)
        
        vectorstore, num_chunks = build_temp_vectorstore(chunk_size, overlap, vs_path)
        
        chunk_results = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": num_chunks,
            "questions": [],
            "avg_relevance": 0,
            "avg_faithfulness": 0,
            "avg_completeness": 0,
            "avg_response_time": 0,
            "avg_overall": 0
        }
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        template = """Answer the question using only the context below.
If not found, say "I could not find this information."

Context: {context}
Question: {question}
Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        
        total_rel = total_faith = total_comp = total_time = 0
        
        for i, question in enumerate(TEST_QUESTIONS):
            if progress_callback:
                progress_callback(f"📝 Chunk {chunk_size} — Question {i+1}/{len(TEST_QUESTIONS)}: {question[:40]}...")
            
            start = time.time()
            answer = chain.invoke(question)
            elapsed = round(time.time() - start, 2)
            
            source_docs = retriever.invoke(question)
            context = "\n".join([d.page_content[:200] for d in source_docs])
            
            scores = score_answer(question, answer, context, llm)
            overall = round((scores["relevance"] + scores["faithfulness"] + scores["completeness"]) / 3, 2)
            
            chunk_results["questions"].append({
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "response_time": elapsed,
                "relevance": scores["relevance"],
                "faithfulness": scores["faithfulness"],
                "completeness": scores["completeness"],
                "overall": overall
            })
            
            total_rel += scores["relevance"]
            total_faith += scores["faithfulness"]
            total_comp += scores["completeness"]
            total_time += elapsed
            
            time.sleep(0.5)
        
        n = len(TEST_QUESTIONS)
        chunk_results["avg_relevance"] = round(total_rel / n, 2)
        chunk_results["avg_faithfulness"] = round(total_faith / n, 2)
        chunk_results["avg_completeness"] = round(total_comp / n, 2)
        chunk_results["avg_response_time"] = round(total_time / n, 2)
        chunk_results["avg_overall"] = round(
            (chunk_results["avg_relevance"] + 
             chunk_results["avg_faithfulness"] + 
             chunk_results["avg_completeness"]) / 3, 2
        )
        
        all_results[str(chunk_size)] = chunk_results
        
        # Cleanup temp vectorstore - close connection first
        try:
            del vectorstore
            del retriever
            import gc
            gc.collect()
            time.sleep(1)
            if os.path.exists(vs_path):
                shutil.rmtree(vs_path, ignore_errors=True)
        except:
            pass  # skip cleanup if still locked, will be cleaned next run
    
    # Save results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results