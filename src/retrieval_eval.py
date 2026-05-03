# retrieval_eval.py - Retrieval Precision, Recall and MRR Evaluation

import os
import json
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(current_dir, "..", "vectorstore")
data_folder = os.path.join(current_dir, "..", "data")
results_path = os.path.join(current_dir, "..", "retrieval_eval_results.json")

# ── Test set with ground truth keywords ──────────────────────────────────
# Each question has keywords that MUST appear in retrieved chunks
# to count as a relevant chunk

TEST_SET = [
    {
        "question": "What is data governance?",
        "relevant_keywords": ["governance", "data governance", "policy", "compliance", "management"]
    },
    {
        "question": "What regulations are mentioned?",
        "relevant_keywords": ["GDPR", "regulation", "compliance", "law", "act", "CCPA"]
    },
    {
        "question": "What is data visualization?",
        "relevant_keywords": ["visualization", "chart", "graph", "visual", "display", "plot"]
    },
    {
        "question": "What security measures are discussed?",
        "relevant_keywords": ["security", "encryption", "protection", "privacy", "secure", "risk"]
    },
    {
        "question": "What are the main topics of this document?",
        "relevant_keywords": ["data", "document", "topic", "subject", "chapter", "section"]
    }
]

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

def is_chunk_relevant(chunk_text, keywords):
    """Check if chunk contains any relevant keywords."""
    chunk_lower = chunk_text.lower()
    return any(kw.lower() in chunk_lower for kw in keywords)

def calculate_precision(retrieved_chunks, keywords):
    """Precision = relevant retrieved / total retrieved."""
    if not retrieved_chunks:
        return 0.0
    relevant = sum(1 for c in retrieved_chunks if is_chunk_relevant(c.page_content, keywords))
    return round(relevant / len(retrieved_chunks), 3)

def calculate_recall(retrieved_chunks, vectorstore, keywords, total_sample=50):
    """Recall = relevant retrieved / total relevant in DB."""
    # Sample chunks from DB to estimate total relevant
    all_data = vectorstore.get()
    all_texts = [doc for doc in all_data['documents'][:total_sample]]
    total_relevant = sum(1 for t in all_texts if any(kw.lower() in t.lower() for kw in keywords))
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for c in retrieved_chunks if is_chunk_relevant(c.page_content, keywords))
    return round(min(retrieved_relevant / total_relevant, 1.0), 3)

def calculate_mrr(retrieved_chunks, keywords):
    """MRR = 1/rank of first relevant chunk."""
    for i, chunk in enumerate(retrieved_chunks):
        if is_chunk_relevant(chunk.page_content, keywords):
            return round(1 / (i + 1), 3)
    return 0.0

def calculate_ndcg(retrieved_chunks, keywords):
    """Simplified nDCG score."""
    import math
    dcg = 0
    for i, chunk in enumerate(retrieved_chunks):
        rel = 1 if is_chunk_relevant(chunk.page_content, keywords) else 0
        dcg += rel / math.log2(i + 2)
    ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(3, len(retrieved_chunks))))
    if ideal_dcg == 0:
        return 0.0
    return round(dcg / ideal_dcg, 3)

def run_retrieval_evaluation(k_values=[1, 3, 5]):
    """Run full retrieval evaluation across different k values."""
    print("🚀 Starting Retrieval Evaluation...")
    
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )
    
    all_results = {}
    
    for k in k_values:
        print(f"\n📊 Evaluating with k={k}...")
        k_results = {
            "k": k,
            "questions": [],
            "avg_precision": 0,
            "avg_recall": 0,
            "avg_mrr": 0,
            "avg_ndcg": 0,
            "avg_response_time": 0
        }
        
        total_prec = total_rec = total_mrr = total_ndcg = total_time = 0
        
        for test in TEST_SET:
            question = test["question"]
            keywords = test["relevant_keywords"]
            
            print(f"  ❓ {question}")
            
            start = time.time()
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k*3}
            )
            retrieved = retriever.invoke(question)
            elapsed = round(time.time() - start, 3)
            
            precision = calculate_precision(retrieved, keywords)
            recall = calculate_recall(retrieved, vectorstore, keywords)
            mrr = calculate_mrr(retrieved, keywords)
            ndcg = calculate_ndcg(retrieved, keywords)
            
            q_result = {
                "question": question,
                "keywords": keywords,
                "chunks_retrieved": len(retrieved),
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "response_time": elapsed,
                "retrieved_previews": [c.page_content[:100] for c in retrieved]
            }
            k_results["questions"].append(q_result)
            
            total_prec += precision
            total_rec += recall
            total_mrr += mrr
            total_ndcg += ndcg
            total_time += elapsed
            
            print(f"    Precision: {precision} | Recall: {recall} | MRR: {mrr} | nDCG: {ndcg}")
        
        n = len(TEST_SET)
        k_results["avg_precision"] = round(total_prec / n, 3)
        k_results["avg_recall"] = round(total_rec / n, 3)
        k_results["avg_mrr"] = round(total_mrr / n, 3)
        k_results["avg_ndcg"] = round(total_ndcg / n, 3)
        k_results["avg_response_time"] = round(total_time / n, 3)
        
        all_results[f"k_{k}"] = k_results
        
        print(f"\n  📈 k={k} Summary:")
        print(f"  Avg Precision: {k_results['avg_precision']}")
        print(f"  Avg Recall:    {k_results['avg_recall']}")
        print(f"  Avg MRR:       {k_results['avg_mrr']}")
        print(f"  Avg nDCG:      {k_results['avg_ndcg']}")
    
    # Save results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: retrieval_eval_results.json")
    return all_results

if __name__ == "__main__":
    run_retrieval_evaluation()