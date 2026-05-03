# evaluate.py - Evaluation system for RAG Analyst Co-Pilot

import os
import csv
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore_path = os.path.join(current_dir, "..", "vectorstore")
results_path = os.path.join(current_dir, "..", "evaluation_results.csv")

# ── Load RAG chain ─────────────────────────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_chain():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    You are a helpful analyst assistant.
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say "I could not find this information in the documents."
    Always mention which part of the document you used.

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
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── Scoring functions ──────────────────────────────────────────────────────

def score_relevance(question, answer, llm):
    """Ask Groq to score how relevant the answer is to the question."""
    prompt = f"""
    Question: {question}
    Answer: {answer}

    Rate how relevant this answer is to the question on a scale of 1 to 5.
    1 = completely irrelevant
    3 = partially relevant
    5 = perfectly relevant

    Reply with ONLY a single number between 1 and 5. Nothing else.
    """
    result = llm.invoke(prompt)
    try:
        score = float(result.content.strip())
        return min(max(score, 1), 5)
    except:
        return 3.0

def score_faithfulness(answer, context_docs, llm):
    """Check if the answer is grounded in the retrieved documents."""
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = f"""
    Context from documents:
    {context}

    Answer given:
    {answer}

    Does this answer only use information from the context above?
    Rate faithfulness on a scale of 1 to 5.
    1 = answer contains made-up information
    3 = answer is partially grounded
    5 = answer is completely grounded in the context

    Reply with ONLY a single number between 1 and 5. Nothing else.
    """
    result = llm.invoke(prompt)
    try:
        score = float(result.content.strip())
        return min(max(score, 1), 5)
    except:
        return 3.0

def score_completeness(question, answer, llm):
    """Check if the answer fully addresses the question."""
    prompt = f"""
    Question: {question}
    Answer: {answer}

    How completely does this answer address the question?
    Rate completeness on a scale of 1 to 5.
    1 = does not address the question at all
    3 = partially addresses the question
    5 = fully and completely addresses the question

    Reply with ONLY a single number between 1 and 5. Nothing else.
    """
    result = llm.invoke(prompt)
    try:
        score = float(result.content.strip())
        return min(max(score, 1), 5)
    except:
        return 3.0

# ── Test questions ─────────────────────────────────────────────────────────
# Modify these questions based on YOUR document content!

TEST_QUESTIONS = [
    "What is this document about?",
    "Who is the author of this document?",
    "What are the main topics covered?",
    "What regulations or standards are mentioned?",
    "Summarize the key points of this document.",
    "What conclusions or recommendations are made?",
    "What data or evidence is presented?",
    "Are there any definitions provided in the document?",
]

# ── Run evaluation ─────────────────────────────────────────────────────────

def run_evaluation():
    print("🚀 Starting RAG Evaluation...")
    print("=" * 60)

    chain, retriever = load_chain()

    # Use a separate LLM for scoring
    scorer_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0  # deterministic scoring
    )

    results = []

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n📝 Question {i+1}/{len(TEST_QUESTIONS)}: {question}")

        # Get answer
        start_time = time.time()
        answer = chain.invoke(question)
        response_time = round(time.time() - start_time, 2)

        # Get source docs
        source_docs = retriever.invoke(question)
        sources_found = len(source_docs)

        # Check if answer was found
        not_found = "could not find" in answer.lower()

        # Score the answer
        print("   ⏳ Scoring...")
        relevance = score_relevance(question, answer, scorer_llm)
        faithfulness = score_faithfulness(answer, source_docs, scorer_llm)
        completeness = score_completeness(question, answer, scorer_llm)
        overall = round((relevance + faithfulness + completeness) / 3, 2)

        result = {
            "question": question,
            "answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "sources_found": sources_found,
            "response_time_sec": response_time,
            "relevance_score": relevance,
            "faithfulness_score": faithfulness,
            "completeness_score": completeness,
            "overall_score": overall,
            "answer_found": not not_found
        }
        results.append(result)

        print(f"   ✅ Relevance: {relevance}/5 | Faithfulness: {faithfulness}/5 | Completeness: {completeness}/5 | Overall: {overall}/5")
        print(f"   ⏱️  Response time: {response_time}s | Sources: {sources_found}")

        # Small delay to avoid rate limiting
        time.sleep(1)

    # ── Save results to CSV ────────────────────────────────────────────────
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    avg_relevance = round(sum(r["relevance_score"] for r in results) / len(results), 2)
    avg_faithfulness = round(sum(r["faithfulness_score"] for r in results) / len(results), 2)
    avg_completeness = round(sum(r["completeness_score"] for r in results) / len(results), 2)
    avg_overall = round(sum(r["overall_score"] for r in results) / len(results), 2)
    avg_response_time = round(sum(r["response_time_sec"] for r in results) / len(results), 2)
    answered = sum(1 for r in results if r["answer_found"])

    print(f"  📌 Total questions tested : {len(results)}")
    print(f"  ✅ Questions answered      : {answered}/{len(results)}")
    print(f"  ⭐ Avg Relevance           : {avg_relevance}/5")
    print(f"  🎯 Avg Faithfulness        : {avg_faithfulness}/5")
    print(f"  📝 Avg Completeness        : {avg_completeness}/5")
    print(f"  🏆 Avg Overall Score       : {avg_overall}/5")
    print(f"  ⏱️  Avg Response Time       : {avg_response_time}s")
    print(f"\n  📁 Full results saved to: evaluation_results.csv")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_evaluation()