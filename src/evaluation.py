"""
evaluation.py — Formal evaluation engine for Analyst Co-Pilot
Implements:
  - RAGAS-style metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
  - BLEU score (nltk)
  - ROUGE-1, ROUGE-2, ROUGE-L (rouge-score)
  - MRR (Mean Reciprocal Rank) for retrieval ranking
  - Baseline comparison: RAG vs No-RAG vs BM25
"""

import json
import time
import math
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict


# ── optional heavy imports (graceful fallback if not installed) ───────────────
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

try:
    from rouge_score import rouge_scorer
    ROUGE_OK = True
except ImportError:
    ROUGE_OK = False

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except ImportError:
    BM25_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One question + ground-truth answer pair for evaluation."""
    question: str
    ground_truth: str
    expected_keywords: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Chunks returned by a retriever for one question."""
    question: str
    chunks: List[str]           # retrieved text chunks
    scores: List[float]         # similarity scores (higher = more relevant)
    latency_ms: float = 0.0


@dataclass
class RAGResult:
    """Full RAG answer for one question."""
    question: str
    answer: str
    retrieved_chunks: List[str]
    latency_ms: float = 0.0
    method: str = "RAG"         # "RAG" | "No-RAG" | "BM25"


@dataclass
class MetricScores:
    """All computed metric scores for one sample."""
    question: str
    method: str
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    bleu: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    mrr: float = 0.0
    latency_ms: float = 0.0

    def avg_score(self) -> float:
        scores = [
            self.context_precision, self.context_recall,
            self.faithfulness, self.answer_relevancy,
            self.bleu, self.rouge1, self.rougeL
        ]
        return round(sum(scores) / len(scores), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer (no NLTK dependency needed)."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def _overlap_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Token-level F1 between prediction and reference."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    ref_set  = set(ref_tokens)
    common   = pred_set & ref_set
    if not common:
        return 0.0
    prec = len(common) / len(pred_set)
    rec  = len(common) / len(ref_set)
    return 2 * prec * rec / (prec + rec)


def _keyword_coverage(text: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present in text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS-style metrics (lightweight, no external RAGAS library needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_context_precision(
    question: str,
    retrieved_chunks: List[str],
    ground_truth: str,
    keywords: List[str]
) -> float:
    """
    Context Precision ≈ fraction of retrieved chunks that are relevant.
    A chunk is 'relevant' if it shares meaningful token overlap with
    the ground truth OR contains expected keywords.
    """
    if not retrieved_chunks:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    relevant  = 0
    for chunk in retrieved_chunks:
        chunk_tokens = _tokenize(chunk)
        f1 = _overlap_f1(chunk_tokens, gt_tokens)
        kw = _keyword_coverage(chunk, keywords)
        if f1 > 0.15 or kw > 0.3:
            relevant += 1
    return round(relevant / len(retrieved_chunks), 4)


def compute_context_recall(
    ground_truth: str,
    retrieved_chunks: List[str],
    keywords: List[str]
) -> float:
    """
    Context Recall ≈ how much of the ground truth is covered by retrieved chunks.
    Measures token recall: gt tokens found across all chunks.
    """
    if not retrieved_chunks:
        return 0.0
    gt_tokens = set(_tokenize(ground_truth))
    if not gt_tokens:
        return 0.0
    all_chunk_tokens = set(_tokenize(" ".join(retrieved_chunks)))
    found = gt_tokens & all_chunk_tokens
    base  = len(found) / len(gt_tokens)
    # bonus for keyword coverage
    kw_cov = _keyword_coverage(" ".join(retrieved_chunks), keywords)
    return round(min(1.0, 0.7 * base + 0.3 * kw_cov), 4)


def compute_faithfulness(
    answer: str,
    retrieved_chunks: List[str]
) -> float:
    """
    Faithfulness ≈ how much of the answer is grounded in retrieved chunks.
    Measures: answer sentence tokens found in chunks.
    """
    if not answer or not retrieved_chunks:
        return 0.0
    context = " ".join(retrieved_chunks)
    context_tokens = set(_tokenize(context))
    answer_tokens  = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    grounded = sum(1 for t in answer_tokens if t in context_tokens)
    return round(grounded / len(answer_tokens), 4)


def compute_answer_relevancy(
    question: str,
    answer: str,
    keywords: List[str]
) -> float:
    """
    Answer Relevancy ≈ how relevant the answer is to the question.
    Uses token overlap between question and answer + keyword presence.
    """
    if not answer:
        return 0.0
    q_tokens  = _tokenize(question)
    a_tokens  = _tokenize(answer)
    overlap   = _overlap_f1(a_tokens, q_tokens)
    kw_cov    = _keyword_coverage(answer, keywords)
    # answers that are too short are penalised
    length_penalty = min(1.0, len(a_tokens) / 20)
    return round((0.4 * overlap + 0.4 * kw_cov + 0.2 * length_penalty), 4)


# ─────────────────────────────────────────────────────────────────────────────
# BLEU score
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str) -> float:
    """BLEU-4 with smoothing. Falls back to simple unigram if NLTK missing."""
    if not hypothesis or not reference:
        return 0.0
    if NLTK_OK:
        try:
            ref_tokens  = _tokenize(reference)
            hyp_tokens  = _tokenize(hypothesis)
            smoother    = SmoothingFunction().method1
            score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
            return round(float(score), 4)
        except Exception:
            pass
    # fallback: unigram precision
    ref_tokens = set(_tokenize(reference))
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    hits = sum(1 for t in hyp_tokens if t in ref_tokens)
    return round(hits / len(hyp_tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Returns ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not hypothesis or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if ROUGE_OK:
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return {
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
            }
        except Exception:
            pass

    # fallback: manual n-gram F1
    ref_tok  = _tokenize(reference)
    hyp_tok  = _tokenize(hypothesis)
    r1 = _overlap_f1(hyp_tok, ref_tok)
    # rouge-2
    ref_bg   = _ngrams(ref_tok, 2)
    hyp_bg   = _ngrams(hyp_tok, 2)
    common2  = set(hyp_bg) & set(ref_bg)
    r2 = 2 * len(common2) / (len(hyp_bg) + len(ref_bg)) if (hyp_bg and ref_bg) else 0.0
    # rouge-L (LCS length approximation via DP)
    def lcs_len(a, b):
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    lcs = lcs_len(hyp_tok, ref_tok)
    prec_l = lcs / len(hyp_tok) if hyp_tok else 0
    rec_l  = lcs / len(ref_tok)  if ref_tok  else 0
    rL = 2*prec_l*rec_l/(prec_l+rec_l) if (prec_l+rec_l) > 0 else 0.0
    return {"rouge1": round(r1, 4), "rouge2": round(r2, 4), "rougeL": round(rL, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# MRR (Mean Reciprocal Rank)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mrr(
    retrieved_chunks: List[str],
    ground_truth: str,
    keywords: List[str],
    relevance_threshold: float = 0.15
) -> float:
    """
    MRR = 1/rank of first relevant chunk.
    A chunk is relevant if token overlap with ground truth > threshold
    OR keyword coverage > 30%.
    """
    gt_tokens = set(_tokenize(ground_truth))
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_tokens = _tokenize(chunk)
        overlap = len(set(chunk_tokens) & gt_tokens) / len(gt_tokens) if gt_tokens else 0
        kw = _keyword_coverage(chunk, keywords)
        if overlap > relevance_threshold or kw > 0.3:
            return round(1.0 / rank, 4)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full metrics computation for one sample
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    rag_result: RAGResult,
    sample: EvalSample
) -> MetricScores:
    """Compute every metric for one (question, answer, chunks) triple."""
    rouge = compute_rouge(rag_result.answer, sample.ground_truth)
    return MetricScores(
        question           = sample.question,
        method             = rag_result.method,
        context_precision  = compute_context_precision(
                                 sample.question,
                                 rag_result.retrieved_chunks,
                                 sample.ground_truth,
                                 sample.expected_keywords),
        context_recall     = compute_context_recall(
                                 sample.ground_truth,
                                 rag_result.retrieved_chunks,
                                 sample.expected_keywords),
        faithfulness       = compute_faithfulness(
                                 rag_result.answer,
                                 rag_result.retrieved_chunks),
        answer_relevancy   = compute_answer_relevancy(
                                 sample.question,
                                 rag_result.answer,
                                 sample.expected_keywords),
        bleu               = compute_bleu(rag_result.answer, sample.ground_truth),
        rouge1             = rouge["rouge1"],
        rouge2             = rouge["rouge2"],
        rougeL             = rouge["rougeL"],
        mrr                = compute_mrr(
                                 rag_result.retrieved_chunks,
                                 sample.ground_truth,
                                 sample.expected_keywords),
        latency_ms         = rag_result.latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BM25 retriever
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 retriever over a list of text chunks."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        if BM25_OK and chunks:
            tokenized = [_tokenize(c) for c in chunks]
            self.bm25 = BM25Okapi(tokenized)
            self._ok  = True
        else:
            self._ok  = False

    def retrieve(self, query: str, k: int = 4) -> Tuple[List[str], List[float]]:
        if not self._ok or not self.chunks:
            return [], []
        q_tokens = _tokenize(query)
        scores   = self.bm25.get_scores(q_tokens)
        top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        top_chunks  = [self.chunks[i] for i in top_idx]
        top_scores  = [float(scores[i]) for i in top_idx]
        return top_chunks, top_scores

    @property
    def available(self) -> bool:
        return self._ok


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_comparison(
    samples: List[EvalSample],
    rag_results: List[RAGResult],       # pre-computed RAG answers
    norag_results: List[RAGResult],     # pre-computed No-RAG answers
    bm25_results: List[RAGResult],      # pre-computed BM25 answers
) -> Dict:
    """
    Given pre-computed results for all three methods, compute metrics
    and return a structured comparison dict.
    """
    all_scores: Dict[str, List[MetricScores]] = {
        "RAG": [], "No-RAG": [], "BM25": []
    }

    result_map = {
        "RAG":    rag_results,
        "No-RAG": norag_results,
        "BM25":   bm25_results,
    }

    for method, results in result_map.items():
        for sample, result in zip(samples, results):
            scores = compute_all_metrics(result, sample)
            all_scores[method].append(scores)

    def _avg(lst: List[MetricScores], attr: str) -> float:
        vals = [getattr(s, attr) for s in lst]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    metrics_list = [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevancy", "bleu", "rouge1", "rouge2", "rougeL", "mrr"
    ]

    comparison = {}
    for method, scores in all_scores.items():
        comparison[method] = {m: _avg(scores, m) for m in metrics_list}
        comparison[method]["avg_score"]   = round(
            sum(comparison[method][m] for m in metrics_list) / len(metrics_list), 4)
        comparison[method]["avg_latency"] = _avg(scores, "latency_ms")
        comparison[method]["per_sample"]  = [asdict(s) for s in scores]

    # winner per metric
    winners = {}
    for m in metrics_list + ["avg_score"]:
        winner = max(comparison.keys(), key=lambda meth: comparison[meth][m])
        winners[m] = winner
    comparison["_winners"] = winners

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI helper — renders the full evaluation page
# ─────────────────────────────────────────────────────────────────────────────

def render_evaluation_page(
    llm,
    retriever,
    chunks: List[str],
    ingested_files: List[str],
):
    """
    Call this from app.py on the Evaluation page.
    Handles the full UI + orchestrates all three baselines.
    """
    import streamlit as st
    from langchain.schema.runnable import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    st.header("📐 Formal Evaluation")
    st.markdown(
        "Run **RAGAS-style metrics**, **BLEU/ROUGE**, **MRR**, and a "
        "**3-way baseline comparison** (RAG vs No-RAG vs BM25) on your uploaded documents."
    )

    if not ingested_files:
        st.warning("⬆️ Upload at least one PDF on the Chat page first.")
        return

    # ── sample questions ──────────────────────────────────────────────────────
    st.subheader("1 · Define evaluation questions")
    st.caption(
        "Enter 3–5 questions with their expected (ground truth) answers. "
        "These are used to compute all metrics."
    )

    DEFAULT_SAMPLES = [
        {
            "question": "What is the main topic of this document?",
            "ground_truth": "The document discusses the primary subject matter in detail.",
            "keywords": "topic,subject,main,overview"
        },
        {
            "question": "What are the key findings or conclusions?",
            "ground_truth": "The key findings include the important results presented.",
            "keywords": "findings,results,conclusions,key"
        },
        {
            "question": "Who are the main entities or authors mentioned?",
            "ground_truth": "Several authors and entities are referenced throughout.",
            "keywords": "authors,entities,mentioned,references"
        },
    ]

    if "eval_samples" not in st.session_state:
        st.session_state.eval_samples = DEFAULT_SAMPLES.copy()

    samples_input = []
    for i, s in enumerate(st.session_state.eval_samples):
        with st.expander(f"Question {i+1}", expanded=(i == 0)):
            q  = st.text_area("Question",      value=s["question"],      key=f"eq_{i}", height=68)
            gt = st.text_area("Ground truth",  value=s["ground_truth"],  key=f"egt_{i}", height=68)
            kw = st.text_input("Keywords (comma-separated)", value=s.get("keywords",""), key=f"ekw_{i}")
            samples_input.append({"question": q, "ground_truth": gt, "keywords": kw})

    col_add, col_remove = st.columns([1, 1])
    with col_add:
        if st.button("＋ Add question") and len(st.session_state.eval_samples) < 8:
            st.session_state.eval_samples.append(
                {"question": "", "ground_truth": "", "keywords": ""}
            )
            st.rerun()
    with col_remove:
        if st.button("－ Remove last") and len(st.session_state.eval_samples) > 1:
            st.session_state.eval_samples.pop()
            st.rerun()

    # ── run button ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("2 · Run evaluation")

    methods_selected = st.multiselect(
        "Methods to compare",
        ["RAG", "No-RAG", "BM25"],
        default=["RAG", "No-RAG", "BM25"]
    )
    k_chunks = st.slider("Chunks retrieved per question (k)", 2, 8, 4)

    if st.button("🚀 Run evaluation", type="primary"):
        # validate
        eval_samples = []
        for s in samples_input:
            if s["question"].strip() and s["ground_truth"].strip():
                eval_samples.append(EvalSample(
                    question=s["question"].strip(),
                    ground_truth=s["ground_truth"].strip(),
                    expected_keywords=[kw.strip() for kw in s["keywords"].split(",") if kw.strip()]
                ))
        if not eval_samples:
            st.error("Add at least one complete question + ground truth.")
            st.stop()

        # BM25 retriever
        bm25_ret = BM25Retriever(chunks) if BM25_OK else None

        # prompt templates
        rag_prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        norag_prompt = ChatPromptTemplate.from_template(
            "Answer the following question from your general knowledge:\n\n"
            "Question: {question}\n\nAnswer:"
        )

        def _rag_answer(question: str, ret, prompt_template, method: str) -> RAGResult:
            t0 = time.time()
            try:
                docs   = ret.invoke(question)
                chunks_used = [d.page_content for d in docs][:k_chunks]
                context = "\n\n".join(chunks_used)
                chain  = prompt_template | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
            except Exception as e:
                chunks_used = []
                answer = f"[Error: {e}]"
            latency = (time.time() - t0) * 1000
            return RAGResult(question=question, answer=answer,
                             retrieved_chunks=chunks_used, latency_ms=latency, method=method)

        def _norag_answer(question: str) -> RAGResult:
            t0 = time.time()
            try:
                chain  = norag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"question": question})
            except Exception as e:
                answer = f"[Error: {e}]"
            latency = (time.time() - t0) * 1000
            return RAGResult(question=question, answer=answer,
                             retrieved_chunks=[], latency_ms=latency, method="No-RAG")

        def _bm25_answer(question: str) -> RAGResult:
            t0 = time.time()
            bm25_chunks, _ = bm25_ret.retrieve(question, k=k_chunks) if bm25_ret else ([], [])
            context = "\n\n".join(bm25_chunks)
            try:
                chain  = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
            except Exception as e:
                answer = f"[Error: {e}]"
            latency = (time.time() - t0) * 1000
            return RAGResult(question=question, answer=answer,
                             retrieved_chunks=bm25_chunks, latency_ms=latency, method="BM25")

        # run all methods
        rag_res, norag_res, bm25_res = [], [], []
        progress = st.progress(0, text="Running evaluation…")
        total    = len(eval_samples) * len(methods_selected)
        done     = 0

        for sample in eval_samples:
            if "RAG" in methods_selected:
                with st.spinner(f"RAG → {sample.question[:50]}…"):
                    rag_res.append(_rag_answer(sample.question, retriever, rag_prompt, "RAG"))
                done += 1; progress.progress(done/total)

            if "No-RAG" in methods_selected:
                with st.spinner(f"No-RAG → {sample.question[:50]}…"):
                    norag_res.append(_norag_answer(sample.question))
                done += 1; progress.progress(done/total)

            if "BM25" in methods_selected:
                if not BM25_OK:
                    st.warning("rank_bm25 not installed — skipping BM25. Add it to requirements.txt")
                elif not chunks:
                    st.warning("No chunks available for BM25.")
                else:
                    with st.spinner(f"BM25 → {sample.question[:50]}…"):
                        bm25_res.append(_bm25_answer(sample.question))
                done += 1; progress.progress(done/total)

        progress.empty()

        # pad missing method lists so zip works
        empty_result = lambda q, m: RAGResult(q, "", [], 0.0, m)
        if "RAG"    not in methods_selected: rag_res   = [empty_result(s.question,"RAG")    for s in eval_samples]
        if "No-RAG" not in methods_selected: norag_res = [empty_result(s.question,"No-RAG") for s in eval_samples]
        if "BM25"   not in methods_selected or not bm25_res:
            bm25_res = [empty_result(s.question,"BM25") for s in eval_samples]

        comparison = run_baseline_comparison(eval_samples, rag_res, norag_res, bm25_res)
        st.session_state["eval_results"] = comparison
        st.session_state["eval_samples_done"] = eval_samples
        st.rerun()

    # ── display results ───────────────────────────────────────────────────────
    if "eval_results" not in st.session_state:
        return

    comparison   = st.session_state["eval_results"]
    eval_samples = st.session_state.get("eval_samples_done", [])
    winners      = comparison.get("_winners", {})
    methods      = [k for k in comparison if not k.startswith("_")]

    st.divider()
    st.subheader("3 · Results")

    # ── avg score cards ───────────────────────────────────────────────────────
    cols = st.columns(len(methods))
    for col, method in zip(cols, methods):
        avg  = comparison[method]["avg_score"]
        lat  = comparison[method]["avg_latency"]
        best = (winners.get("avg_score") == method)
        col.metric(
            label=f"{'🏆 ' if best else ''}{method} — avg score",
            value=f"{avg:.1%}",
            delta=f"{lat:.0f} ms avg"
        )

    st.divider()

    # ── metric comparison table ───────────────────────────────────────────────
    st.subheader("Metric comparison table")
    metric_labels = {
        "context_precision":  "Context Precision ↑",
        "context_recall":     "Context Recall ↑",
        "faithfulness":       "Faithfulness ↑",
        "answer_relevancy":   "Answer Relevancy ↑",
        "bleu":               "BLEU ↑",
        "rouge1":             "ROUGE-1 ↑",
        "rouge2":             "ROUGE-2 ↑",
        "rougeL":             "ROUGE-L ↑",
        "mrr":                "MRR ↑",
    }

    import pandas as pd
    table_data = {"Metric": []}
    for m in methods:
        table_data[m] = []

    for metric_key, label in metric_labels.items():
        table_data["Metric"].append(label)
        for m in methods:
            val  = comparison[m][metric_key]
            best = (winners.get(metric_key) == m)
            table_data[m].append(f"{'★ ' if best else ''}{val:.4f}")

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── charts ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Visual comparison")

    import plotly.graph_objects as go

    metric_keys   = list(metric_labels.keys())
    display_names = list(metric_labels.values())
    COLORS = {"RAG": "#4F8EF7", "No-RAG": "#F7874F", "BM25": "#50C878"}

    # radar chart
    fig_radar = go.Figure()
    for method in methods:
        vals = [comparison[method][m] for m in metric_keys]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=display_names + [display_names[0]],
            fill="toself",
            name=method,
            line_color=COLORS.get(method, "#888"),
            opacity=0.7
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Metric radar — all methods",
        height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # bar chart per metric
    fig_bar = go.Figure()
    for method in methods:
        vals = [comparison[method][m] for m in metric_keys]
        fig_bar.add_trace(go.Bar(
            name=method,
            x=display_names,
            y=vals,
            marker_color=COLORS.get(method, "#888"),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside"
        ))
    fig_bar.update_layout(
        barmode="group",
        title="Side-by-side metric scores",
        yaxis=dict(range=[0, 1.15]),
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # latency chart
    latencies = {m: comparison[m]["avg_latency"] for m in methods}
    fig_lat = go.Figure(go.Bar(
        x=list(latencies.keys()),
        y=list(latencies.values()),
        marker_color=[COLORS.get(m, "#888") for m in latencies],
        text=[f"{v:.0f} ms" for v in latencies.values()],
        textposition="outside"
    ))
    fig_lat.update_layout(title="Average latency (ms) — lower is better", height=300)
    st.plotly_chart(fig_lat, use_container_width=True)

    # ── per-sample answers viewer ─────────────────────────────────────────────
    st.divider()
    st.subheader("Per-question answers & scores")
    for i, sample in enumerate(eval_samples):
        with st.expander(f"Q{i+1}: {sample.question[:80]}…"):
            for method in methods:
                per = comparison[method].get("per_sample", [])
                if i >= len(per):
                    continue
                s = per[i]
                st.markdown(f"**{method}**")
                st.caption(
                    f"Precision {s['context_precision']:.3f} · "
                    f"Recall {s['context_recall']:.3f} · "
                    f"Faithfulness {s['faithfulness']:.3f} · "
                    f"Relevancy {s['answer_relevancy']:.3f} · "
                    f"BLEU {s['bleu']:.3f} · "
                    f"ROUGE-L {s['rougeL']:.3f} · "
                    f"MRR {s['mrr']:.3f}"
                )

    # ── export ────────────────────────────────────────────────────────────────
    st.divider()
    export_data = {
        "summary": {m: {k: v for k, v in comparison[m].items() if k != "per_sample"}
                    for m in methods},
        "winners": winners,
        "per_sample": {m: comparison[m].get("per_sample", []) for m in methods}
    }
    st.download_button(
        "⬇️ Download full results (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name="evaluation_results.json",
        mime="application/json"
    )
