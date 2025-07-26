#!/usr/bin/env python3
"""
Evaluate the RAG pipeline using RAGAS metrics.

This script reads the evaluation questions from `eval/questions.jsonl`,
executes the RAG pipeline for each question, collects the answers
and contexts, then computes RAGAS metrics such as faithfulness,
answer relevance, context precision and context recall.  Results are
written to the `reports/` directory both as a Markdown summary and
as a CSV file for further analysis.

If no OpenAI API key is provided, the script will still run the
pipeline using the local fallback model, but the RAGAS metrics may
degrade because generative evaluations rely on large language models.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict

from datasets import Dataset  # type: ignore
from ragas import evaluate  # type: ignore
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall  # type: ignore

import sys
from pathlib import Path

# Ensure internal imports work when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1] / "rag-industrial-eval-guardrails"))

from app.settings import Settings
from app.rag_pipeline import answer_query
from app.retriever import Retriever
from app.reranker import Reranker


async def run_evaluation() -> Dict[str, float]:
    settings = Settings()
    eval_file = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file {eval_file} not found")
    # Read evaluation dataset
    questions: List[str] = []
    ground_truths: List[List[str]] = []
    with eval_file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj["question"])
            ground_truths.append(obj["ground_truths"])
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    retriever = Retriever(settings)
    reranker = Reranker(settings)  # type: ignore[name-defined]
    # Run pipeline for each question
    for q in questions:
        response = await answer_query(q, settings)
        answers.append(response.answer)
        # Retrieve contexts used by the pipeline for evaluation
        retrieved = await retriever.hybrid_search(q)
        reranked = reranker.rerank(q, retrieved)
        top_contexts = reranked[:4]
        contexts_list.append([c.content for c in top_contexts])
    # Build HuggingFace dataset
    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truths": ground_truths,
    })
    metrics = [faithfulness, answer_relevance, context_precision, context_recall]
    results = evaluate(ds, metrics)  # type: ignore[no-untyped-call]
    # Save report
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_md = reports_dir / "ragas_report.md"
    report_csv = reports_dir / "ragas_report.csv"
    with report_md.open("w", encoding="utf-8") as md_f:
        md_f.write("# Rapport d'évaluation RAGAS\n\n")
        md_f.write("| Metric | Score |\n")
        md_f.write("|-------|-------|\n")
        for metric_name, score in results.items():
            md_f.write(f"| {metric_name} | {score:.3f} |\n")
    # CSV
    with report_csv.open("w", encoding="utf-8") as csv_f:
        csv_f.write("metric,score\n")
        for metric_name, score in results.items():
            csv_f.write(f"{metric_name},{score}\n")
    return results


def main() -> None:
    results = asyncio.run(run_evaluation())
    for metric, score in results.items():
        print(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    main()