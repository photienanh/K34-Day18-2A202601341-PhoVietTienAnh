from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    fallback = {name: 0.0 for name in metric_names}
    fallback["per_question"] = []
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if not questions:
        return fallback
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        evaluation = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                context_precision, context_recall])
        frame = evaluation.to_pandas()

        def number(value) -> float:
            try:
                value = float(value)
                return 0.0 if math.isnan(value) else value
            except (TypeError, ValueError):
                return 0.0

        per_question = [EvalResult(
            str(row["question"]), str(row["answer"]), list(row["contexts"]),
            str(row["ground_truth"]), *(number(row.get(name, 0.0)) for name in metric_names)
        ) for _, row in frame.iterrows()]
        return {**{name: (sum(getattr(item, name) for item in per_question) /
                          len(per_question) if per_question else 0.0)
                    for name in metric_names}, "per_question": per_question}
    except Exception as exc:  # noqa: BLE001 - evaluation must not crash the pipeline
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        return fallback


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("Câu trả lời có chi tiết không được context hỗ trợ",
                           "Siết prompt chỉ dùng bằng chứng và giảm temperature"),
        "answer_relevancy": ("Câu trả lời chưa bám đúng ý câu hỏi",
                              "Cải thiện prompt và giữ nguyên ràng buộc của câu hỏi"),
        "context_precision": ("Retriever đưa vào quá nhiều chunk không liên quan",
                              "Tăng chất lượng reranking hoặc lọc theo metadata/version"),
        "context_recall": ("Context thiếu bằng chứng cần thiết",
                           "Điều chỉnh chunking và mở rộng hybrid retrieval"),
    }
    metric_names = tuple(diagnostic_tree)
    ranked = []
    for item in eval_results:
        scores = {name: float(getattr(item, name)) for name in metric_names}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append((sum(scores.values()) / len(scores), {
            "question": item.question, "answer": item.answer,
            "ground_truth": item.ground_truth, "contexts": item.contexts,
            "worst_metric": worst_metric, "score": scores[worst_metric],
            "average_score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        }))
    ranked.sort(key=lambda pair: pair[0])
    return [entry for _, entry in ranked[:max(bottom_n, 0)]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
