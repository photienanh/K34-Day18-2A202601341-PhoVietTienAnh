from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001 - provide offline fallback for model errors
                print(f"  ⚠️  CrossEncoder unavailable, using lexical fallback: {exc}")

                class LexicalReranker:
                    @staticmethod
                    def predict(pairs):
                        scores = []
                        for query, document in pairs:
                            query_tokens = set(re.findall(r"\w+", query.lower()))
                            document_tokens = set(re.findall(r"\w+", document.lower()))
                            scores.append(len(query_tokens & document_tokens) /
                                          max(len(query_tokens), 1))
                        return scores

                self._model = LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        def document_with_provenance(doc: dict) -> str:
            metadata = doc.get("metadata", {})
            fields = ("source", "title", "section", "version", "effective_date", "status")
            header = " | ".join(f"{key}={metadata[key]}" for key in fields if metadata.get(key))
            return f"[{header}]\n{doc['text']}" if header else doc["text"]

        scores = self._load_model().predict(
            [(query, document_with_provenance(doc)) for doc in documents]
        )
        try:
            score_values = list(scores)
        except TypeError:
            score_values = [scores]
        explicit_history_query = bool(re.search(
            r"\b(?:v\d+(?:\.\d+)?|20\d{2})\b|phiên bản|bản cũ|trước đây|lịch sử",
            query, re.IGNORECASE,
        ))

        def sort_key(item):
            score, doc = item
            current_tiebreak = 0 if explicit_history_query else int(
                doc.get("metadata", {}).get("is_current", False)
            )
            return float(score), current_tiebreak

        scored = sorted(zip(score_values, documents), key=sort_key, reverse=True)
        return [RerankResult(doc["text"], float(doc.get("score", 0.0)), float(score),
                             dict(doc.get("metadata", {})), rank)
                for rank, (score, doc) in enumerate(scored[:top_k], start=1)]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        from flashrank import Ranker, RerankRequest
        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": d["text"], "meta": d.get("metadata", {})}
                    for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))[:top_k]
        return [RerankResult(item["text"], float(documents[int(item["id"])].get("score", 0.0)),
                             float(item["score"]), dict(item.get("meta", {})), rank)
                for rank, item in enumerate(results, start=1)]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
