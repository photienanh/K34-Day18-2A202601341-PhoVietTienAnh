from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except (ImportError, RuntimeError):
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c["text"]).lower().split()
                              for c in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).lower().split())
        indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [SearchResult(self.documents[i]["text"], float(scores[i]),
                             dict(self.documents[i].get("metadata", {})), "bm25")
                for i in indices[:top_k] if float(scores[i]) > 0]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:  # noqa: BLE001 - provide offline fallback for model errors
                print(f"  ⚠️  Dense model unavailable, using hashing encoder: {exc}")
                import numpy as np

                class HashingEncoder:
                    @staticmethod
                    def _encode_one(text: str):
                        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                        for token in re.findall(r"\w+", text.lower()):
                            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                            index = int.from_bytes(digest, "big") % EMBEDDING_DIM
                            vector[index] += 1.0
                        norm = np.linalg.norm(vector)
                        return vector / norm if norm else vector

                    def encode(self, values, **_kwargs):
                        if isinstance(values, str):
                            return self._encode_one(values)
                        return np.asarray([self._encode_one(value) for value in values])

                self._encoder = HashingEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return
        vectors = self._get_encoder().encode([c["text"] for c in chunks], show_progress_bar=True)
        points = [PointStruct(id=i, vector=vector.tolist(),
                              payload={**chunk.get("metadata", {}), "text": chunk["text"]})
                  for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection_name=collection, query=vector, limit=top_k)
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            results.append(SearchResult(text, float(point.score), payload, "dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    fused: dict[tuple, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            # Text alone is not a safe identity: two policy versions may contain
            # the same clause while carrying different provenance/version metadata.
            identity = (result.metadata.get("source"), result.metadata.get("parent_id"),
                        result.metadata.get("chunk_index"), result.text)
            entry = fused.setdefault(identity, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:max(top_k, 0)]
    return [SearchResult(item["result"].text, item["score"],
                         dict(item["result"].metadata), "hybrid") for item in ranked]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
