from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def extract_document_metadata(text: str, source: str) -> dict:
    """Extract stable provenance/version fields from a policy document."""
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    version_match = re.search(r"Phiên bản:\s*([^|\n]+)", text, flags=re.IGNORECASE)
    effective_match = re.search(r"Ngày hiệu lực:\s*([^|\n]+)", text, flags=re.IGNORECASE)
    department_match = re.search(r"Phòng ban:\s*([^|\n]+)", text, flags=re.IGNORECASE)
    status_match = re.search(r"Trạng thái:\s*([^|\n]+)", text, flags=re.IGNORECASE)
    superseded = bool(re.search(r"đã (?:được )?thay thế|trạng thái:\s*đã thay thế",
                                text, flags=re.IGNORECASE))
    version_from_source = re.search(r"(?:^|[_-])v(\d+(?:\.\d+)*)", source, re.IGNORECASE)

    version = (version_match.group(1).strip() if version_match else
               version_from_source.group(1) if version_from_source else "")
    status = status_match.group(1).strip() if status_match else (
        "superseded" if superseded else "current"
    )
    family = re.sub(r"_v\d+(?:\.\d+)?(?=\.[^.]+$)", "", source, flags=re.IGNORECASE)
    return {
        "source": source,
        "document_family": family,
        "title": title_match.group(1).strip() if title_match else os.path.splitext(source)[0],
        "version": version,
        "effective_date": effective_match.group(1).strip() if effective_match else "",
        "department": department_match.group(1).strip() if department_match else "",
        "status": status,
        "is_current": not superseded,
    }


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(fp)
        docs.append({"text": text, "metadata": extract_document_metadata(text, source)})

    # Reconcile versioned files as a family. An old document may not say that it
    # was superseded because that fact only exists in its successor.
    families: dict[str, list[dict]] = {}
    for doc in docs:
        families.setdefault(doc["metadata"]["document_family"], []).append(doc)
    for family_docs in families.values():
        if len(family_docs) < 2:
            continue

        def version_key(doc: dict) -> tuple:
            date = doc["metadata"].get("effective_date", "")
            date_match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", date)
            if date_match:
                day, month, year = map(int, date_match.groups())
                return year, month, day
            numbers = tuple(int(part) for part in re.findall(r"\d+", doc["metadata"].get("version", "")))
            return (0, 0, 0, *numbers)

        newest = max(family_docs, key=version_key)
        for doc in family_docs:
            is_current = doc is newest
            doc["metadata"]["is_current"] = is_current
            if not is_current:
                doc["metadata"]["status"] = "superseded"
                doc["metadata"]["superseded_by"] = newest["metadata"]["source"]

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            source = os.path.basename(fp)
            docs.append({"text": text, "metadata": extract_document_metadata(text, source)})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
                 if part.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    try:
        from numpy import dot
        from numpy.linalg import norm
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(
            sentences, show_progress_bar=False
        )
        similarities = [
            float(dot(embeddings[i - 1], embeddings[i]) /
                  (norm(embeddings[i - 1]) * norm(embeddings[i]) + 1e-9))
            for i in range(1, len(sentences))
        ]
    except Exception as exc:  # noqa: BLE001 - model loading has many optional-runtime errors
        # A lexical similarity fallback keeps ingestion usable offline.
        print(f"  ⚠️  Semantic model unavailable, using lexical fallback: {exc}")
        fallback = chunk_basic(text, chunk_size=500, metadata=metadata)
        for index, chunk in enumerate(fallback):
            chunk.metadata.update({"strategy": "semantic_fallback", "chunk_index": index})
        return fallback

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    return [Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": i})
            for i, group in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [], []

    def split_units(value: str, limit: int) -> list[str]:
        """Keep paragraphs/sentences intact when possible, hard-split only oversized units."""
        if len(value) <= limit:
            return [value]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", value) if s.strip()]
        units: list[str] = []
        for sentence in sentences:
            units.extend(sentence[i:i + limit] for i in range(0, len(sentence), limit))
        return units

    parent_texts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for unit in split_units(paragraph, parent_size):
            candidate = f"{current}\n\n{unit}" if current else unit
            if current and len(candidate) > parent_size:
                parent_texts.append(current)
                current = unit
            else:
                current = candidate
    if current:
        parent_texts.append(current)

    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    safe_source = re.sub(r"[^\w.-]+", "_", source)
    source_cursor = 0
    for parent_index, parent_text in enumerate(parent_texts):
        parent_start = text.find(parent_text, source_cursor)
        if parent_start < 0:
            parent_start = source_cursor
        source_cursor = parent_start + len(parent_text)
        headings_before_parent = re.findall(r"^#{1,3}\s+(.+)$", text[:parent_start + 1], re.MULTILINE)
        parent_section = headings_before_parent[-1].strip() if headings_before_parent else metadata.get("title", "")
        parent_id = f"{safe_source}:parent_{parent_index}"
        parent_meta = {**metadata, "chunk_type": "parent", "parent_id": parent_id,
                       "chunk_index": parent_index, "section": parent_section}
        parents.append(Chunk(parent_text, parent_meta))

        child_units = split_units(parent_text, child_size)
        child_texts: list[str] = []
        child_current = ""
        for unit in child_units:
            candidate = f"{child_current} {unit}".strip()
            if child_current and len(candidate) > child_size:
                child_texts.append(child_current)
                child_current = unit
            else:
                child_current = candidate
        if child_current:
            child_texts.append(child_current)
        child_cursor = 0
        for child_index, child_text in enumerate(child_texts):
            child_start = parent_text.find(child_text, child_cursor)
            if child_start < 0:
                child_start = child_cursor
            child_cursor = child_start + len(child_text)
            headings = re.findall(r"^#{1,3}\s+(.+)$", parent_text[:child_start + 1], re.MULTILINE)
            section = headings[-1].strip() if headings else parent_section
            child_meta = {**metadata, "chunk_type": "child", "chunk_index": child_index,
                          "section": section}
            children.append(Chunk(child_text, child_meta, parent_id=parent_id))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    header = ""
    preamble = ""
    for part in parts:
        if not part:
            continue
        if re.match(r"^#{1,3}\s+", part):
            header = part.strip()
            continue
        content = part.strip()
        if not content:
            continue
        if header:
            chunk_text = f"{header}\n\n{content}"
            section = re.sub(r"^#{1,3}\s+", "", header)
        else:
            preamble = content
            chunk_text = content
            section = "preamble"
        chunks.append(Chunk(chunk_text, {**metadata, "section": section,
                                         "strategy": "structure", "chunk_index": len(chunks)}))
    if not chunks and preamble:
        chunks.append(Chunk(preamble, {**metadata, "section": "preamble",
                                       "strategy": "structure", "chunk_index": 0}))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
