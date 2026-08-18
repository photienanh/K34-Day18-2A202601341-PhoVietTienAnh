from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def _chat(system: str, user: str, max_tokens: int) -> str:
    """Call the configured LLM; callers own their offline fallback."""
    if not OPENAI_API_KEY:
        return ""
    from openai import OpenAI
    response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json_object(value: str) -> dict:
    """Accept plain JSON and JSON wrapped in Markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    try:
        generated = _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                          text, 150)
        if generated:
            return generated
    except Exception as exc:  # noqa: BLE001 - API failures use deterministic fallback
        print(f"  ⚠️  OpenAI summarize failed: {exc}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    summary = " ".join(sentences[:2])
    return summary or text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0 or not text.strip():
        return []
    try:
        generated = _chat(
            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
            "Trả về mỗi câu hỏi trên một dòng.", text, 200)
        if generated:
            questions = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                         for line in generated.splitlines() if line.strip()]
            return questions[:n_questions]
    except Exception as exc:  # noqa: BLE001 - API failures use deterministic fallback
        print(f"  ⚠️  OpenAI HyQA failed: {exc}")
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    return [f"Thông tin nào được nêu về {sentence.rstrip('.')}?"
            for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text:
        return text
    try:
        context = _chat(
            "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì.",
            f"Tài liệu: {document_title or 'không rõ'}\n\nĐoạn văn:\n{text}", 80)
        if context:
            return f"{context}\n\n{text}"
    except Exception as exc:  # noqa: BLE001 - API failures use deterministic fallback
        print(f"  ⚠️  OpenAI contextual failed: {exc}")
    prefix = f"Trích từ {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    try:
        generated = _chat(
            'Trích xuất metadata và chỉ trả JSON: {"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}',
            text, 150)
        if generated:
            return _parse_json_object(generated)
    except Exception as exc:  # noqa: BLE001 - API/JSON failures use deterministic fallback
        print(f"  ⚠️  OpenAI metadata failed: {exc}")
    lowered = text.lower()
    category = ("it" if any(word in lowered for word in ("mật khẩu", "vpn", "dữ liệu"))
                else "finance" if any(word in lowered for word in ("lương", "chi phí", "thưởng"))
                else "hr" if any(word in lowered for word in ("nhân viên", "nghỉ", "đào tạo"))
                else "policy")
    return {"topic": "general", "entities": [], "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    try:
        generated = _chat(
            """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{"summary":"tóm tắt 2-3 câu","questions":["câu hỏi 1","câu hỏi 2","câu hỏi 3"],
"context":"một câu mô tả vị trí/chủ đề","metadata":{"topic":"...","entities":[],
"category":"policy|hr|it|finance","language":"vi|en"}}""",
            f"Tài liệu: {source or 'không rõ'}\n\nĐoạn văn:\n{text}", 400)
        if generated:
            result = _parse_json_object(generated)
            if result:
                return result
    except Exception as exc:  # noqa: BLE001 - API/JSON failures use deterministic fallback
        print(f"  ⚠️  Enrichment API failed: {exc}")
    # Deterministic fallback still enriches retrieval text and preserves provenance.
    title = source or "tài liệu không rõ nguồn"
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Đoạn này trích từ {title}.",
        "metadata": extract_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
