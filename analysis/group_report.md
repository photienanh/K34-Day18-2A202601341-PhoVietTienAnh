# Individual Report — Lab 18: Production RAG

**Học viên:** Phó Viết Tiến Anh
**Ngày:** 18/08/2026

## Phạm vi hoàn thành

| Module | Nội dung | Kết quả |
|---|---|---:|
| M1 | Semantic, hierarchical, structure-aware chunking | Pass |
| M2 | Vietnamese BM25, Qdrant dense, RRF | Pass |
| M3 | CrossEncoder reranking + benchmark | Pass |
| M4 | RAGAS wrapper + diagnostic analysis | Pass |
| M5 | Bốn techniques + combined single-call | Pass |

Tổng cộng **37/37 tests pass**. Pipeline offline chạy end-to-end với 26 tài liệu, 106 child chunks và Qdrant local.

## Kết quả đánh giá

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | N/A | N/A | N/A |
| Answer Relevancy | N/A | N/A | N/A |
| Context Precision | N/A | N/A | N/A |
| Context Recall | N/A | N/A | N/A |

RAGAS chưa chạy bằng LLM judge để tránh tự động sử dụng API key. `0.0` trong JSON là fallback “chưa đo”. Cần chạy lại `python main.py` với đủ requirements và key hợp lệ để có điểm chính thức.

## Key Findings

1. Provenance được giữ xuyên suốt chunk → enrichment → Qdrant → rerank, giúp truy nguyên source/version.
2. Multi-hop và numeric range cần nhiều bằng chứng; top-3 đơn thuần dễ chỉ giữ một nửa câu trả lời.
3. Trùng số tiền (“30 triệu”) có thể kéo tài liệu đào tạo lên trên mua sắm.
4. Không có metadata hiệu lực thì hybrid retrieval vẫn có thể ưu tiên policy cũ.

## Latency Breakdown (offline fallback)

| Bước | Thời gian quan sát |
|---|---:|
| Load + hierarchical chunking | ~0.1 s |
| Deterministic enrichment (106 chunks) | <0.1 s |
| BM25 + hashing dense index | ~1.4–2.7 s |
| Tổng production pipeline | ~3.0 s |

Các số này đo fallback CPU, không đại diện latency BGE-M3, cross-encoder hoặc OpenAI API.

## Next Optimization

Metadata hiệu lực → query decomposition → cross-encoder thật → RAGAS judge → regression test cho version, negation và numeric range.
