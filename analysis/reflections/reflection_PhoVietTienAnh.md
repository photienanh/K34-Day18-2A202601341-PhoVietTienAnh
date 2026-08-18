# Individual Reflection — Lab 18

**Tên:** Phó Viết Tiến Anh
**Phạm vi:** M1–M5 và pipeline tích hợp

## 1. Mapping bài giảng vào code

| Lecture concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | So cosine giữa câu kề nhau; có fallback khi model chưa sẵn sàng. |
| Parent-child retrieval | M1 | `chunk_hierarchical()` | 26 tài liệu tạo 106 child chunks với `parent_id` ổn định. |
| Structure-aware | M1 | `chunk_structure_aware()` | Heading cấp 1–3 được giữ trong text và metadata. |
| BM25 + dense fusion | M2 | `reciprocal_rank_fusion()` | RRF dùng thứ hạng, không trộn hai thang score. |
| Cross-encoder | M3 | `CrossEncoderReranker.rerank()` | Giữ retrieval và rerank score để debug. |
| RAGAS | M4 | `evaluate_ragas()` | Trả đủ bốn metric và fallback không làm pipeline crash. |
| Diagnostic tree | M4 | `failure_analysis()` | Metric thấp nhất map sang diagnosis/fix kiểm thử được. |
| Contextual enrichment | M5 | `_enrich_single_call()` | Một call trả summary, HyQA, context và metadata. |

## 2. Khó khăn và cách giải quyết

### Python environment

- **Exact errors:** `/bin/bash: python: command not found`; `/usr/bin/python3: No module named pytest`.
- **Debug:** Tạo `.venv`, cài dependency trong workspace và dùng `.venv/bin/python` nhất quán.
- **Bài học:** Không giả định alias `python` hay global packages luôn tồn tại.

### Dependency quá lớn

- **Hiện tượng:** Resolver bắt đầu tải PyTorch/CUDA hàng trăm MB dù chỉ chạy CPU.
- **Debug:** Dừng lượt cài, cài dependency CPU tối thiểu để test và xác minh model fallback.
- **Bài học:** Nên pin CPU wheel theo platform hoặc tách extra `models` khỏi core requirements.

### Qdrant local bị sandbox chặn

- **Exact error:** `httpx.ConnectError: [Errno 1] Operation not permitted`.
- **Debug:** Xác nhận container `Up`, rồi chạy với quyền local service; pipeline exit code 0.
- **Bài học:** Phân biệt network policy với lỗi service trước khi sửa client.

### Policy sai phiên bản

- **Quan sát:** Query phép năm trả v2023 (12 ngày) thay vì v2024 (15 ngày).
- **Debug:** Theo dõi `source` trong top context.
- **Bài học:** Relevance chưa đủ; hiệu lực phải là metadata/filter hạng nhất.

## 3. Action Plan

## Project: Trợ lý tra cứu chính sách nội bộ

### Hiện tại

- Pipeline: Markdown/PDF text → chunk → hybrid retrieval → answer.
- Issues: xung đột phiên bản, multi-hop, PDF scan chưa OCR, thiếu regression theo loại lỗi.

### Plan áp dụng

1. [ ] **Tuần 1:** Structure-aware parent/child chunking; thêm OCR queue.
2. [ ] **Tuần 1:** Chuẩn hóa `version`, `effective_date`, `status`, `category`, `superseded_by`.
3. [ ] **Tuần 2:** BM25 + BGE-M3; filter policy hiện hành trước RRF.
4. [ ] **Tuần 2:** BGE reranker; benchmark p50/p95 và đa dạng source.
5. [ ] **Tuần 3:** Query decomposition và hợp nhất bằng chứng multi-hop.
6. [ ] **Tuần 3:** RAGAS + exact/numeric/version metrics và regression set.
7. [ ] **Tuần 4:** Combined enrichment, cache theo content hash, theo dõi chi phí.

### Tiêu chí hoàn thành

- Regression version/negation/numeric pass.
- Ít nhất ba RAGAS metrics đạt 0.70.
- Không trả policy superseded cho query hiện hành.
- Có báo cáo latency p50/p95 và chi phí/query.

## 4. Tự đánh giá

| Tiêu chí | Tự chấm (1–5) |
|---|---:|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Problem solving | 4 |
| Chẩn đoán pipeline | 4 |

Điểm cần cải thiện là chạy model đầy đủ và LLM judge để thay fallback bằng số đo production thực.
