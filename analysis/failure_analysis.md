# Failure Analysis — Lab 18: Production RAG

**Cá nhân:** Phó Viết Tiến Anh
**Ngày chạy:** 18/08/2026

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | N/A | N/A | N/A |
| Answer Relevancy | N/A | N/A | N/A |
| Context Precision | N/A | N/A | N/A |
| Context Recall | N/A | N/A | N/A |

Lần xác minh không dùng `OPENAI_API_KEY`, nên RAGAS đi vào fallback và report lưu `0.0`. Đây là trạng thái “chưa đo”, không phải điểm chất lượng. Bottom-5 dưới đây được chọn bằng lexical ground-truth recall trên top-3 context của pipeline thật.

## Bottom-5 Failures

### 1. Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt; có xác nhận cấu hình CNTT và ít nhất ba báo giá.
- **Got:** Context đầu nói về hoàn chi đào tạo 30 triệu.
- **Worst signal:** Context recall xấp xỉ 0.29.
- **Error Tree:** Output sai → context thiếu policy mua sắm → “30 triệu/phê duyệt” kéo nhầm tài liệu đào tạo → retrieval failure.
- **Root cause:** Trùng số tiền lấn át chủ đề; câu hỏi còn cần hai quy định khác nhau.
- **Suggested fix:** Query decomposition thành ngưỡng phê duyệt và yêu cầu thiết bị CNTT; lọc `category=mua_sam` rồi hợp nhất context.

### 2. Nghỉ không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** CEO phê duyệt; trên 14 ngày người lao động tự đóng phần bảo hiểm.
- **Got:** Context đầu là quyền lợi nghỉ không lương trong thời gian thử việc.
- **Worst signal:** Context recall xấp xỉ 0.37.
- **Error Tree:** Output sai → context sai loại nhân viên → ràng buộc “20 ngày” không được ưu tiên → rerank failure.
- **Root cause:** Lexical reranker coi các đoạn cùng cụm “nghỉ không lương” gần như nhau.
- **Suggested fix:** Dùng cross-encoder thật và regression test cho numeric range 16–30 ngày.

### 3. Senior 9 năm: phép và lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày; 20–35 triệu VNĐ/tháng.
- **Got:** Context đầu nói về nghỉ không lương.
- **Worst signal:** Context recall xấp xỉ 0.43.
- **Error Tree:** Output sai → context chỉ bao phủ một phần → câu hỏi cần hai nguồn → single-query retrieval failure.
- **Root cause:** Multi-hop giữa policy phép v2024 và bảng lương; top-3 không bảo đảm đủ hai nguồn.
- **Suggested fix:** Tách thành hai subquery, giữ ít nhất một context mỗi nguồn, rồi tính `15 + floor(9/3)`.

### 4. Số ngày phép năm hiện hành

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày theo v2024; v2023 có 12 ngày nhưng đã bị thay thế.
- **Got:** Context đầu là policy v2023 với 12 ngày.
- **Worst signal:** Context recall xấp xỉ 0.47.
- **Error Tree:** Output sai → context liên quan nhưng sai phiên bản → thiếu trạng thái hiệu lực → version-selection failure.
- **Root cause:** RRF không biết `v2024` mới hơn `v2023`; source mới chỉ là tên file.
- **Suggested fix:** Trích `effective_date`, `version`, `status/superseded_by`; mặc định lọc phiên bản hiện hành.

### 5. Lương thử việc Junior cao nhất

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** `85% × 20 triệu = 17 triệu VNĐ/tháng`.
- **Got:** Context đầu chỉ có tỷ lệ 85%, thiếu mức trần Junior.
- **Worst signal:** Context recall xấp xỉ 0.56.
- **Error Tree:** Output thiếu → context có công thức nhưng thiếu toán hạng → multi-hop retrieval failure.
- **Root cause:** Hai bằng chứng nằm ở `thu_viec.md` và `bang_luong_2024.md`.
- **Suggested fix:** Retrieve hai nguồn, yêu cầu generator nêu phép tính và kiểm tra đơn vị.

## Case Study

Lỗi phép năm 2023/2024 nằm ở lựa chọn bằng chứng, không phải generation. Query khớp cả hai phiên bản và RRF không hiểu thời gian hiệu lực. Fix ưu tiên là chuẩn hóa metadata phiên bản lúc ingestion, lọc policy hiện hành trước rerank và thêm regression test khẳng định query chung trả 15 ngày.

Nếu có thêm một giờ, ưu tiên version-aware retrieval rồi query decomposition. Hai thay đổi này xử lý ba trong năm failure quan sát được.
