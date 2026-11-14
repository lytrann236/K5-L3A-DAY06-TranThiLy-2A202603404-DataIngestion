# Benchmark — Ch.8 Strings & Ch.9 Lists (OpenStax Intro to Python)

File: `questions.jsonl` — 20 câu (15 dev / 5 test), chốt **trước** khi chạy pipeline.

## Trường dữ liệu
`id, split, type, question, question_vi, expected_answer, gold_source_id, gold_page, evidence_quote, filter, evidence_verified`

- `gold_page`: số trang PDF vật lý (1-based, pypdf). Trang sách in = `gold_page - 10`.
- `gold_page` là list. Khi `evidence_quote` là **list**, phần tử thứ i là evidence thay thế nằm ở `gold_page[i]`; chấp nhận bất kỳ cặp nào.
- Nguồn thay thế: q15 (221 / 235); q01, q02, q10 (bảng tóm tắt chương 8, p.219); q04 (p.218); q11 (bảng 9.1, p.236). Các nguồn thay thế từ bảng tóm tắt được bổ sung **trước khi chạy bất kỳ embedding thật nào**: phát hiện khi chạy thử plumbing bằng embedder băm từ khóa (không phải kết quả lab), top-1 của q01 là bảng tóm tắt p.219 chứa câu tương đương. Chỉ sửa câu dev; test không bị sửa.
- Mọi evidence (kể cả thay thế) được kiểm tra là chuỗi con của text trang sau chuẩn hóa của pipeline: `cd starter && python -m src.run check`.
- Truy vấn dùng trường `question` (tiếng Anh, cùng ngôn ngữ corpus). `question_vi` chỉ để đọc; nếu muốn chạy truy vấn tiếng Việt thì phải dùng embedding đa ngữ và ghi rõ là thí nghiệm thêm.

## Loại câu
| Loại | Dev | Test |
|---|---|---|
| direct | q01 q03 q05 q06 q09 q11 | t03 t05 |
| paraphrase | q04 q07 | t04 |
| conditional / exception | q02 q08 q10 q12 q13 | t02 |
| filter (`chapter`) | q14 (ch8), q15 (ch9) | t01 (ch8) |

## Câu filter
Filter `chapter` là **input giả định có trước khi truy xuất** (người dùng đang học chương đó). Câu hỏi cố ý không nhắc "string"/"list", nên nếu không lọc thì có thể khớp nhầm chương kia:
- q14: duyệt bằng `for … in` (8.3) — dễ nhầm với "Iterating lists" (9.1).
- q15: có thay đổi được nội dung không (list, 9.1) — dễ nhầm với "String immutability" (8.2).
- t01: so sánh bằng `<`/`>` (8.1, ASCII) — dễ nhầm với sorting / dictionary order (9.2).
Cấu hình C chỉ lọc ở 3 câu này; 17 câu còn lại chạy không lọc.

## Cách chia dev/test
Chia theo **mục** để giảm rò rỉ: test lấy từ 8.1 (so sánh), 8.4 (numbered fields, numeric format), 9.4 (nested list), 9.5 (list comprehension).
Nguy cơ rò rỉ còn lại: 8.4 và 8.1 cũng có câu dev (q07–q09 cùng mục 8.4; q12 ở 9.2 cũng nhắc ASCII). Corpus chỉ có 2 chương nên không tách hoàn toàn theo tài liệu được.

## Tiêu chí chọn cấu hình (chốt trước khi xem test)
1. Evidence Hit@3 trên dev cao nhất;
2. nếu hòa: Hit@1 dev;
3. nếu vẫn hòa: coherence trung bình cao hơn;
4. rồi số chunk/thời gian ingest thấp hơn.
Test chỉ chạy một lần sau khi đã cố định cấu hình.
5. (Bổ sung sau khi xem dev, trước khi chạy test) Nếu vẫn hòa: chọn cấu hình có filter compliance trên dev cao hơn. Dev: B = C ở Hit@1 9/15, Hit@3 13/15; C tuân thủ filter 2/2, B 0/2 → chọn C.

## Ghi chú nguồn
Bảng 9.1 trong sách ghi `min(list)` là "Returns the maximum element" (lỗi của nguồn) — không dùng làm gold.
