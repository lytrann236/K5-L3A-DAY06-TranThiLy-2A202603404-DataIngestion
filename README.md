# Day 06 · Data Ingestion — bài làm

**Corpus:** OpenStax *Introduction to Python Programming*, Chương 8 Strings và Chương 9 Lists (PDF tr. 197–236 = tr. sách 187–226; 39 trang nội dung, CC BY-NC-SA 4.0). Chi tiết: [data/manifest.json](data/manifest.json).
**Benchmark:** 20 câu (15 dev / 5 test, 3 câu có filter `chapter`): [benchmark/](benchmark/README.md).
**Stack:** Python 3.11+, PostgreSQL 16 + pgvector (Docker), embedding local `BAAI/bge-small-en-v1.5` (384 chiều, sentence-transformers), cosine, exact search.

## Cấu trúc

```text
data/manifest.json          nguồn, giấy phép, trang được tính, số ký tự, lỗi trích xuất đã biết
data/raw/                   đặt PDF ở đây (không commit)
benchmark/questions.jsonl   câu hỏi + gold page + evidence (đã kiểm chứng)
starter/configs/strategies.json   tham số A / B / C
starter/src/                ingestion, chunking, embeddings, store (pgvector), evaluation, run (CLI)
starter/tests/smoke_test.py kiểm thử đường ống bằng embedder băm (KHÔNG phải kết quả)
artifacts/experiment_log.csv
results/                    sinh ra khi chạy: stats, traces, metrics, manual/ (nhãn chấm tay)
```

## Chạy

```bash
# 0) Tải PDF vào data/raw/ (giữ đúng tên)
#    https://assets.openstax.org/oscms-prodcms/media/documents/Introduction_to_Python_Programming-WEB.pdf
cd starter
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py                           # -> Setup OK

python -m src.run check            # >=30 trang, 20 câu, mọi evidence có trong text trang
python -m src.run reingest-check   # ingest A,B rồi ingest lại: không trùng, không embed lại
python -m src.run evaluate --split dev     # chọn cấu hình trên dev
python -m src.run evaluate --split all     # chạy test MỘT lần sau khi đã cố định cấu hình
python -m src.run audit            # sinh bảng chấm tay results/manual/*.csv
```

Lần chạy đầu sẽ tải model (~130 MB) từ Hugging Face; chi phí 0 USD.

## Kết quả và việc chấm tay

- `results/ingest_stats.json`: số chunk, phân bố độ dài, ký tự/token, thời gian embed, phiên bản thư viện.
- `results/reingest_check.json`: bằng chứng nạp lại không nhân đôi.
- `results/traces/<A|B|C>_<split>.jsonl`: top-3 đầy đủ (chunk ID, text, score, metadata, nhãn tự động).
- `results/metrics_<split>.md|json`: Hit@1, Evidence Hit@3 (n/N), filter compliance.
- `results/manual/`: `coherence_labels.csv` (10 vị trí cố định = giữa 10 trang đầu), `metadata_audit.csv`, `grounding_labels.csv`. Các cột điểm để trống để người chấm điền.

## Định nghĩa hit
Chunk thuộc `gold_source_id`, có ít nhất một trang trùng `gold_page`, và chứa **toàn bộ** `evidence_quote` (so sánh không phân biệt khoảng trắng). Chấp nhận mọi evidence thay thế đã ghi.

---

*Đề bài gốc:* [LAB.md](LAB.md)
