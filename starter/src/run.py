"""Day 06 pipeline CLI. Run from starter/:  python -m src.run <command>

Commands
  check            validate corpus (>=30 counted pages) and benchmark evidence vs. extracted text
  ingest           chunk + embed + upsert indexes A and B (C reuses index B + filter)
  reingest-check   run ingest again and prove no duplicates / no re-embedding
  evaluate         top-3 for every question x strategy -> traces, metrics (dev/test)
  audit            write manual-labelling sheets (coherence, metadata, grounding)
  all              check -> ingest -> reingest-check -> evaluate -> audit
Options: --split dev|test|all (evaluate only; default all)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as im
from pathlib import Path

from dotenv import load_dotenv

from . import chunking
from .embeddings import get_embedder
from .evaluation import evaluate_query, summarize
from .ingestion import collapse, load_manifest, load_pdf
from .store import PgVectorStore

STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
load_dotenv(STARTER / ".env")
RESULTS = Path(os.getenv("RESULTS_DIR") or ROOT / "results")

MANIFEST = ROOT / "data" / "manifest.json"
QUESTIONS = ROOT / "benchmark" / "questions.jsonl"
CONFIG = STARTER / "configs" / "strategies.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_corpus(log: list | None = None):
    man = load_manifest(MANIFEST)
    src = man["sources"][0]
    pdf = Path(os.getenv("PDF_PATH") or ROOT / "data" / "raw" / "Introduction_to_Python_Programming-WEB.pdf")
    if not pdf.is_absolute():
        pdf = (STARTER / pdf).resolve()
    pages = load_pdf(pdf, src["source_id"], src["source_path_or_url"], src["pages"], log)
    return src, pages


def build_chunks(cfg: dict, pages) -> list:
    fn = getattr(chunking, cfg["chunker"])
    return fn(pages, **cfg["params"], strategy=cfg["index"])


def unique_indexes(cfgs: dict) -> dict:
    out = {}
    for name, cfg in cfgs.items():
        out.setdefault(cfg["index"], cfg)
    return out


def versions() -> dict:
    v = {"python": platform.python_version()}
    for pkg in ("pypdf", "psycopg", "sentence-transformers", "torch", "openai", "wordllama"):
        try:
            v[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            pass
    return v


# ---------------------------------------------------------------- commands
def cmd_check() -> None:
    log: list = []
    src, pages = load_corpus(log)
    counted = len(pages)
    by_page = {p.page: collapse(p.text) for p in pages}
    qs = load_questions()
    problems = []
    from .evaluation import gold_alternatives
    for q in qs:
        for ev, ps in gold_alternatives(q):  # every documented alternative must exist
            if not any(collapse(ev) in by_page.get(p, "") for p in ps):
                problems.append(f"{q['id']}@{ps}")
    splits = {s: sum(q["split"] == s for q in qs) for s in ("dev", "test")}
    filt = {s: sum(bool(q.get("filter")) and q["split"] == s for q in qs) for s in ("dev", "test")}
    report = {"checked_at": now(), "content_pages": counted,
              "total_chars": sum(len(p.text) for p in pages), "questions": len(qs), "splits": splits,
              "filter_questions": filt, "evidence_not_found": problems, "ingest_log": log}
    write_json(RESULTS / "corpus_check.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "ingest_log"}, indent=2))
    errs = []
    if counted < 30: errs.append("fewer than 30 content pages")
    if len(qs) < 20 or splits["dev"] < 15 or splits["test"] < 5: errs.append("benchmark too small")
    if sum(filt.values()) < 3 or filt["test"] < 1: errs.append("need >=3 filter questions, >=1 in test")
    if problems: errs.append(f"evidence not found: {problems}")
    if errs:
        sys.exit("CHECK FAILED: " + "; ".join(errs))
    print("Corpus and benchmark OK")


_EMBEDDER = None


def embedder():
    """One embedder per process (tests may inject one via set_embedder)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = get_embedder()
    return _EMBEDDER


def set_embedder(emb) -> None:
    global _EMBEDDER
    _EMBEDDER = emb


def get_store(emb) -> PgVectorStore:
    store = PgVectorStore(os.getenv("DATABASE_URL", ""), emb.dimensions, emb.model_name)
    store.create_schema()
    return store


def cmd_ingest(emb=None, tag: str = "ingest") -> dict:
    t0 = time.perf_counter()
    emb = emb or embedder()
    log: list = []
    src, pages = load_corpus(log)
    store = get_store(emb)
    cfgs = json.loads(CONFIG.read_text())
    stats = {"run_at": now(), "provider": emb.provider, "model": emb.model_name,
             "dimensions": emb.dimensions, "metric": "cosine (pgvector <=>), exact scan, no ANN index",
             "versions": versions(), "cost_usd": 0.0 if emb.provider != "openai" else None,
             "pages": len(pages), "indexes": {}}
    for index, cfg in unique_indexes(cfgs).items():
        ti = time.perf_counter()
        chunks = build_chunks(cfg, pages)
        before = store.count(index)
        existing = store.existing_hashes(index, src["source_id"])
        todo = [c for c in chunks if existing.get(c.chunk_id) != c.metadata["content_hash"]]
        te = time.perf_counter()
        vecs = emb.embed_documents([c.text for c in todo]) if todo else []
        embed_s = time.perf_counter() - te
        if todo:
            store.upsert(index, todo, vecs, src["source_id"])
        deleted = store.delete_obsolete(index, src["source_id"], [c.chunk_id for c in chunks])
        lens = [c.metadata["char_len"] for c in chunks]
        stats["indexes"][index] = {
            "config": cfg, "chunks": len(chunks), "rows_before": before, "rows_after": store.count(index),
            "embedded": len(todo), "unchanged": len(chunks) - len(todo), "deleted_obsolete": deleted,
            "len_min": min(lens), "len_median": statistics.median(lens), "len_mean": round(statistics.mean(lens), 1),
            "len_max": max(lens), "total_chars": sum(lens), "approx_tokens": sum(lens) // 4,
            "chunks_spanning_pages": sum(len(c.metadata["pages"]) > 1 for c in chunks),
            "embed_seconds": round(embed_s, 2), "index_seconds": round(time.perf_counter() - ti, 2)}
        print(f"[{tag}] index {index}: {len(chunks)} chunks, embedded {len(todo)}, "
              f"deleted {deleted}, rows {before}->{stats['indexes'][index]['rows_after']}")
        write_chunks(index, chunks)
    stats["total_seconds"] = round(time.perf_counter() - t0, 2)
    stats["ingest_log"] = log
    write_json(RESULTS / f"{tag}_stats.json", stats)
    return stats


def write_chunks(index: str, chunks) -> None:
    path = RESULTS / "chunks" / f"{index}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, "page": c.page, **c.metadata, "text": c.text},
                               ensure_ascii=False) + "\n")


def cmd_reingest_check() -> None:
    emb = embedder()
    s1 = cmd_ingest(emb, tag="ingest")
    s2 = cmd_ingest(emb, tag="reingest")
    rep = {"run_at": now(), "indexes": {}}
    for idx in s1["indexes"]:
        a, b = s1["indexes"][idx], s2["indexes"][idx]
        rep["indexes"][idx] = {"rows_after_first": a["rows_after"], "rows_after_second": b["rows_after"],
                               "embedded_second": b["embedded"],
                               "no_duplicates": a["rows_after"] == b["rows_after"] == b["chunks"]}
    write_json(RESULTS / "reingest_check.json", rep)
    print(json.dumps(rep, indent=2))
    if not all(v["no_duplicates"] for v in rep["indexes"].values()):
        sys.exit("re-ingest produced duplicates")


def cmd_evaluate(split: str = "all") -> None:
    emb = embedder()
    store = get_store(emb)
    cfgs = json.loads(CONFIG.read_text())
    qs = [q for q in load_questions() if split == "all" or q["split"] == split]
    for idx in unique_indexes(cfgs):
        if store.count(idx) == 0:
            sys.exit(f"index {idx} is empty for model {emb.model_name}: run ingest first")
    traces, t0 = [], time.perf_counter()
    qvecs = {q["id"]: emb.embed_query(q["question"]) for q in qs}
    for name, cfg in cfgs.items():
        for q in qs:
            filt = q.get("filter") if cfg["use_filter"] and q.get("filter") else None
            hits = store.search(cfg["index"], qvecs[q["id"]], 3, filt)
            traces.append(evaluate_query(q, hits, name, filt))
    elapsed = time.perf_counter() - t0
    (RESULTS / "traces").mkdir(parents=True, exist_ok=True)
    for name in cfgs:
        with (RESULTS / "traces" / f"{name}_{split}.jsonl").open("w", encoding="utf-8") as f:
            for t in traces:
                if t["strategy"] == name:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
    rows = summarize(traces)
    write_json(RESULTS / f"metrics_{split}.json", {"run_at": now(), "model": emb.model_name,
                                                  "search_seconds_all_strategies": round(elapsed, 2),
                                                  "rows": rows})
    lines = ["| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Filter compliant top-3 |",
             "|---|---|---|---|---|"]
    for r in rows:
        fc = f"{r['filter_compliant']}/{r['filter_questions']}" if r["filter_questions"] else "N/A"
        lines.append(f"| {r['strategy']} | {r['split']} | {r['hit1']}/{r['N']} | {r['hit3']}/{r['N']} | {fc} |")
    (RESULTS / f"metrics_{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_audit() -> None:
    """Sheets for manual labels. Score/label columns are left EMPTY for the human grader."""
    cfgs = json.loads(CONFIG.read_text())
    _, pages = load_corpus()
    out = RESULTS / "manual"
    out.mkdir(parents=True, exist_ok=True)
    # 1) coherence: fixed positions = middle of the first 10 counted pages (chosen before scoring)
    stream = chunking.Stream(pages)
    chunks = {idx: build_chunks(cfg, pages) for idx, cfg in unique_indexes(cfgs).items()}
    with (out / "coherence_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position_id", "pdf_page", "anchor_text", "strategy", "chunk_id", "chunk_text",
                    "score_0_2", "reason"])
        for i, pg in enumerate(pages[:10]):
            off = stream.page_starts[i] + len(pg.text) // 2
            anchor = collapse(stream.text[max(0, off - 30): off + 30])
            for name, cfg in cfgs.items():
                ch = next(c for c in chunks[cfg["index"]]
                          if c.metadata["char_start"] <= off < c.metadata["char_end"])
                w.writerow([f"pos{i + 1:02d}", pg.page, anchor, name, ch.chunk_id, ch.text, "", ""])
    # 2) metadata + 3) grounding: from test traces
    tr = []
    for name in cfgs:
        p = RESULTS / "traces" / f"{name}_all.jsonl"
        if not p.exists():
            p = RESULTS / "traces" / f"{name}_test.jsonl"
        if not p.exists():
            sys.exit("run evaluate first")
        tr += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    test = [t for t in tr if t["split"] == "test"]
    page_text = {p.page: collapse(p.text) for p in pages}
    with (out / "metadata_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "qid", "rank", "chunk_id", "page", "pages", "chapter", "section",
                    "question_filter", "auto_text_on_claimed_pages", "auto_filter_ok",
                    "manual_source_page_correct", "manual_section_correct", "note"])
        for t in test:
            for r in t["results"]:
                head = collapse(r["text"])[:80]
                on_page = any(head in page_text.get(p, "") or head[:40] in page_text.get(p, "")
                              for p in r["pages"])
                fok = (all(r[k] == v for k, v in t["question_filter"].items())
                       if t["question_filter"] else "N/A")
                w.writerow([t["strategy"], t["id"], r["rank"], r["chunk_id"], r["page"], r["pages"],
                            r["chapter"], r["section"], json.dumps(t["question_filter"]), on_page, fok,
                            "", "", ""])
    with (out / "grounding_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "qid", "question", "evidence_quote", "auto_evidence_hit3",
                    "top1", "top2", "top3", "grade_0_2", "reason"])
        for t in test:
            texts = [f"[{r['chunk_id']} p{r['pages']}] {r['text']}" for r in t["results"]] + ["", "", ""]
            w.writerow([t["strategy"], t["id"], t["question"], json.dumps(t["evidence_quote"]),
                        t["hit@3"], *texts[:3], "", ""])
    print(f"Wrote manual sheets to {out} (fill the empty score/label columns yourself)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "ingest", "reingest-check", "evaluate", "audit", "all"])
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    a = ap.parse_args()
    if a.command == "check": cmd_check()
    elif a.command == "ingest": cmd_ingest()
    elif a.command == "reingest-check": cmd_reingest_check()
    elif a.command == "evaluate": cmd_evaluate(a.split)
    elif a.command == "audit": cmd_audit()
    else:
        cmd_check(); cmd_reingest_check(); cmd_evaluate("all"); cmd_audit()


if __name__ == "__main__":
    main()
