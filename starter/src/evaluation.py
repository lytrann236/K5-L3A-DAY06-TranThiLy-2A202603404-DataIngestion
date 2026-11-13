"""Evidence-based retrieval scoring (Hit@1, Evidence Hit@3) with full per-query traces.

A retrieved chunk is a hit when it (a) belongs to the gold source, (b) covers a gold page
and (c) contains the whole evidence quote (whitespace-insensitive). Alternative gold
evidence (list in `evidence_quote`, paired with `gold_page`) is accepted.
Human labels (coherence, grounding, metadata correctness) are NOT inferred here.
"""
from __future__ import annotations

from typing import Any

from .ingestion import collapse


def gold_alternatives(q: dict[str, Any]) -> list[tuple[str, list[int]]]:
    ev = q["evidence_quote"]
    if isinstance(ev, list):  # one evidence per alternative gold page
        return [(e, [p]) for e, p in zip(ev, q["gold_page"])]
    return [(ev, list(q["gold_page"]))]


def chunk_is_hit(q: dict[str, Any], hit: dict[str, Any]) -> tuple[bool, str]:
    if hit["document_id"] != q["gold_source_id"]:
        return False, "wrong_source"
    text = collapse(hit["text"])
    page_ok = False
    for ev, pages in gold_alternatives(q):
        if set(hit["pages"]) & set(pages):
            page_ok = True
            if collapse(ev) in text:
                return True, "hit"
    return False, "page_ok_evidence_incomplete" if page_ok else "wrong_page"


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]], strategy: str,
                   applied_filter: dict | None) -> dict[str, Any]:
    labels = [chunk_is_hit(question, h) for h in hits]
    filt = question.get("filter")
    return {
        "strategy": strategy, "id": question["id"], "split": question["split"],
        "type": question.get("type"), "question": question["question"],
        "gold_page": question["gold_page"], "evidence_quote": question["evidence_quote"],
        "question_filter": filt, "applied_filter": applied_filter,
        "hit@1": bool(labels and labels[0][0]), "hit@3": any(ok for ok, _ in labels),
        "filter_compliant_top3": (all(all(h[k] == v for k, v in filt.items()) for h in hits)
                                  if filt else None),
        "results": [{"rank": i + 1, **{k: h[k] for k in ("chunk_id", "page", "pages", "chapter",
                                                         "section", "score")},
                     "score": round(float(h["score"]), 5), "auto_label": lab, "text": h["text"]}
                    for i, (h, (_, lab)) in enumerate(zip(hits, labels))],
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strat in sorted({t["strategy"] for t in traces}):
        for split in ("dev", "test", "all"):
            ts = [t for t in traces if t["strategy"] == strat and (split == "all" or t["split"] == split)]
            if not ts:
                continue
            fq = [t for t in ts if t["question_filter"]]
            rows.append({
                "strategy": strat, "split": split, "N": len(ts),
                "hit1": sum(t["hit@1"] for t in ts), "hit3": sum(t["hit@3"] for t in ts),
                "filter_questions": len(fq),
                "filter_compliant": sum(bool(t["filter_compliant_top3"]) for t in fq),
            })
    return rows
