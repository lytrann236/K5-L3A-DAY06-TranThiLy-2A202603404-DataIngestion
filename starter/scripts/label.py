"""Interactive labelling helper for results/manual/*.csv (the human grader types every label).

Usage (from starter/):
  python3 scripts/label.py coherence     # score_0_2 + reason, 10 positions x A/B/C
  python3 scripts/label.py grounding     # grade_0_2 + reason, 5 test questions x A/B/C
  python3 scripts/label.py metadata      # source/page and section correct? (y/n) + note

Progress is saved after every answer; rows already labelled are skipped, so you can
quit (q) and resume later. Type 'b' to go back one row (within the current session) and relabel it.
"""
import csv
import sys
import textwrap
from pathlib import Path

import os

MANUAL = Path(os.getenv("RESULTS_DIR") or Path(__file__).resolve().parents[2] / "results") / "manual"
SHEETS = {
    "coherence": ("coherence_labels.csv", [("score_0_2", "012"), ("reason", None)]),
    "grounding": ("grounding_labels.csv", [("grade_0_2", "012"), ("reason", None)]),
    "metadata": ("metadata_audit.csv", [("manual_source_page_correct", "yn"),
                                        ("manual_section_correct", "yn"), ("note", None)]),
}
SCALE = {
    "coherence": "0 = cut mid-idea / no subject, unreadable alone | 1 = partly understandable, missing context | 2 = self-contained, complete idea",
    "grounding": "0 = no evidence in top-3 | 1 = evidence present but missing condition/context | 2 = enough to answer and trace to source",
    "metadata": "Check page/pages and section against the PDF (pdf_page = book page + 10). y = correct, n = wrong",
}


def show(kind, row):
    bar = "=" * 78
    print("\n" + bar)
    if kind == "coherence":
        print(f"{row['position_id']}  page {row['pdf_page']}  strategy {row['strategy']}  {row['chunk_id']}")
        print(f"anchor (position being judged): ...{row['anchor_text']}...")
        print("-" * 78)
        print(row["chunk_text"])
    elif kind == "grounding":
        print(f"{row['qid']}  strategy {row['strategy']}  auto_evidence_hit3={row['auto_evidence_hit3']}")
        print("Q:", row["question"])
        print("Evidence:", row["evidence_quote"])
        for k in ("top1", "top2", "top3"):
            print("-" * 78)
            print(k.upper(), textwrap.shorten(row[k], 900, placeholder=" ..."))
    else:
        print(f"{row['qid']} rank {row['rank']}  strategy {row['strategy']}  {row['chunk_id']}")
        print(f"page={row['page']} pages={row['pages']} chapter={row['chapter']} section={row['section']}")
        print(f"question_filter={row['question_filter']}  auto_text_on_claimed_pages={row['auto_text_on_claimed_pages']}"
              f"  auto_filter_ok={row['auto_filter_ok']}")
    print(bar)


def ask(field, allowed):
    while True:
        try:
            v = input(f"{field}{' [' + '/'.join(allowed) + ']' if allowed else ''} (q=quit, b=back): ").strip()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if v in ("q", "b"):
            return v
        if allowed is None or v in allowed:
            return v
        print("  invalid value")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SHEETS:
        sys.exit(__doc__)
    kind = sys.argv[1]
    fname, fields = SHEETS[kind]
    path = MANUAL / fname
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)

    def save():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    print(SCALE[kind])
    todo = [i for i, r in enumerate(rows) if not r[fields[0][0]]]
    print(f"{len(rows) - len(todo)}/{len(rows)} rows already labelled")
    pos = 0
    while pos < len(todo):
        i = todo[pos]
        row = rows[i]
        show(kind, row)
        twin = None
        if kind == "coherence":
            twin = next((r for r in rows if r is not row and r["position_id"] == row["position_id"]
                         and r["chunk_id"] == row["chunk_id"] and r["score_0_2"]), None)
        elif kind == "grounding":
            twin = next((r for r in rows if r is not row and r["qid"] == row["qid"] and r["grade_0_2"]
                         and all(r[k] == row[k] for k in ("top1", "top2", "top3"))), None)
        if twin:
            print(f"Same retrieved text as strategy {twin['strategy']} (labelled {twin[fields[0][0]]}).")
            try:
                reuse = input("Press Enter to copy that label, or type n to label manually: ").strip() == ""
            except (EOFError, KeyboardInterrupt):
                break
            if reuse:
                row[fields[0][0]] = twin[fields[0][0]]
                row[fields[-1][0]] = f"same chunk as {twin['strategy']}; {twin[fields[-1][0]]}"
                save()
                pos += 1
                continue
        out = {}
        for name, allowed in fields:
            v = ask(name, list(allowed) if allowed else None)
            if v in ("q", "b"):
                break
            out[name] = v
        if v == "q":
            break
        if v == "b":
            pos = max(0, pos - 1)
            for name, _ in fields:
                rows[todo[pos]][name] = ""
            continue
        row.update(out)
        save()
        pos += 1
    done = sum(bool(r[fields[0][0]]) for r in rows)
    print(f"\nSaved {path.name}: {done}/{len(rows)} labelled")


if __name__ == "__main__":
    main()
