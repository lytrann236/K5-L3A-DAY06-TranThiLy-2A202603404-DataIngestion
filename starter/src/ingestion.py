"""Read and normalize source data (PDF -> one DocumentPage per physical page)."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

FOOTER = "Access for free at openstax.org"
# Running headers/footers, e.g. "188     8 • Strings" or "8.4 • String formatting     203"
RUNNING_HEADER = [
    re.compile(r"^\d{1,3}\s+\d{1,2} • .+$"),
    re.compile(r"^\d{1,2}(\.\d)? • .+\s\d{1,3}$"),
]
SECTION_HEADING = re.compile(r"^(\d{1,2}\.\d) ([A-Z][A-Za-z/ ]+)$")


def normalize_text(text: str) -> str:
    """NFKC, drop footer/running headers, collapse spaces inside lines, keep line breaks."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line or line == FOOTER or any(p.match(line) for p in RUNNING_HEADER):
            continue
        lines.append(line.replace(FOOTER, "").strip())
    return "\n".join(l for l in lines if l)


def collapse(text: str) -> str:
    """Whitespace-insensitive form used for evidence matching."""
    return re.sub(r"\s+", " ", text).strip()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pdf(path: Path, document_id: str, source: str, manifest_pages: list[dict],
             log: list[dict] | None = None) -> list[DocumentPage]:
    """Extract every page listed in the manifest; record empty/excluded pages in `log`.

    `page` = 1-based physical PDF page. Section metadata is propagated from in-text
    headings ("8.2 String slicing"), because even pages only carry the chapter header.
    """
    log = log if log is not None else []
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}. See data/manifest.json -> how_to_obtain")
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    current_section = None
    for meta in manifest_pages:
        p = meta["pdf_page"]
        try:
            text = normalize_text(reader.pages[p - 1].extract_text() or "")
        except Exception as exc:  # keep going, but record the failure
            log.append({"level": "error", "page": p, "event": "extract_failed", "detail": repr(exc)})
            continue
        if not meta.get("counted", True):
            log.append({"level": "info", "page": p, "event": "excluded",
                        "detail": meta.get("excluded_reason", ""), "chars": len(text)})
            continue
        if len(text) < 100:
            log.append({"level": "warning", "page": p, "event": "near_empty", "chars": len(text)})
        # section spans inside this page: list of (char_offset, section_id)
        spans = []
        offset = 0
        if current_section is None or not current_section.startswith(f"{meta['chapter']}."):
            current_section = f"{meta['chapter']}.0"  # chapter intro before first section
        start_section = current_section
        in_outline = False
        for line in text.split("\n"):
            # the "Chapter Outline" list repeats every section title: skip it until "Introduction"
            if line == "Chapter Outline":
                in_outline = True
            elif line == "Introduction":
                in_outline = False
            m = None if in_outline else SECTION_HEADING.match(line)
            if m and m.group(1).split(".")[0] == str(meta["chapter"]):
                current_section = m.group(1)
                spans.append((offset, current_section))
            offset += len(line) + 1
        pages.append(DocumentPage(
            document_id=document_id, source=source, page=p, text=text,
            metadata={"book_page": meta["book_page"], "chapter": meta["chapter"],
                      "chapter_title": meta["chapter_title"], "start_section": start_section,
                      "section_spans": spans}))
    if not pages:
        raise RuntimeError("No pages extracted")
    return pages
