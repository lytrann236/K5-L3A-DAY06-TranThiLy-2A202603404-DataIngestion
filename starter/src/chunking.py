"""Fixed-size (A) and structure-aware (B) chunking over a document's page stream.

Both chunkers work on the same concatenated stream of normalized pages so that the
only A-vs-B difference is the splitting rule. Every chunk keeps document_id, source,
start page, all covered pages, chapter and section (section active at chunk start).
"""
from __future__ import annotations

import bisect
import hashlib
import re

from .models import Chunk, DocumentPage

BOX_MARKERS = re.compile(
    r"^(Learning objectives|CONCEPTS IN PRACTICE|CHECKPOINT|TRY IT|Chapter Outline|Introduction"
    r"|EXAMPLE \d+\.\d+|MULTIPLE USE OF A NAMED ARGUMENT|Highlights from this chapter include:)$")
SECTION_HEADING = re.compile(r"^\d{1,2}\.\d [A-Z][A-Za-z/ ]+$")
QUIZ_ITEM = re.compile(r"^\d{1,2} \. ")
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


class Stream:
    """Concatenated text of one document plus offset -> page/section lookup."""

    def __init__(self, pages: list[DocumentPage]):
        self.pages = pages
        self.text = ""
        self.page_starts: list[int] = []
        self.sec_offsets: list[int] = []
        self.sec_ids: list[str] = []
        for pg in pages:
            if self.text:
                self.text += "\n"
            start = len(self.text)
            self.page_starts.append(start)
            self.sec_offsets.append(start)
            self.sec_ids.append(pg.metadata["start_section"])
            for off, sid in pg.metadata["section_spans"]:
                self.sec_offsets.append(start + off)
                self.sec_ids.append(sid)
            self.text += pg.text

    def page_index(self, offset: int) -> int:
        return max(0, bisect.bisect_right(self.page_starts, offset) - 1)

    def section_at(self, offset: int) -> str:
        return self.sec_ids[max(0, bisect.bisect_right(self.sec_offsets, offset) - 1)]

    def make_chunk(self, strategy: str, idx: int, start: int, end: int, text: str) -> Chunk:
        first = self.pages[self.page_index(start)]
        last_i = self.page_index(max(start, end - 1))
        covered = [p.page for p in self.pages[self.page_index(start): last_i + 1]]
        text = text.strip()
        return Chunk(
            chunk_id=f"{strategy}-{first.document_id}-{idx:04d}",
            document_id=first.document_id, source=first.source, page=first.page, text=text,
            metadata={"pages": covered, "chapter": first.metadata["chapter"],
                      "section": self.section_at(start), "char_start": start, "char_end": end,
                      "char_len": len(text),
                      "content_hash": hashlib.sha256(text.encode()).hexdigest()[:16]})


def fixed_size(pages: list[DocumentPage], size: int, overlap: int, strategy: str = "A") -> list[Chunk]:
    """Sliding character windows of `size` with `overlap`, ignoring structure."""
    if not 0 <= overlap < size:
        raise ValueError("require 0 <= overlap < size")
    s = Stream(pages)
    chunks, start, idx = [], 0, 0
    while start < len(s.text):
        end = min(start + size, len(s.text))
        chunks.append(s.make_chunk(strategy, idx, start, end, s.text[start:end]))
        idx += 1
        if end == len(s.text):
            break
        start = end - overlap
    return chunks


def _is_subheading(line: str, prev: str | None, nxt: str | None) -> bool:
    """Short title-like line between paragraphs, e.g. 'String immutability'."""
    if len(line) > 60 or len(line.split()) > 7 or not line[0].isalpha():
        return False
    if line[-1] in ".,;:?!\"')]" or "=" in line or line[0].islower() and "(" not in line:
        return False
    prev_ok = prev is None or prev[-1] in ".:?!)" or BOX_MARKERS.match(prev) is not None
    return prev_ok and nxt is not None and len(nxt) > 60


def _blocks(s: Stream) -> list[tuple[int, int]]:
    """Split the stream at section headings, box markers and sub-headings."""
    lines, pos = [], 0
    for line in s.text.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    cuts = {0}
    in_outline = False
    for i, (off, line) in enumerate(lines):
        if line == "Chapter Outline":
            in_outline = True
        elif line == "Introduction":
            in_outline = False
        elif in_outline:
            continue
        prev = lines[i - 1][1] if i else None
        nxt = lines[i + 1][1] if i + 1 < len(lines) else None
        if SECTION_HEADING.match(line) or BOX_MARKERS.match(line) or _is_subheading(line, prev, nxt):
            cuts.add(off)
    cuts = sorted(cuts) + [len(s.text)]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if s.text[a:b].strip()]


def _segments(s: Stream, a: int, b: int) -> list[tuple[int, int]]:
    """Paragraph/quiz-item segments inside a block (line ends with . : ? or a quiz item starts)."""
    segs, seg_start, pos = [], a, a
    lines = s.text[a:b].split("\n")
    for i, line in enumerate(lines):
        line_end = pos + len(line) + 1
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        boundary = nxt is None or QUIZ_ITEM.match(nxt) or (line.endswith((".", ":", "?")) and len(line) < 95)
        if boundary:
            segs.append((seg_start, min(line_end, b)))
            seg_start = line_end
        pos = line_end
    return [x for x in segs if s.text[x[0]:x[1]].strip()]


def _split_long(s: Stream, a: int, b: int, max_size: int) -> list[tuple[int, int]]:
    """Split an oversized segment at sentence ends, then hard-cut if a sentence is too long."""
    out, cur = [], a
    while b - cur > max_size:
        window = s.text[cur:cur + max_size]
        cut = max((m.start() for m in SENT_END.finditer(window)), default=-1)
        if cut <= max_size // 3:
            cut = window.rfind("\n")
        if cut <= max_size // 3:
            cut = max_size
        out.append((cur, cur + cut))
        cur += cut
    out.append((cur, b))
    return out


def by_paragraph(pages: list[DocumentPage], max_size: int, min_size: int = 200,
                 strategy: str = "B") -> list[Chunk]:
    """Structure-aware chunks: one block per heading/box, packed by paragraph up to max_size.

    Small blocks (< min_size, e.g. a lone 'CHECKPOINT' + video link) are merged with the
    neighbouring block when both lie in the same section and the result fits max_size.
    """
    s = Stream(pages)
    pieces: list[tuple[int, int]] = []
    for a, b in _blocks(s):
        cur_a = cur_b = None
        for sa, sb in _segments(s, a, b):
            for xa, xb in (_split_long(s, sa, sb, max_size) if sb - sa > max_size else [(sa, sb)]):
                if cur_a is None:
                    cur_a, cur_b = xa, xb
                elif xb - cur_a <= max_size:
                    cur_b = xb
                else:
                    pieces.append((cur_a, cur_b))
                    cur_a, cur_b = xa, xb
        if cur_a is not None:
            pieces.append((cur_a, cur_b))
    merged: list[tuple[int, int]] = []
    for a, b in pieces:
        if merged:
            pa, pb = merged[-1]
            small = pb - pa < min_size or b - a < min_size
            if small and b - pa <= max_size and s.section_at(pa) == s.section_at(a):
                merged[-1] = (pa, b)
                continue
        merged.append((a, b))
    return [s.make_chunk(strategy, i, a, b, s.text[a:b]) for i, (a, b) in enumerate(merged)]
