"""PostgreSQL + pgvector store: stable IDs, idempotent re-ingest, SQL metadata filter."""
from __future__ import annotations

import json
from typing import Any

import psycopg

from .models import Chunk

TABLE = "day6_chunks"
FILTERABLE = {"chapter": "chapter", "section": "section", "document_id": "document_id"}


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, model_name: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL missing")
        if dimensions <= 0:
            raise ValueError("invalid embedding dimensions")
        self.url, self.dim, self.model = database_url, dimensions, model_name

    def connect(self):
        return psycopg.connect(self.url, connect_timeout=5)

    def create_schema(self) -> None:
        with self.connect() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    chunk_id     text NOT NULL,
                    index_name   text NOT NULL,
                    model        text NOT NULL,
                    document_id  text NOT NULL,
                    source       text NOT NULL,
                    page         int  NOT NULL,
                    pages        int[] NOT NULL,
                    chapter      int  NOT NULL,
                    section      text NOT NULL,
                    text         text NOT NULL,
                    char_len     int  NOT NULL,
                    content_hash text NOT NULL,
                    metadata     jsonb NOT NULL DEFAULT '{{}}',
                    embedding    vector({self.dim}) NOT NULL,
                    PRIMARY KEY (index_name, model, chunk_id))""")
            row = c.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,)).fetchone()
            if row and row[0] != f"vector({self.dim})":
                raise RuntimeError(f"{TABLE}.embedding is {row[0]} but model has {self.dim} dims; "
                                   f"drop the table or use the matching model")

    def existing_hashes(self, index_name: str, document_id: str) -> dict[str, str]:
        with self.connect() as c:
            rows = c.execute(f"SELECT chunk_id, content_hash FROM {TABLE} WHERE index_name=%s "
                             "AND model=%s AND document_id=%s", (index_name, self.model, document_id))
            return dict(rows.fetchall())

    def upsert(self, index_name: str, chunks: list[Chunk], vectors: list[list[float]],
               document_id: str) -> dict[str, int]:
        """Insert/update given chunks and delete obsolete chunks of this document/index."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"vector has {len(v)} dims, expected {self.dim}")
        with self.connect() as c, c.cursor() as cur:
            for ch, v in zip(chunks, vectors):
                m = ch.metadata
                cur.execute(f"""
                    INSERT INTO {TABLE} (chunk_id, index_name, model, document_id, source, page, pages,
                        chapter, section, text, char_len, content_hash, metadata, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (index_name, model, chunk_id) DO UPDATE SET
                        document_id=EXCLUDED.document_id, source=EXCLUDED.source, page=EXCLUDED.page,
                        pages=EXCLUDED.pages, chapter=EXCLUDED.chapter, section=EXCLUDED.section,
                        text=EXCLUDED.text, char_len=EXCLUDED.char_len,
                        content_hash=EXCLUDED.content_hash, metadata=EXCLUDED.metadata,
                        embedding=EXCLUDED.embedding""",
                    (ch.chunk_id, index_name, self.model, ch.document_id, ch.source, ch.page,
                     m["pages"], m["chapter"], m["section"], ch.text, m["char_len"],
                     m["content_hash"], json.dumps(m), _vec(v)))
        return {"written": len(chunks)}

    def delete_obsolete(self, index_name: str, document_id: str, keep_ids: list[str]) -> int:
        with self.connect() as c:
            cur = c.execute(f"DELETE FROM {TABLE} WHERE index_name=%s AND model=%s AND document_id=%s "
                            "AND NOT (chunk_id = ANY(%s))", (index_name, self.model, document_id, keep_ids))
            return cur.rowcount

    def count(self, index_name: str) -> int:
        with self.connect() as c:
            return c.execute(f"SELECT count(*) FROM {TABLE} WHERE index_name=%s AND model=%s",
                             (index_name, self.model)).fetchone()[0]

    def search(self, index_name: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Exact cosine search (no ANN index), metadata filter applied in SQL WHERE."""
        where, params = ["index_name = %s", "model = %s"], [index_name, self.model]
        for key, value in (filters or {}).items():
            if key not in FILTERABLE:
                raise ValueError(f"filter on '{key}' not allowed")
            where.append(f"{FILTERABLE[key]} = %s")
            params.append(value)
        q = _vec(query_vector)
        sql = (f"SELECT chunk_id, document_id, source, page, pages, chapter, section, text, "
               f"1 - (embedding <=> %s::vector) AS score FROM {TABLE} WHERE {' AND '.join(where)} "
               f"ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self.connect() as c:
            rows = c.execute(sql, [q, *params, q, top_k]).fetchall()
        cols = ["chunk_id", "document_id", "source", "page", "pages", "chapter", "section", "text", "score"]
        return [dict(zip(cols, r)) for r in rows]
