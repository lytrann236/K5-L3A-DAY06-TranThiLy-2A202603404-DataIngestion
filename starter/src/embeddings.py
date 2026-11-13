"""Real embedding providers. No mock/random fallback: any provider error is raised."""
from __future__ import annotations

import math
import os
from typing import Protocol


class Embedder(Protocol):
    provider: str
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, question: str) -> list[float]: ...


def _l2(v) -> list[float]:
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class SentenceTransformerEmbedder:
    """Local model via sentence-transformers (default BAAI/bge-small-en-v1.5, 384-d).

    Query/document prefixes follow the model card: e5 -> 'query: '/'passage: ';
    bge-*-en -> retrieval instruction on queries only.
    """
    provider = "sentence-transformers"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=os.getenv("EMBEDDING_DEVICE") or None)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        low = model_name.lower()
        if "e5" in low:
            self.q_prefix, self.d_prefix = "query: ", "passage: "
        elif "bge" in low and "-en" in low:
            self.q_prefix, self.d_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self.q_prefix = self.d_prefix = ""

    def _enc(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def embed_documents(self, texts):
        return self._enc([self.d_prefix + t for t in texts])

    def embed_query(self, question):
        return self._enc([self.q_prefix + question])[0]


class OpenAIEmbedder:
    """OpenAI-compatible API (OpenAI or OpenRouter via OPENAI_BASE_URL)."""
    provider = "openai"

    def __init__(self, model_name: str) -> None:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY / OPENROUTER_API_KEY missing in starter/.env")
        base = os.getenv("OPENAI_BASE_URL") or (
            "https://openrouter.ai/api/v1" if not os.getenv("OPENAI_API_KEY") else None)
        self.client = OpenAI(api_key=key, base_url=base)
        self.model_name = model_name
        self.dimensions = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i:i + 64])
            out.extend(_l2(d.embedding) for d in sorted(resp.data, key=lambda d: d.index))
        return out

    def embed_query(self, question):
        return self.embed_documents([question])[0]


class WordLlamaEmbedder:
    """Static-token-embedding model (pip `wordllama`, weights ship inside the wheel).

    Useful when the machine cannot download models from Hugging Face. Weaker than a
    transformer encoder; report it as such if used.
    """
    provider = "wordllama"

    def __init__(self, model_name: str = "l2_supercat", dim: int = 256) -> None:
        from wordllama import WordLlama
        self.model_name = f"wordllama/{model_name}-{dim}"
        self.model = WordLlama.load(config=model_name, dim=dim)
        self.dimensions = dim

    def embed_documents(self, texts):
        return [_l2(v) for v in self.model.embed(texts, norm=True)]

    def embed_query(self, question):
        return self.embed_documents([question])[0]


def get_embedder() -> Embedder:
    provider = (os.getenv("EMBEDDING_PROVIDER") or "sentence-transformers").lower()
    model = os.getenv("EMBEDDING_MODEL") or ""
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model or "BAAI/bge-small-en-v1.5")
    if provider in ("openai", "openrouter"):
        return OpenAIEmbedder(model or "text-embedding-3-small")
    if provider == "wordllama":
        return WordLlamaEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER={provider}")
