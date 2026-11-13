"""Plumbing test ONLY: runs the whole CLI with a lexical hashing embedder.

This is NOT a semantic embedding and its numbers must NOT be reported as lab results.
Outputs go to RESULTS_DIR (default /tmp/day6_smoke). Requires DATABASE_URL.
"""
import hashlib, math, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("RESULTS_DIR", "/tmp/day6_smoke")
from src import run  # noqa: E402


class HashingTestEmbedder:
    provider, model_name, dimensions = "test-hashing", "TEST-ONLY-hashing-256", 256

    def _v(self, t):
        v = [0.0] * self.dimensions
        for w in re.findall(r"[a-z0-9_()]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dimensions] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    def embed_documents(self, texts): return [self._v(t) for t in texts]
    def embed_query(self, q): return self._v(q)


run.set_embedder(HashingTestEmbedder())
run.cmd_check(); run.cmd_reingest_check(); run.cmd_evaluate("all"); run.cmd_audit()
print("SMOKE TEST PASSED (numbers above are not lab results)")
