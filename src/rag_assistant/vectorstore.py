"""FAISS vector index: build, save, load and search."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int) -> None:
        # IndexFlatIP = inner product; with normalized vectors this equals cosine.
        self.index = faiss.IndexFlatIP(dim)
        self.texts: list[str] = []
        self.sources: list[str] = []

    def add(self, embeddings: np.ndarray, texts: list[str], sources: list[str]) -> None:
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.sources.extend(sources)

    def search(self, query_emb: np.ndarray, top_k: int) -> list[dict]:
        """Return the top_k most similar chunks with their score."""
        scores, idxs = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0], strict=True):
            if idx == -1:
                continue
            results.append(
                {"text": self.texts[idx], "source": self.sources[idx], "score": float(score)}
            )
        return results

    def save(self, index_path: str, chunks_path: str) -> None:
        faiss.write_index(self.index, index_path)
        Path(chunks_path).write_text(
            json.dumps({"texts": self.texts, "sources": self.sources}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: str, chunks_path: str) -> VectorStore:
        index = faiss.read_index(index_path)
        payload = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        store = cls.__new__(cls)
        store.index = index
        store.texts = payload["texts"]
        store.sources = payload["sources"]
        return store
