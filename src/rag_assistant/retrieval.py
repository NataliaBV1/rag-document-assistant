"""Retrieval: embed the query and search for the most relevant chunks."""
from __future__ import annotations

from .embeddings import Embedder
from .vectorstore import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_emb = self.embedder.encode([query])
        return self.store.search(query_emb, self.top_k)
        # TODO (impressive improvement): add reranking with a cross-encoder
        # over these initial top_k, or hybrid search (BM25 + dense).
