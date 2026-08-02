"""Orchestrate the full flow: index ingestion and end-to-end querying."""
from __future__ import annotations

from .config import Config
from .embeddings import Embedder
from .generation import generate_answer
from .ingestion import build_chunks
from .retrieval import Retriever
from .vectorstore import VectorStore


def ingest(cfg: Config) -> None:
    """Build the vector index from data/raw and persist it."""
    chunks = build_chunks(cfg.data_dir, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No documents found in {cfg.data_dir}")
    embedder = Embedder(cfg.embedding_model, cfg.embedding_batch_size)
    embeddings = embedder.encode([c.text for c in chunks])
    store = VectorStore(embedder.dim)
    store.add(embeddings, [c.text for c in chunks], [c.source for c in chunks])
    store.save(cfg.index_path, cfg.chunks_path)
    print(f"Indexed {len(chunks)} chunks -> {cfg.index_path}")


class RAGPipeline:
    """Load the index and answer questions."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedding_model, cfg.embedding_batch_size)
        self.store = VectorStore.load(cfg.index_path, cfg.chunks_path)
        self.retriever = Retriever(self.embedder, self.store, cfg.top_k)

    def answer(self, question: str) -> dict:
        contexts = self.retriever.retrieve(question)
        answer = generate_answer(
            question, contexts, self.cfg.llm_model,
            self.cfg.max_tokens, self.cfg.temperature,
        )
        return {"answer": answer, "contexts": contexts}
