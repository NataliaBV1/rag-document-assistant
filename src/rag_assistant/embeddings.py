"""Embedding model wrapper (sentence-transformers)."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return normalized float32 embeddings for cosine similarity search."""
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # lets us use inner product = cosine
            show_progress_bar=len(texts) > 100,
        )
        return np.asarray(emb, dtype="float32")

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
