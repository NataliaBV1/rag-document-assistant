"""Load configuration from YAML into a typed object."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    data_dir: str
    index_path: str
    chunks_path: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_batch_size: int
    top_k: int
    llm_provider: str
    llm_model: str
    max_tokens: int
    temperature: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
