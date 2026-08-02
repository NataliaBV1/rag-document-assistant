"""Load documents (pdf/txt/md) and split them into chunks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str            # name of the source file
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def load_document(path: Path) -> str:
    """Read a document and return its plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported format: {suffix}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with overlap.

    Character-based baseline. TODO: try token-based chunking (tiktoken) and
    semantic chunking, and compare retrieval metrics across strategies.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def build_chunks(data_dir: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Load every document in a directory and split it into chunks."""
    chunks: list[Chunk] = []
    cid = 0
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        text = load_document(path)
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, source=path.name, chunk_id=cid))
            cid += 1
    return chunks
