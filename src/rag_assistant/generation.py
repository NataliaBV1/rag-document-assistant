"""Generation: build the prompt with context and call the local LLM (Ollama)."""
from __future__ import annotations

import ollama

SYSTEM_PROMPT = (
    "You are an assistant that answers ONLY based on the provided context. "
    "If the context does not contain the answer, say so clearly. "
    "Cite sources using [source: filename]."
)


def build_prompt(query: str, contexts: list[dict]) -> str:
    """Assemble the retrieved context and the question into a single prompt."""
    context_block = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in contexts
    )
    return f"CONTEXT:\n{context_block}\n\nQUESTION: {query}\n\nANSWER:"


def generate_answer(query: str, contexts: list[dict], model: str,
                    max_tokens: int, temperature: float) -> str:
    """Call the local Ollama model with the assembled prompt."""
    prompt = build_prompt(query, contexts)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response["message"]["content"]
