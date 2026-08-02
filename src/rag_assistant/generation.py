"""Generation: build the prompt with context and call the LLM."""
from __future__ import annotations

import os

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
    """Call the LLM with the assembled prompt. Skeleton for Anthropic.

    TODO: if you use OpenAI or another provider, abstract this behind an interface.
    """
    import anthropic  # local import so tests don't require the dependency

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_prompt(query, contexts)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
