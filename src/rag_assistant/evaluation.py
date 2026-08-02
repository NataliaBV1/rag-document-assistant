"""RAG evaluation: retrieval and generation metrics (LLM-as-judge).

This is the layer that sets you apart. Start simple and keep it honest.
For production, consider RAGAS (https://github.com/explodinggradients/ragas).
"""
from __future__ import annotations


# ---------- Retrieval metrics (require ground-truth chunk) ----------
def hit_rate_at_k(retrieved_sources: list[str], relevant_source: str) -> float:
    """1.0 if the relevant document is among the retrieved ones, else 0.0."""
    return 1.0 if relevant_source in retrieved_sources else 0.0


def mrr(retrieved_sources: list[str], relevant_source: str) -> float:
    """Mean Reciprocal Rank for a single example: 1/rank of the first hit."""
    for rank, src in enumerate(retrieved_sources, start=1):
        if src == relevant_source:
            return 1.0 / rank
    return 0.0


# ---------- Generation metrics (LLM-as-judge) ----------
FAITHFULNESS_PROMPT = (
    "Given the CONTEXT and the ANSWER, is the answer supported solely by the "
    "context, without inventing information? Reply with a number between 0 and 1.\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nSCORE:"
)

RELEVANCY_PROMPT = (
    "Given the QUESTION and the ANSWER, does the answer directly address the "
    "question? Reply with a number between 0 and 1.\n\n"
    "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSCORE:"
)


def judge_score(prompt: str, model: str) -> float:
    """Call an LLM as a judge and parse a score in [0, 1].

    TODO: implement the call (reuse generation.generate_answer or a direct
    client) and parse the number. Handle parsing failures robustly.
    """
    raise NotImplementedError("Implement the LLM-judge call and score parsing.")


def faithfulness(context: str, answer: str, model: str) -> float:
    return judge_score(FAITHFULNESS_PROMPT.format(context=context, answer=answer), model)


def answer_relevancy(question: str, answer: str, model: str) -> float:
    return judge_score(RELEVANCY_PROMPT.format(question=question, answer=answer), model)
