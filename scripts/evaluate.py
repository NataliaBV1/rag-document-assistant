"""Evaluate the RAG system over a set of questions.

Expected format for data/eval_qa.json:
[
  {"question": "...", "relevant_source": "paper1.pdf"},
  ...
]
Add "expected_answer" if you want to compare answers.
"""
import argparse
import json

from dotenv import load_dotenv

from rag_assistant.config import Config
from rag_assistant.evaluation import hit_rate_at_k, mrr
from rag_assistant.pipeline import RAGPipeline


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--qa", required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    pipeline = RAGPipeline(cfg)
    with open(args.qa, encoding="utf-8") as f:
        qa = json.load(f)

    hits, mrrs = [], []
    for item in qa:
        result = pipeline.answer(item["question"])
        sources = [c["source"] for c in result["contexts"]]
        hits.append(hit_rate_at_k(sources, item["relevant_source"]))
        mrrs.append(mrr(sources, item["relevant_source"]))
        # TODO: add faithfulness / answer_relevancy with LLM-as-judge.

    n = len(qa)
    print(f"Questions evaluated: {n}")
    print(f"Hit Rate@{cfg.top_k}: {sum(hits) / n:.3f}")
    print(f"MRR: {sum(mrrs) / n:.3f}")


if __name__ == "__main__":
    main()
