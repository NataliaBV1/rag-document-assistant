"""Test the evaluation layer end-to-end."""
from rag_assistant.config import Config
from rag_assistant.evaluation import answer_relevancy, faithfulness
from rag_assistant.pipeline import RAGPipeline

cfg = Config.from_yaml("configs/config.yaml")
pipeline = RAGPipeline(cfg)

question = "How is protein subcellular localization determined?"
result = pipeline.answer(question)
answer = result["answer"]
context = "\n\n".join(c["text"] for c in result["contexts"])

print("\n=== ANSWER ===")
print(answer)

print("\n=== EVALUATION ===")
faith = faithfulness(context, answer, cfg.llm_model)
rel = answer_relevancy(question, answer, cfg.llm_model)
print(f"Faithfulness:     {faith:.2f}")
print(f"Answer relevancy: {rel:.2f}")