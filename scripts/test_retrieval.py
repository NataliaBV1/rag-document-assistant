"""Quick test: retrieval only, no LLM needed. Asks questions interactively."""
from rag_assistant.config import Config
from rag_assistant.embeddings import Embedder
from rag_assistant.retrieval import Retriever
from rag_assistant.vectorstore import VectorStore

cfg = Config.from_yaml("configs/config.yaml")
embedder = Embedder(cfg.embedding_model, cfg.embedding_batch_size)
store = VectorStore.load(cfg.index_path, cfg.chunks_path)
retriever = Retriever(embedder, store, cfg.top_k)

print("\nReady. Type a question (or press Enter to quit).\n")
while True:
    question = input("Question: ").strip()
    if not question:
        break
    results = retriever.retrieve(question)
    print()
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score={r['score']:.3f}) ---")
        print(r["text"][:300], "...\n")