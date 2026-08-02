"""Ask a question against the already-built index from the command line."""
import argparse

from dotenv import load_dotenv

from rag_assistant.config import Config
from rag_assistant.pipeline import RAGPipeline


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    pipeline = RAGPipeline(Config.from_yaml(args.config))
    result = pipeline.answer(args.question)
    print("\n=== ANSWER ===\n", result["answer"])
    print("\n=== SOURCES ===")
    for c in result["contexts"]:
        print(f"  [{c['source']}] score={c['score']:.3f}")


if __name__ == "__main__":
    main()
