"""Build the vector index from the documents in data/raw."""
import argparse

from rag_assistant.config import Config
from rag_assistant.pipeline import ingest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    ingest(Config.from_yaml(args.config))


if __name__ == "__main__":
    main()
