from __future__ import annotations

import argparse
import json

from .retrieve import retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="Manual RAG retrieval demo (no LLM needed).")
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--k", type=int, default=5, help="Number of chunks to return")
    args = p.parse_args()

    contexts = retrieve(query=args.query, k=args.k)
    print(json.dumps({"contexts": contexts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

