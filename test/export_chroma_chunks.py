import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_service.chroma_store import ChromaConfig, ChromaStore
from rag_service.config import RagConfig


def _safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _embedding_stats(emb: Any) -> dict[str, Any]:
    if not isinstance(emb, list) or not emb:
        return {"embedding_dim": 0, "embedding_l2_norm": None, "embedding_head": []}
    s2 = 0.0
    for x in emb:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        s2 += v * v
    return {
        "embedding_dim": len(emb),
        "embedding_l2_norm": math.sqrt(s2),
        "embedding_head": emb[:16],
    }


def export_chunks(limit: int | None = None, with_full_embedding: bool = False) -> Path:
    cfg = RagConfig()
    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    col = store._collection

    res = col.get(include=["documents", "metadatas", "embeddings"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    embs = res.get("embeddings") or []

    total = len(ids)
    if limit is not None:
        limit = max(0, int(limit))
        total = min(total, limit)

    out_root = Path(PROJECT_ROOT) / "test" / "chunk"
    chunks_dir = out_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    index_items: list[dict[str, Any]] = []
    exported = 0

    for i in range(total):
        _id = ids[i] if i < len(ids) else None
        doc = docs[i] if i < len(docs) else None
        md = metas[i] if i < len(metas) else None
        emb = embs[i] if i < len(embs) else None

        record: dict[str, Any] = {
            "row_index": i,
            "id": _id,
            "content": doc,
            "content_type": type(doc).__name__,
            "content_length": len(doc) if isinstance(doc, str) else None,
            "metadata": md if isinstance(md, dict) else md,
            "source": (md or {}).get("source", "") if isinstance(md, dict) else "",
            "source_path": (md or {}).get("source_path", "") if isinstance(md, dict) else "",
            "type": (md or {}).get("type", "") if isinstance(md, dict) else "",
        }
        record.update(_embedding_stats(emb))
        if with_full_embedding:
            record["embedding"] = emb

        chunk_file = chunks_dir / f"chunk_{i:06d}.json"
        _safe_json_dump(chunk_file, record)

        index_items.append(
            {
                "row_index": i,
                "id": _id,
                "chunk_file": str(chunk_file.relative_to(out_root)),
                "source": record["source"],
                "source_path": record["source_path"],
                "type": record["type"],
                "content_length": record["content_length"],
                "embedding_dim": record["embedding_dim"],
            }
        )
        exported += 1

    summary = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "collection_name": cfg.collection_name,
        "persist_directory": cfg.persist_directory,
        "total_in_collection": len(ids),
        "exported_count": exported,
        "with_full_embedding": with_full_embedding,
        "files": {
            "index": "index.json",
            "chunks_dir": "chunks/",
        },
    }
    _safe_json_dump(out_root / "summary.json", summary)
    _safe_json_dump(out_root / "index.json", index_items)
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 Chroma 向量库 chunk 明细到 test/chunk 目录。"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只导出前 N 条；不传则导出全部。",
    )
    parser.add_argument(
        "--with-full-embedding",
        action="store_true",
        help="是否在每个 chunk 文件中包含完整 embedding 向量（文件会很大）。",
    )
    args = parser.parse_args()

    out_root = export_chunks(
        limit=args.limit,
        with_full_embedding=args.with_full_embedding,
    )
    print(f"[ok] chunk 明细已导出到: {out_root}")
    print(f"[ok] 总览文件: {out_root / 'summary.json'}")
    print(f"[ok] 索引文件: {out_root / 'index.json'}")


if __name__ == "__main__":
    main()
