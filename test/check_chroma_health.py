from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_service.chroma_store import ChromaConfig, ChromaStore
from rag_service.config import RagConfig


def _load_ingested_paths(index_path: Path) -> list[str]:
    if not index_path.exists():
        return []
    try:
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x) for x in data]
    except Exception as exc:
        return [f"<load-error: {exc}>"]


def _dir_summary(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False}

    file_count = 0
    total_size = 0
    largest: list[tuple[int, str]] = []
    suffix_counter: Counter[str] = Counter()

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        file_count += 1
        total_size += size
        suffix_counter[p.suffix.lower() or "<no_suffix>"] += 1
        largest.append((size, str(p)))

    largest.sort(reverse=True)
    return {
        "exists": True,
        "file_count": file_count,
        "total_size_bytes": total_size,
        "top_suffixes": suffix_counter.most_common(10),
        "largest_files": [
            {"size_bytes": size, "path": path} for size, path in largest[:10]
        ],
    }


def main() -> None:
    cfg = RagConfig()
    db_dir = Path(cfg.persist_directory).resolve()
    ingested_index = Path(PROJECT_ROOT) / "ingested_files.json"

    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    col = store._collection
    res = col.get(include=["documents", "metadatas"])
    ids = [str(x) for x in (res.get("ids") or [])]
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []

    id_counter = Counter(ids)
    duplicate_ids = [item for item, count in id_counter.items() if count > 1]

    path_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    id_samples_by_path: dict[str, list[str]] = defaultdict(list)

    for _id, md in zip(ids, metas):
        if not isinstance(md, dict):
            continue
        source_path = str(md.get("source_path", "")).strip()
        source = str(md.get("source", "")).strip()
        doc_type = str(md.get("type", "")).strip()
        if source_path:
            path_counter[source_path] += 1
            if len(id_samples_by_path[source_path]) < 5:
                id_samples_by_path[source_path].append(_id)
        if source:
            source_counter[source] += 1
        if doc_type:
            type_counter[doc_type] += 1

    ingested_paths = _load_ingested_paths(ingested_index)
    ingested_set = {p for p in ingested_paths if not p.startswith("<load-error:")}
    collection_path_set = set(path_counter.keys())
    missing_on_disk = [p for p in ingested_set if not Path(p).exists()]

    payload = {
        "collection": {
            "name": cfg.collection_name,
            "persist_directory": str(db_dir),
            "total_rows": len(ids),
            "unique_ids": len(id_counter),
            "duplicate_id_count": len(duplicate_ids),
            "duplicate_id_examples": duplicate_ids[:20],
            "empty_document_count": sum(
                1 for doc in docs if not isinstance(doc, str) or not doc.strip()
            ),
            "top_source_paths": path_counter.most_common(20),
            "top_sources": source_counter.most_common(20),
            "top_types": type_counter.most_common(20),
            "path_id_samples": {
                path: id_samples_by_path[path]
                for path, _count in path_counter.most_common(10)
            },
        },
        "ingested_index": {
            "path": str(ingested_index),
            "count": len(ingested_paths),
            "missing_on_disk_count": len(missing_on_disk),
            "missing_on_disk_examples": missing_on_disk[:20],
            "paths_only_in_ingested": sorted(ingested_set - collection_path_set)[:50],
            "paths_only_in_collection": sorted(collection_path_set - ingested_set)[:50],
            "all_entries": ingested_paths,
        },
        "storage": _dir_summary(db_dir),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
