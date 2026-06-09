from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

# Compatibility shim for chromadb 0.4.x on NumPy 2.x.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint"):
    np.uint = np.uint64

import chromadb
from chromadb.config import Settings


@dataclass(frozen=True)
class ChromaConfig:
    persist_directory: str
    collection_name: str


class ChromaStore:
    def __init__(self, cfg: ChromaConfig):
        self.cfg = cfg
        self._client = chromadb.PersistentClient(
            path=cfg.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        # Drop and recreate the collection
        self._client.delete_collection(self.cfg.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._collection.add(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(e) for e in embeddings],
            metadatas=[dict(m) for m in metadatas],
        )

    def query(
        self,
        query_embedding: Sequence[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """
        汇总当前 collection 中按 source_path 去重后的“文档级”信息。
        """
        data = self._collection.get(
            include=["metadatas"],
        )
        metadatas = data.get("metadatas") or []
        counter: Counter[str] = Counter()
        source_name: dict[str, str] = {}

        for md in metadatas:
            if not isinstance(md, dict):
                continue
            source_path = str(md.get("source_path", "")).strip()
            if not source_path:
                continue
            counter[source_path] += 1
            if source_path not in source_name:
                source_name[source_path] = str(md.get("source", "")).strip()

        items: list[dict[str, Any]] = []
        for source_path, chunk_count in counter.items():
            items.append(
                {
                    "source_path": source_path,
                    "source": source_name.get(source_path, ""),
                    "chunk_count": int(chunk_count),
                }
            )
        items.sort(key=lambda x: (x.get("source", ""), x.get("source_path", "")))
        return items

    def delete_by_source_paths(self, source_paths: Sequence[str]) -> int:
        """
        根据 source_path 批量删除对应向量，返回删除条数（按 id 计）。
        """
        normalized = [str(p).strip() for p in source_paths if str(p).strip()]
        if not normalized:
            return 0

        data = self._collection.get(
            include=["metadatas"],
        )
        ids = data.get("ids") or []
        metadatas = data.get("metadatas") or []
        source_path_set = set(normalized)

        to_delete_ids: list[str] = []
        for _id, md in zip(ids, metadatas):
            if not isinstance(md, dict):
                continue
            if str(md.get("source_path", "")).strip() in source_path_set:
                to_delete_ids.append(str(_id))

        if not to_delete_ids:
            return 0

        self._collection.delete(ids=to_delete_ids)
        return len(to_delete_ids)

