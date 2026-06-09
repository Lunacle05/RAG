from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

ProgressCallback = Callable[[Dict[str, Any]], None]

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: A001
        def __init__(self, total=0, desc="", unit="", dynamic_ncols=True, **kwargs):
            self.total = total
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass

from .chunking import chunk_docs
from .chroma_store import ChromaConfig, ChromaStore
from .config import RagConfig
from .embeddings import get_global_embedder
from .loaders import iter_files, load_one


BASE_DIR = Path(__file__).resolve().parent.parent
INGESTED_INDEX_PATH = BASE_DIR / "ingested_files.json"


def _make_id(
    source: str,
    source_path: str,
    page: int | None,
    chunk_index: int,
    doc_type: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    # 同一页既有文本 chunk 又有图片 chunk 时用 type 区分；同页多图用 image_path 哈希区分，避免 DuplicateIDError
    base = f"{source}|{source_path}"
    if page is not None:
        base += f"|p{page}"
    if doc_type:
        base += f"|{doc_type}"
    base += f"|c{chunk_index}"
    if doc_type == "image" and metadata and metadata.get("image_path"):
        path_hash = hashlib.md5(metadata["image_path"].encode()).hexdigest()[:12]
        base += f"|{path_hash}"
    if doc_type == "excel" and metadata:
        base += f"|{metadata.get('sheet', '')}|r{metadata.get('row_index', 0)}"
    if doc_type == "pdf_table" and metadata:
        base += f"|t{metadata.get('table_index', 0)}|r{metadata.get('row_index', 0)}"
    if doc_type == "docx_table" and metadata:
        base += f"|t{metadata.get('table_index', 0)}|r{metadata.get('row_index', 0)}"
    if doc_type in ("docx", "docx_table") and metadata and "section_index" in metadata:
        base += f"|s{metadata.get('section_index', 0)}"
    return base


def _load_ingested_paths() -> Set[str]:
    if not INGESTED_INDEX_PATH.exists():
        return set()
    try:
        with INGESTED_INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(p) for p in data}
    except Exception:
        return set()


def _save_ingested_paths(paths: Set[str]) -> None:
    with INGESTED_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(paths), f, ensure_ascii=False, indent=2)


def _emit_progress(
    on_progress: Optional[ProgressCallback],
    event: Dict[str, Any],
) -> None:
    if on_progress is not None:
        on_progress(event)


def ingest(
    input_path: str,
    reset: bool = False,
    cfg: RagConfig | None = None,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    cfg = cfg or RagConfig()
    source_name = os.path.basename(os.path.abspath(input_path))

    # 1) 计算本次要处理的文件列表（只处理“未入库”的新文件）
    all_files = [os.path.abspath(p) for p in iter_files(input_path)]
    ingested_paths = set() if reset else _load_ingested_paths()

    new_files = [p for p in all_files if p not in ingested_paths]
    if not new_files:
        return 0

    # 2) 仅加载这些新文件
    _emit_progress(
        on_progress,
        {
            "stage": "loading",
            "file": source_name,
            "message": f"正在解析 {source_name}...",
        },
    )
    print(f"[ingest] Loading {len(new_files)} file(s)...", flush=True)
    docs = []
    for fp in new_files:
        docs.extend(load_one(fp))
    _emit_progress(
        on_progress,
        {
            "stage": "chunking",
            "file": source_name,
            "message": f"正在分块 {source_name}...",
        },
    )
    print(f"[ingest] Chunking {len(docs)} doc(s)...", flush=True)
    chunks = chunk_docs(
        docs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chunk_chars=cfg.min_chunk_chars,
    )

    if not chunks:
        return 0

    _emit_progress(
        on_progress,
        {
            "stage": "chunking",
            "file": source_name,
            "done": len(chunks),
            "total": len(chunks),
            "message": f"分块完成，共 {len(chunks)} 个 chunk",
        },
    )

    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    if reset:
        store.reset()

    if cfg.embed_backend == "vllm":
        print(
            f"[ingest] Using vLLM embeddings at {cfg.vllm_embed_url} (model={cfg.vllm_embed_model})...",
            flush=True,
        )
    else:
        print(f"[ingest] Loading embedding model (first run may take 1–2 min)...", flush=True)
    embedder = get_global_embedder(
        model_path=cfg.model_path,
        query_prompt_name=cfg.query_prompt_name,
    )
    texts = [c.content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    print(f"[ingest] Embedding {len(chunks)} chunk(s) in batches of {cfg.batch_size}...", flush=True)
    embs: List[List[float]] = []
    batch_size = max(1, int(cfg.batch_size))
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        _emit_progress(
            on_progress,
            {
                "stage": "embedding",
                "file": source_name,
                "done": start,
                "total": len(texts),
                "message": f"向量化中 {start}/{len(texts)}...",
            },
        )
        embs.extend(
            embedder.embed_documents(
                texts[start:end],
                metadatas=metadatas[start:end],
                batch_size=batch_size,
            )
        )
        _emit_progress(
            on_progress,
            {
                "stage": "embedding",
                "file": source_name,
                "done": end,
                "total": len(texts),
                "message": f"向量化中 {end}/{len(texts)}...",
            },
        )
    print("[ingest] Writing to Chroma...", flush=True)

    ids: List[str] = []
    metadatas = []
    for c in chunks:
        md = dict(c.metadata)
        doc_type = str(md.get("type", "")).strip()
        ids.append(
            _make_id(
                source=str(md.get("source", "")),
                source_path=str(md.get("source_path", "")),
                page=int(md["page"]) if "page" in md and md["page"] is not None else None,
                chunk_index=int(md.get("chunk_index", 0)),
                doc_type=doc_type,
                metadata=md,
            )
        )
        metadatas.append(md)

    # 分批写入 Chroma 并显示进度条
    batch_size_db = max(1, min(200, len(chunks)))
    with tqdm(total=len(chunks), desc="存入向量库", unit="chunk", dynamic_ncols=True) as pbar:
        for start in range(0, len(chunks), batch_size_db):
            end = min(start + batch_size_db, len(chunks))
            _emit_progress(
                on_progress,
                {
                    "stage": "writing",
                    "file": source_name,
                    "done": start,
                    "total": len(chunks),
                    "message": f"写入向量库 {start}/{len(chunks)}...",
                },
            )
            store.add_texts(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embs[start:end],
                metadatas=metadatas[start:end],
            )
            pbar.update(end - start)
            _emit_progress(
                on_progress,
                {
                    "stage": "writing",
                    "file": source_name,
                    "done": end,
                    "total": len(chunks),
                    "message": f"写入向量库 {end}/{len(chunks)}...",
                },
            )

    # 记录已经入库过的文件路径，后续再次运行时会跳过这些文件，实现增量入库
    ingested_paths.update(new_files)
    _save_ingested_paths(ingested_paths)

    return len(chunks)


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest files into Chroma for RAG.")
    p.add_argument("--input", required=True, help="File or directory path containing pdf/docx/txt/md")
    p.add_argument("--reset", action="store_true", help="Drop and recreate the Chroma collection before ingest")
    args = p.parse_args()

    n = ingest(args.input, reset=args.reset)
    print(f"Ingested chunks: {n}")


if __name__ == "__main__":
    main()

