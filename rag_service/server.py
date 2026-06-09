from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chroma_store import ChromaConfig, ChromaStore
from .config import RagConfig
from .embeddings import get_global_embedder
from .ingest import _load_ingested_paths, _save_ingested_paths, ingest
from .loaders import SUPPORTED_EXTS
from .reranker import RerankConfig, get_global_reranker
from .retrieve import retrieve


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User query text")
    k: int = Field(5, ge=1, le=50, description="Number of chunks to return")


class ContextItem(BaseModel):
    content: str = Field(..., description="检索到的文档片段内容")
    similarity: float = Field(
        ...,
        description="最终排序分数。启用 rerank 时为 rerank_score，否则为 vector_similarity。",
    )
    source: str = Field(..., description="来源文件名，例如 某某通知.pdf")
    vector_similarity: float | None = Field(
        None,
        description="向量召回阶段的相似度分数。",
    )


class RetrieveResponse(BaseModel):
    contexts: list[ContextItem]


class DocumentItem(BaseModel):
    source: str = Field(..., description="文件名")
    source_path: str = Field(..., description="文件绝对路径")
    chunk_count: int = Field(..., description="该文档在向量库中的 chunk 数")


class DocumentsResponse(BaseModel):
    items: list[DocumentItem]


class UploadResponse(BaseModel):
    uploaded_files: list[str] = Field(default_factory=list, description="成功上传并入库的文件名")
    ingested_chunks: int = Field(0, description="总入库 chunk 数")
    skipped_files: list[str] = Field(default_factory=list, description="被跳过的文件名")
    errors: list[str] = Field(default_factory=list, description="失败明细")


class DeleteRequest(BaseModel):
    source_paths: list[str] = Field(..., description="需要删除的 source_path 列表")
    remove_files: bool = Field(False, description="是否同步删除磁盘文件")


class DeleteResponse(BaseModel):
    deleted_chunks: int = Field(..., description="删除的向量条数")
    deleted_files: list[str] = Field(default_factory=list, description="同步删除的磁盘文件")
    not_found_files: list[str] = Field(default_factory=list, description="磁盘上不存在的文件")


cfg = RagConfig()
BASE_DIR = Path(__file__).resolve().parent.parent
ADMIN_UPLOAD_DIR = BASE_DIR / "uploaded_docs"
UPLOAD_CATEGORY_DIRS = {
    "frequent": ADMIN_UPLOAD_DIR / "frequent",
    "infrequent": ADMIN_UPLOAD_DIR / "infrequent",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.stats = {
        "started_at": int(time.time()),
        "inflight_requests": 0,
        "total_requests": 0,
        "success_requests": 0,
        "error_requests": 0,
    }
    app.state.stats_lock = asyncio.Lock()

    # 服务启动时预加载 embedding 模型，避免首个请求触发冷启动
    get_global_embedder(
        model_path=cfg.model_path,
        query_prompt_name=cfg.query_prompt_name,
    ).preload()
    if cfg.rerank_enabled:
        try:
            get_global_reranker(
                RerankConfig(
                    rerank_url=cfg.vllm_rerank_url,
                    model=cfg.vllm_rerank_model,
                    instruction=cfg.rerank_instruction,
                    timeout_s=cfg.vllm_rerank_timeout_s,
                    top_n_default=None,
                )
            ).preload()
        except Exception as exc:
            print(
                f"[startup] warning: reranker preload failed, service continues without startup warmup: {exc}",
                flush=True,
            )

    try:
        yield
    finally:
        pass


app = FastAPI(title="教务处RAG检索服务", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    stats = request.app.state.stats
    lock = request.app.state.stats_lock
    async with lock:
        stats["total_requests"] += 1
        stats["inflight_requests"] += 1

    try:
        response = await call_next(request)
    except Exception:
        async with lock:
            stats["error_requests"] += 1
            stats["inflight_requests"] = max(0, stats["inflight_requests"] - 1)
        raise

    async with lock:
        if response.status_code >= 400:
            stats["error_requests"] += 1
        else:
            stats["success_requests"] += 1
        stats["inflight_requests"] = max(0, stats["inflight_requests"] - 1)
    return response


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
async def stats(request: Request):
    raw = request.app.state.stats
    lock = request.app.state.stats_lock
    async with lock:
        snapshot = dict(raw)
    snapshot["uptime_seconds"] = max(0, int(time.time()) - int(snapshot["started_at"]))
    return snapshot


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_api(req: RetrieveRequest):
    contexts = retrieve(query=req.query, k=req.k)
    return {"contexts": contexts}


@app.get("/admin/documents", response_model=DocumentsResponse)
def list_documents():
    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    return {"items": store.list_documents()}


def _progress_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _ingest_file_with_progress(
    *,
    dest_path: Path,
    filename: str,
    file_index: int,
    file_total: int,
    store: ChromaStore,
    ingested_paths: set[str],
    progress_queue: queue.Queue[dict[str, Any] | None],
) -> tuple[bool, int, str | None]:
    def on_progress(event: dict[str, Any]) -> None:
        progress_queue.put(
            {
                **event,
                "file_index": file_index,
                "file_total": file_total,
            }
        )

    try:
        progress_queue.put(
            {
                "stage": "saving",
                "file": filename,
                "file_index": file_index,
                "file_total": file_total,
                "message": f"准备入库 {filename} ({file_index}/{file_total})...",
            }
        )
        store.delete_by_source_paths([str(dest_path)])

        # 重传同名文件时必须同步清理增量索引，否则 ingest 会误判为已入库而跳过
        current_paths = _load_ingested_paths()
        if str(dest_path) in current_paths:
            current_paths.discard(str(dest_path))
            _save_ingested_paths(current_paths)
        if str(dest_path) in ingested_paths:
            ingested_paths.discard(str(dest_path))

        chunks = ingest(
            input_path=str(dest_path),
            reset=False,
            cfg=cfg,
            on_progress=on_progress,
        )
        if int(chunks) <= 0:
            progress_queue.put(
                {
                    "stage": "error",
                    "file": filename,
                    "file_index": file_index,
                    "file_total": file_total,
                    "message": "入库未产生 chunk，文件可能为空或解析失败",
                }
            )
            return False, 0, "入库未产生 chunk"

        progress_queue.put(
            {
                "stage": "file_done",
                "file": filename,
                "file_index": file_index,
                "file_total": file_total,
                "chunks": int(chunks),
                "message": f"{filename} 入库完成，新增 {chunks} 个 chunk",
            }
        )
        return True, int(chunks), None
    except Exception as exc:
        progress_queue.put(
            {
                "stage": "error",
                "file": filename,
                "file_index": file_index,
                "file_total": file_total,
                "message": str(exc),
            }
        )
        return False, 0, str(exc)


@app.post("/admin/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    category: str = Form("frequent"),
):
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件")

    category_key = str(category or "").strip().lower()
    if category_key not in UPLOAD_CATEGORY_DIRS:
        raise HTTPException(
            status_code=400,
            detail="category 必须是 frequent 或 infrequent",
        )

    upload_dir = UPLOAD_CATEGORY_DIRS[category_key]
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files: list[str] = []
    skipped_files: list[str] = []
    errors: list[str] = []
    ingested_chunks = 0

    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    ingested_paths = _load_ingested_paths()

    for f in files:
        filename = os.path.basename(f.filename or "").strip()
        if not filename:
            skipped_files.append("(empty filename)")
            continue

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            skipped_files.append(filename)
            continue

        dest_path = (upload_dir / filename).resolve()
        try:
            content = await f.read()
            with dest_path.open("wb") as out:
                out.write(content)

            # 同名文件重传时，先删旧向量 + 清理增量索引，再重新入库
            store.delete_by_source_paths([str(dest_path)])
            if str(dest_path) in ingested_paths:
                ingested_paths.remove(str(dest_path))
                _save_ingested_paths(ingested_paths)

            chunks = ingest(input_path=str(dest_path), reset=False, cfg=cfg)
            ingested_chunks += int(chunks)
            uploaded_files.append(filename)
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
        finally:
            await f.close()

    return {
        "uploaded_files": uploaded_files,
        "ingested_chunks": ingested_chunks,
        "skipped_files": skipped_files,
        "errors": errors,
    }


@app.post("/admin/upload-stream")
async def upload_documents_stream(
    files: list[UploadFile] = File(...),
    category: str = Form("frequent"),
):
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件")

    category_key = str(category or "").strip().lower()
    if category_key not in UPLOAD_CATEGORY_DIRS:
        raise HTTPException(
            status_code=400,
            detail="category 必须是 frequent 或 infrequent",
        )

    upload_dir = UPLOAD_CATEGORY_DIRS[category_key]
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[tuple[str, Path]] = []
    skipped_files: list[str] = []

    for f in files:
        filename = os.path.basename(f.filename or "").strip()
        if not filename:
            skipped_files.append("(empty filename)")
            await f.close()
            continue

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            skipped_files.append(filename)
            await f.close()
            continue

        dest_path = (upload_dir / filename).resolve()
        try:
            content = await f.read()
            with dest_path.open("wb") as out:
                out.write(content)
            saved_files.append((filename, dest_path))
        except Exception as exc:
            skipped_files.append(f"{filename}: {exc}")
        finally:
            await f.close()

    def event_stream() -> Iterator[str]:
        uploaded_files: list[str] = []
        errors: list[str] = []
        ingested_chunks = 0

        if not saved_files:
            yield _progress_event(
                {
                    "stage": "complete",
                    "message": "没有可入库的文件",
                    "result": {
                        "uploaded_files": [],
                        "ingested_chunks": 0,
                        "skipped_files": skipped_files,
                        "errors": [],
                    },
                }
            )
            return

        store = ChromaStore(
            ChromaConfig(
                persist_directory=cfg.persist_directory,
                collection_name=cfg.collection_name,
            )
        )
        ingested_paths = set(_load_ingested_paths())
        file_total = len(saved_files)

        yield _progress_event(
            {
                "stage": "batch_start",
                "file_total": file_total,
                "message": f"开始入库 {file_total} 个文件...",
            }
        )

        for file_index, (filename, dest_path) in enumerate(saved_files, start=1):
            progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()

            def _worker(
                dest_path: Path = dest_path,
                filename: str = filename,
                file_index: int = file_index,
            ) -> None:
                _ingest_file_with_progress(
                    dest_path=dest_path,
                    filename=filename,
                    file_index=file_index,
                    file_total=file_total,
                    store=store,
                    ingested_paths=ingested_paths,
                    progress_queue=progress_queue,
                )

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            while worker.is_alive() or not progress_queue.empty():
                try:
                    event = progress_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                yield _progress_event(event)

                if event.get("stage") == "file_done":
                    uploaded_files.append(filename)
                    ingested_chunks += int(event.get("chunks", 0))
                elif event.get("stage") == "error":
                    errors.append(f"{filename}: {event.get('message', '')}")

            worker.join()

        yield _progress_event(
            {
                "stage": "complete",
                "message": "全部处理完成",
                "result": {
                    "uploaded_files": uploaded_files,
                    "ingested_chunks": ingested_chunks,
                    "skipped_files": skipped_files,
                    "errors": errors,
                },
            }
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/admin/delete", response_model=DeleteResponse)
def delete_documents(req: DeleteRequest):
    source_paths = [str(p).strip() for p in req.source_paths if str(p).strip()]
    if not source_paths:
        raise HTTPException(status_code=400, detail="source_paths 不能为空")

    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    deleted_chunks = store.delete_by_source_paths(source_paths)

    # 同步更新“已入库路径索引”，避免后续增量逻辑误判
    ingested_paths = _load_ingested_paths()
    for p in source_paths:
        if p in ingested_paths:
            ingested_paths.remove(p)
    _save_ingested_paths(ingested_paths)

    deleted_files: list[str] = []
    not_found_files: list[str] = []

    if req.remove_files:
        for p in source_paths:
            try:
                if os.path.exists(p) and os.path.isfile(p):
                    os.remove(p)
                    deleted_files.append(p)
                else:
                    not_found_files.append(p)
            except Exception:
                not_found_files.append(p)

    return {
        "deleted_chunks": deleted_chunks,
        "deleted_files": deleted_files,
        "not_found_files": not_found_files,
    }
