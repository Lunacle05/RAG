from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .loaders import LoadedDoc


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按段落优先切分，回到原始的简单分块方式。"""
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    # Guard against invalid configs that can cause non-progress loops.
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size - 1

    # Prefer paragraph boundaries first
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        s = "\n".join(cur).strip()
        if s:
            chunks.append(s)
        cur = []
        cur_len = 0

    for p in paras:
        if len(p) > chunk_size:
            flush()
            start = 0
            while start < len(p):
                end = min(len(p), start + chunk_size)
                chunks.append(p[start:end].strip())
                next_start = end - chunk_overlap if chunk_overlap > 0 else end
                # Ensure forward progress even under edge configs.
                start = next_start if next_start > start else end
                if start < 0:
                    start = 0
            continue

        if cur_len + len(p) + (1 if cur else 0) <= chunk_size:
            cur.append(p)
            cur_len += len(p) + (1 if cur else 0)
        else:
            flush()
            cur.append(p)
            cur_len = len(p)

    flush()

    # Add overlap between chunks (character-level), to help retrieval continuity
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    out: List[str] = []
    prev_tail = ""
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            head = prev_tail[-chunk_overlap:]
            merged = (head + "\n" + c).strip() if head else c
            out.append(merged)
        prev_tail = c
    return out


@dataclass(frozen=True)
class Chunk:
    content: str
    metadata: Dict


def chunk_docs(
    docs: Iterable[LoadedDoc],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = 0,
) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        # 图片类：不切分，单块保留；向量化在 embeddings 中按 image_path 处理。
        if d.metadata.get("type") == "image":
            md = dict(d.metadata)
            md["chunk_index"] = 0
            out.append(Chunk(content=d.text or f"[IMAGE] {md.get('source', '')}", metadata=md))
            continue
        # Excel：按行已拆，每行一块且带表头，不切分。
        if d.metadata.get("type") == "excel":
            md = dict(d.metadata)
            md["chunk_index"] = 0
            out.append(Chunk(content=(d.text or "").strip(), metadata=md))
            continue
        # Word 表格：整表一块（含表头），不切分，便于 LLM 理解列含义。
        if d.metadata.get("type") == "docx_table":
            md = dict(d.metadata)
            md["chunk_index"] = 0
            out.append(Chunk(content=(d.text or "").strip(), metadata=md))
            continue
        # PDF 表格：已按“表头 + 行”展开成多条，每条一块，不再切分。
        if d.metadata.get("type") == "pdf_table":
            md = dict(d.metadata)
            md["chunk_index"] = 0
            out.append(Chunk(content=(d.text or "").strip(), metadata=md))
            continue

        pieces = split_text(d.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if min_chunk_chars and len(piece) < min_chunk_chars:
                continue
            md = dict(d.metadata)
            md["chunk_index"] = idx
            out.append(Chunk(content=piece, metadata=md))
    return out

