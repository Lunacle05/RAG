from __future__ import annotations

from typing import Any, Dict, List, Optional

from .chroma_store import ChromaConfig, ChromaStore
from .config import RagConfig
from .embeddings import get_global_embedder
from .reranker import RerankConfig, VLLMReranker, get_global_reranker


def _distance_to_similarity(dist: Any) -> float:
    # With cosine space and normalized embeddings, distance ~= 1 - cosine_similarity
    try:
        sim = 1.0 - float(dist)
    except Exception:
        sim = 0.0

    if sim > 1.0:
        sim = 1.0
    if sim < -1.0:
        sim = -1.0
    return sim


def _build_contexts(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs0 = (res.get("documents") or [[]])[0]
    metas0 = (res.get("metadatas") or [[]])[0]
    dists0 = (res.get("distances") or [[]])[0]

    contexts: List[Dict[str, Any]] = []
    for doc, md, dist in zip(docs0, metas0, dists0):
        vector_similarity = _distance_to_similarity(dist)
        source = ""
        if isinstance(md, dict):
            source = str(md.get("source", ""))

        contexts.append(
            {
                "content": str(doc),
                "similarity": vector_similarity,          # 默认先等于向量相似度
                "vector_similarity": vector_similarity,   # 保留原始向量召回分数
                "rerank_score": None,                     # rerank 后再填
                "source": source,
            }
        )
    return contexts


def _make_reranker(cfg: RagConfig) -> VLLMReranker:
    return get_global_reranker(
        RerankConfig(
            rerank_url=cfg.vllm_rerank_url,
            model=cfg.vllm_rerank_model,
            instruction=cfg.rerank_instruction,
            timeout_s=cfg.vllm_rerank_timeout_s,
            top_n_default=None,
        )
    )


def _filter_contexts_by_similarity(
    contexts: List[Dict[str, Any]],
    k: int,
    min_similarity: float,
) -> List[Dict[str, Any]]:
    filtered = [
        ctx
        for ctx in contexts
        if float(ctx.get("similarity", 0.0) or 0.0) >= min_similarity
    ]
    return filtered[:k]


def _apply_rerank(
    query: str,
    contexts: List[Dict[str, Any]],
    k: int,
    reranker: VLLMReranker,
    rerank_weight: float,
    vector_weight: float,
) -> List[Dict[str, Any]]:
    ranked = reranker.rerank(
        query=query,
        documents=[ctx["content"] for ctx in contexts],
        top_n=len(contexts),
    )

    if not ranked:
        return contexts[:k]

    out: List[Dict[str, Any]] = []
    for item in ranked:
        idx = int(item["index"])
        rerank_score = float(item["relevance_score"])

        row = dict(contexts[idx])
        vector_similarity = float(row.get("vector_similarity", 0.0) or 0.0)
        final_score = rerank_weight * rerank_score + vector_weight * vector_similarity
        row["rerank_score"] = rerank_score
        row["similarity"] = final_score  # 对外返回的最终排序分数（融合打分）
        out.append(row)

    out.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    return out[:k]


def retrieve(
    query: str,
    k: int,
    cfg: RagConfig | None = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    cfg = cfg or RagConfig()
    k = max(1, int(k))

    recall_k = k
    if cfg.rerank_enabled:
        recall_k = max(k, int(cfg.rerank_recall_k))

    # 使用进程级全局 embedder，避免在服务模式下重复加载模型
    embedder = get_global_embedder(
        model_path=cfg.model_path,
        query_prompt_name=cfg.query_prompt_name,
    )
    qvec = embedder.embed_query(query)

    store = ChromaStore(
        ChromaConfig(
            persist_directory=cfg.persist_directory,
            collection_name=cfg.collection_name,
        )
    )
    res = store.query(query_embedding=qvec, n_results=recall_k, where=where)
    contexts = _build_contexts(res)

    if not contexts:
        return []

    if cfg.rerank_enabled:
        try:
            rerank_weight = max(0.0, float(cfg.rerank_fusion_weight))
            vector_weight = max(0.0, float(cfg.vector_fusion_weight))
            total_weight = rerank_weight + vector_weight
            if total_weight <= 0:
                rerank_weight, vector_weight = 0.7, 0.3
            else:
                rerank_weight /= total_weight
                vector_weight /= total_weight

            reranker = _make_reranker(cfg)
            reranked_contexts = _apply_rerank(
                query=query,
                contexts=contexts,
                k=k,
                reranker=reranker,
                rerank_weight=rerank_weight,
                vector_weight=vector_weight,
            )
            return _filter_contexts_by_similarity(
                reranked_contexts,
                k=k,
                min_similarity=cfg.min_retrieve_similarity,
            )
        except Exception as exc:
            # rerank 临时异常时，服务不直接挂，回退到向量召回
            print(f"[retrieve] rerank failed, fallback to vector recall: {exc}", flush=True)

    return _filter_contexts_by_similarity(
        contexts,
        k=k,
        min_similarity=cfg.min_retrieve_similarity,
    )


