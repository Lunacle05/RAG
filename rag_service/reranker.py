from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

_GLOBAL_RERANKER: "VLLMReranker | None" = None


@dataclass(frozen=True)
class RerankConfig:
    rerank_url: str
    model: str
    instruction: str
    timeout_s: float = 60.0
    top_n_default: Optional[int] = None


class VLLMReranker:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self._headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        self.prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n'
            '<|im_start|>user\n'
        )
        self.suffix = (
            '<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<think>\n\n</think>\n\n'
        )

    def _format_query(self, query: str) -> str:
        instruction = (self.cfg.instruction or "").strip()
        query = (query or "").strip()
        return f"{self.prefix}<Instruct>: {instruction}\n<Query>: {query}\n"

    def _format_document(self, document: str) -> str:
        document = (document or "").strip()
        return f"<Document>: {document}{self.suffix}"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        formatted_query = self._format_query(query)
        formatted_documents = [self._format_document(doc) for doc in documents]

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "query": formatted_query,
            "documents": formatted_documents,
        }

        if top_n is not None:
            payload["top_n"] = max(1, int(top_n))
        elif self.cfg.top_n_default is not None:
            payload["top_n"] = max(1, int(self.cfg.top_n_default))

        with httpx.Client(timeout=self.cfg.timeout_s) as client:
            resp = client.post(self.cfg.rerank_url, json=payload, headers=self._headers)
            resp.raise_for_status()
            body = resp.json()

        results = body.get("results") or []
        ranked: List[Dict[str, Any]] = []
        for item in results:
            ranked.append(
                {
                    "index": int(item["index"]),
                    "relevance_score": float(item["relevance_score"]),
                }
            )

        ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked

    def preload(self) -> None:
        """
        通过一次最小化 rerank 请求预热 vLLM rerank 路由，避免首个真实请求冷启动。
        """
        self.rerank(query="ping", documents=["ping"], top_n=1)


def get_global_reranker(cfg: RerankConfig) -> VLLMReranker:
    """
    获取进程内复用的全局 reranker，避免请求路径上重复构造对象。
    """
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        _GLOBAL_RERANKER = VLLMReranker(cfg)
    return _GLOBAL_RERANKER

