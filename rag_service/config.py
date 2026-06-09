from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# 默认优先使用环境变量，其次使用项目根目录下的本地多模态模型文件夹 `Qwen3-VL-Embedding-8B`
DEFAULT_MODEL_PATH = os.getenv("RAG_EMBED_MODEL") or str(BASE_DIR / "Qwen3-VL-Embedding-8B")
DEFAULT_RERANK_MODEL_PATH = os.getenv("RAG_VLLM_RERANK_MODEL") or str(BASE_DIR / "Qwen3-Reranker-0.6B-seq-cls")


@dataclass(frozen=True)
class RagConfig:
    persist_directory: str = os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "chroma_db"))
    collection_name: str = os.getenv("RAG_COLLECTION", "jwc_rag")

    # HTTP 服务配置：对外“接收请求”的地址和端口
    api_host: str = os.getenv("RAG_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("RAG_API_PORT", "6006"))

    # Embedding vLLM 访问配置（通过 OpenAI 兼容 /v1/embeddings）
    vllm_api_key: str = os.getenv("RAG_VLLM_API_KEY", "")
    vllm_timeout_s: float = float(os.getenv("RAG_VLLM_TIMEOUT_S", "600"))

    # 模型路径：默认指向项目根目录下的 Qwen3-Embedding（本地目录）
    # 已升级为 Qwen3-Embedding-VL，多模态（文本 / 图片）统一向量空间。
    model_path: str = DEFAULT_MODEL_PATH
    query_prompt_name: str = os.getenv("RAG_QUERY_PROMPT_NAME", "query")
    embed_backend: str = os.getenv("RAG_EMBED_BACKEND", "vllm").strip().lower()
    vllm_embed_url: str = os.getenv("RAG_VLLM_EMBED_URL", "http://127.0.0.1:6010/v1/embeddings")
    vllm_embed_model: str = os.getenv("RAG_VLLM_EMBED_MODEL", DEFAULT_MODEL_PATH)

    # Rerank vLLM 访问配置（通过 /v1/rerank）
    rerank_enabled: bool = os.getenv("RAG_RERANK_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    rerank_recall_k: int = int(os.getenv("RAG_RERANK_RECALL_K", "50"))
    rerank_instruction: str = os.getenv(
        "RAG_RERANK_INSTRUCTION",
        "给定高校问答查询，优先返回能直接回答问题的段落；若是FAQ，优先包含问句与答案原句的内容。",
    )
    vllm_rerank_url: str = os.getenv("RAG_VLLM_RERANK_URL", "http://127.0.0.1:6009/v1/rerank")
    vllm_rerank_model: str = DEFAULT_RERANK_MODEL_PATH
    vllm_rerank_timeout_s: float = float(os.getenv("RAG_VLLM_RERANK_TIMEOUT_S", "120"))
    rerank_fusion_weight: float = float(os.getenv("RAG_RERANK_FUSION_WEIGHT", "0.7"))
    vector_fusion_weight: float = float(os.getenv("RAG_VECTOR_FUSION_WEIGHT", "0.3"))
    min_retrieve_similarity: float = float(os.getenv("RAG_MIN_RETRIEVE_SIMILARITY", "0.30"))

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))

    # Basic hygiene: ignore tiny chunks
    min_chunk_chars: int = int(os.getenv("RAG_MIN_CHUNK_CHARS", "40"))

    # 入库时 embedding 批大小，显存紧张时改小（如 1 或 2）避免 OOM
    batch_size: int = int(os.getenv("RAG_BATCH_SIZE", "4"))

