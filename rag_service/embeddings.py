from __future__ import annotations

import base64
import importlib.util
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx
import numpy as np

# 进程级全局 embedder，用于避免在服务模式下重复加载模型
_GLOBAL_EMBEDDER: "SentenceTransformerEmbedder | VLLMEmbedder | None" = None

BASE_DIR = Path(__file__).resolve().parent.parent
_QWEN_VL_SCRIPT = BASE_DIR / "Qwen3-VL-Embedding-8B" / "scripts" / "qwen3_vl_embedding.py"


def _default_device() -> str:
    """
    设备选择：优先使用环境变量 RAG_DEVICE（如 cuda / cuda:0 / cpu），
    未设置时若 CUDA 可用则用 cuda，否则用 cpu。入库和检索都会使用该设备。
    """
    env_device = os.getenv("RAG_DEVICE")
    if env_device:
        return env_device.strip()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_qwen_vl_class():
    """
    动态从本仓库的 Qwen3-VL-Embedding-8B 脚本中加载 Qwen3VLEmbedder 类。
    这样可以避免目录名中带 `-` 导致无法正常作为包导入的问题。
    """
    if not _QWEN_VL_SCRIPT.exists():
        raise RuntimeError(f"Qwen3-VL 脚本不存在：{_QWEN_VL_SCRIPT}")

    module_name = "qwen3_vl_embedding_local"
    spec = importlib.util.spec_from_file_location(module_name, _QWEN_VL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法为 Qwen3-VL 加载 spec：{_QWEN_VL_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    # 确保 transformers 能在 sys.modules 中通过 __module__ 找到该模块
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "Qwen3VLEmbedder"):
        raise RuntimeError("在 qwen3_vl_embedding.py 中未找到 Qwen3VLEmbedder 类")
    return module.Qwen3VLEmbedder


def _normalize_l2(vecs: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    统一做 L2 归一化，保证 Chroma cosine 检索时分布更稳定。
    """
    if not vecs:
        return []
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32).tolist()


def _image_file_to_data_url(path: str) -> str:
    """
    将本地图片文件转为 data URL，便于通过 vLLM OpenAI 兼容接口传入 image_url。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"图片文件不存在：{path}")

    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"

    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@dataclass
class EmbeddingConfig:
    model_path: str
    query_prompt_name: Optional[str] = "query"
    device: str = _default_device()


class SentenceTransformerEmbedder:
    """
    名称保留为 SentenceTransformerEmbedder，以减少对现有调用方的影响，
    但内部已经替换为使用 Qwen3-Embedding-VL（Qwen3VLEmbedder）实现。
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._backend = None  # 懒加载 Qwen3VLEmbedder 实例

    def _load(self):
        if self._backend is None:
            import torch

            Qwen3VLEmbedder = _load_qwen_vl_class()
            self._backend = Qwen3VLEmbedder(self.cfg.model_path)
            # 按 RAG_DEVICE 或默认策略将模型放到指定设备（入库/检索均使用 GPU 时显存更快）
            device = torch.device(self.cfg.device)
            self._backend.model.to(device)
        return self._backend

    def preload(self) -> None:
        """显式预加载模型，供服务启动阶段调用。"""
        self._load()

    def _to_numpy(self, tensor) -> np.ndarray:
        import torch

        # Qwen3-VL 可能返回 bfloat16 / float16，需要先安全地转成 float32 再转 numpy
        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float32)

        if hasattr(tensor, "detach"):
            t = tensor.detach()
            if isinstance(t, torch.Tensor):
                if t.dtype != torch.float32:
                    t = t.to(dtype=torch.float32)
                return t.cpu().numpy().astype(np.float32)
            return np.asarray(t, dtype=np.float32)

        return np.asarray(tensor, dtype=np.float32)

    def embed_documents(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        metadatas: Optional[Sequence[Dict]] = None,
    ) -> List[List[float]]:
        """
        使用 Qwen3-Embedding-VL 对文档进行向量化，发挥 VL 图文同一向量空间的优势。

        - 普通文本：走文本编码；
        - 图片类（type==image 且 image_path）：走图像编码，与 query 文本在同一 VL 空间，
          用文字搜索即可通过图文对齐匹配到对应图片；存库的 document 仍为 OCR / 上下文文字。
        """
        backend = self._load()

        texts_list = list(texts)
        if metadatas is None:
            metadatas = [None] * len(texts_list)
        else:
            metadatas = list(metadatas)

        if len(texts_list) != len(metadatas):
            raise ValueError("texts 与 metadatas 长度不一致")

        inputs_batch: List[Dict[str, Any]] = []
        for txt, md in zip(texts_list, metadatas):
            md = md or {}
            if md.get("type") == "image" and md.get("image_path"):
                inputs_batch.append({"text": None, "image": md["image_path"]})
            else:
                inputs_batch.append({"text": (txt or "").strip(), "image": None})

        # 分批送入 Qwen3-VL，避免一次性处理过多样本导致显存溢出
        all_vecs: List[List[float]] = []
        bsz = max(1, int(batch_size))
        total_batches = (len(inputs_batch) + bsz - 1) // bsz
        for batch_idx, start in enumerate(range(0, len(inputs_batch), bsz), 1):
            end = min(start + bsz, len(inputs_batch))
            print(
                f"[ingest]   batch {batch_idx}/{total_batches} ({start + 1}-{end}/{len(inputs_batch)})",
                flush=True,
            )
            batch_inputs = inputs_batch[start:end]
            embeddings_tensor = backend.process(batch_inputs, normalize=True)
            arr = self._to_numpy(embeddings_tensor)
            all_vecs.extend(arr.tolist())
        return all_vecs

    def embed_query(self, query: str) -> List[float]:
        """
        查询统一走文本通路，保证与文档向量在同一向量空间中。
        如后续需要支持“以图搜文”，可以在上层调用中扩展一个 image_query 接口。
        """
        backend = self._load()
        inputs = [{"text": query, "image": None}]
        embeddings_tensor = backend.process(inputs, normalize=True)
        arr = self._to_numpy(embeddings_tensor)[0]
        return arr.tolist()


class VLLMEmbedder:
    """
    通过 OpenAI 兼容的 /v1/embeddings 调用 vLLM。

    统一策略：
    - 纯文本：使用 input
    - 图片文档（type=image 且 image_path 存在）：使用 messages 多模态输入
    """

    def __init__(
        self,
        embed_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 600.0,
    ):
        self.embed_url = embed_url
        self.model = model
        self.timeout_s = timeout_s
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"

    def preload(self) -> None:
        # 用一次极轻量请求预热 vLLM embedding 路由
        self.embed_query("ping")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(self.embed_url, json=payload, headers=self.headers)
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") or []
        return [item.get("embedding", []) for item in data]

    def _embed_multimodal_messages(
        self,
        messages_batch: List[List[Dict[str, Any]]],
    ) -> List[List[float]]:
        """
        对多模态样本逐条调用 vLLM /v1/embeddings。
        每条 messages 对应一个 embedding。
        """
        out: List[List[float]] = []
        with httpx.Client(timeout=self.timeout_s) as client:
            for messages in messages_batch:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "encoding_format": "float",
                }
                resp = client.post(self.embed_url, json=payload, headers=self.headers)
                resp.raise_for_status()
                body = resp.json()
                data = body.get("data") or []
                if not data:
                    raise RuntimeError("vLLM multimodal embeddings 返回为空")
                out.append(data[0].get("embedding", []))
        return out

    def embed_documents(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        metadatas: Optional[Sequence[Dict]] = None,
    ) -> List[List[float]]:
        """
        文档入库统一走 vLLM：
        - 普通文本：/v1/embeddings + input
        - 图片文档：/v1/embeddings + messages(image_url + text)
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        if metadatas is None:
            metadatas = [None] * len(texts_list)
        else:
            metadatas = list(metadatas)

        if len(texts_list) != len(metadatas):
            raise ValueError("texts 与 metadatas 长度不一致")

        outputs: List[Optional[List[float]]] = [None] * len(texts_list)

        text_items: List[str] = []
        text_indices: List[int] = []

        mm_messages: List[List[Dict[str, Any]]] = []
        mm_indices: List[int] = []

        for i, (txt, md) in enumerate(zip(texts_list, metadatas)):
            md = md or {}
            txt = (txt or "").strip()

            if md.get("type") == "image" and md.get("image_path"):
                data_url = _image_file_to_data_url(str(md["image_path"]))

                content: List[Dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]

                # 将 OCR / 页上下文文本一起带上，做图文联合 embedding
                if txt:
                    content.append({"type": "text", "text": txt})
                else:
                    content.append({"type": "text", "text": "Represent the user's input."})

                mm_messages.append(
                    [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                )
                mm_indices.append(i)
            else:
                text_items.append(txt)
                text_indices.append(i)

        # 文本批量请求
        if text_items:
            bsz = max(1, int(batch_size))
            text_vecs: List[List[float]] = []
            for start in range(0, len(text_items), bsz):
                end = min(start + bsz, len(text_items))
                text_vecs.extend(self._embed_batch(text_items[start:end]))
            text_vecs = _normalize_l2(text_vecs)

            for idx, vec in zip(text_indices, text_vecs):
                outputs[idx] = vec

        # 图片逐条多模态请求
        if mm_messages:
            mm_vecs = self._embed_multimodal_messages(mm_messages)
            mm_vecs = _normalize_l2(mm_vecs)

            for idx, vec in zip(mm_indices, mm_vecs):
                outputs[idx] = vec

        if any(v is None for v in outputs):
            missing = [i for i, v in enumerate(outputs) if v is None]
            raise RuntimeError(f"部分 embedding 结果缺失，索引：{missing}")

        return [v for v in outputs if v is not None]

    def embed_query(self, query: str) -> List[float]:
        vectors = _normalize_l2(self._embed_batch([query]))
        if not vectors:
            raise RuntimeError("vLLM embeddings 返回为空")
        return vectors[0]


def get_global_embedder(
    model_path: str,
    query_prompt_name: Optional[str] = "query",
) -> "SentenceTransformerEmbedder | VLLMEmbedder":
    """
    获取一个进程内复用的全局 Qwen3-Embedding-VL embedder。
    第一次调用时构建，之后所有调用都会复用同一个模型实例（不会重复加载权重）。
    """
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        from .config import RagConfig

        rag_cfg = RagConfig()
        if rag_cfg.embed_backend == "vllm":
            _GLOBAL_EMBEDDER = VLLMEmbedder(
                embed_url=rag_cfg.vllm_embed_url,
                model=rag_cfg.vllm_embed_model,
                api_key=rag_cfg.vllm_api_key,
                timeout_s=rag_cfg.vllm_timeout_s,
            )
        else:
            cfg = EmbeddingConfig(model_path=model_path, query_prompt_name=query_prompt_name)
            _GLOBAL_EMBEDDER = SentenceTransformerEmbedder(cfg)
    return _GLOBAL_EMBEDDER


