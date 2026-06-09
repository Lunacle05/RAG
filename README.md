# RAG 多模态检索服务

基于 **Qwen3-VL-Embedding** + **Qwen3-Reranker** + **ChromaDB** 的文档检索系统，支持 PDF / Word / Excel 等多格式入库，并提供向量库管理 Web UI。

## 项目结构

```
RAG/
├── rag_service/          # 核心服务：入库、检索、Rerank、FastAPI Gateway
├── scripts/              # 启动脚本与工具
├── vector_admin_ui/      # React 向量库管理前端
├── test/                 # 调试与诊断脚本
├── uploaded_docs/        # 待入库文档目录
├── download_embedding.py # 下载 Embedding 模型
├── download_reranker.py  # 下载 Reranker 模型
└── environment_*.yml     # Conda 环境配置
```

## 环境准备

### 1. 创建 Conda 环境

```bash
# 后端（Python + vLLM + Chroma）
conda env create -f environment_env.yml
conda activate env

# 前端（Node.js）
conda env create -f environment_rag_ui.yml
conda activate rag-ui
```

### 2. 下载模型

```bash
conda activate env
cd RAG
python download_embedding.py
python download_reranker.py
python scripts/run_rag.py convert-rerank   # 仅需执行一次
```

### 3. 安装前端依赖

```bash
conda activate rag-ui
cd vector_admin_ui
npm install
```

## 启动服务

需要 **3 个终端** 分别启动 Embedding、Rerank 和 Gateway：

```bash
conda activate env
cd RAG

# 终端 1：Embedding vLLM（端口 6010）
python scripts/run_rag.py vllm-embed

# 终端 2：Rerank vLLM（端口 6009）
python scripts/run_rag.py vllm-rerank

# 终端 3：Gateway + 入库
python scripts/run_rag.py ingest --reset   # 首次建库
# python scripts/run_rag.py ingest         # 之后增量入库
python scripts/run_rag.py gateway          # API 网关，默认 6006
```

### 启动管理 UI

```bash
conda activate rag-ui
cd vector_admin_ui
npm run dev    # 默认 http://127.0.0.1:6008
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/retrieve` | 检索相关文档片段 |
| GET  | `/admin/documents` | 列出已入库文档 |
| POST | `/admin/upload` | 上传并入库文档 |
| POST | `/admin/delete` | 删除文档向量 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_API_PORT` | `6006` | Gateway 端口 |
| `RAG_CHROMA_DIR` | `./chroma_db` | 向量库目录 |
| `RAG_VLLM_EMBED_URL` | `http://127.0.0.1:6010/v1/embeddings` | Embedding 服务 |
| `RAG_VLLM_RERANK_URL` | `http://127.0.0.1:6009/v1/rerank` | Rerank 服务 |
| `RAG_EMBED_MODEL` | `./Qwen3-VL-Embedding-8B` | Embedding 模型路径 |

完整配置见 `rag_service/config.py`。

## 硬件要求

- **GPU**：建议 24GB+ 显存（Embedding 8B + Reranker 0.6B 同时运行）
- **系统**：Linux（已在 AutoDL 环境验证）
