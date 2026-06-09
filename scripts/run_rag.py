# conda init bash
# source ~/.bashrc
# conda activate env
#第一次--------------------------------------------------
# 第一次需要先转换一次 reranker（只做一次，后续不用重复做）
# cd /root/autodl-tmp/RAG
# bash scripts/run_rag.sh convert-rerank

#-----------------------------------------------------------
# # 第一个终端：启动 embedding vLLM
# cd /root/autodl-tmp/RAG
# python scripts/run_rag.py vllm-embed

# #第二个终端：启动 rerank vLLM
# cd /root/autodl-tmp/RAG
# python scripts/run_rag.py vllm-rerank

# 第三个终端：
# 第一次建库：
# cd /root/autodl-tmp/RAG
# python scripts/run_rag.py ingest --reset
#
# 以后增量入库：
# cd /root/autodl-tmp/RAG
# python scripts/run_rag.py ingest

# 启动 gateway：
# cd /root/autodl-tmp/RAG
# python scripts/run_rag.py gateway
#-------------------------------------
#ui存入文档
# source ~/.bashrc
# conda activate rag-ui
# cd /root/autodl-tmp/vector_admin_ui
# npm run dev


from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"

# 过滤 终端的警告
_TELEMETRY_NOISE_PATTERNS = (
    "Failed to send telemetry event",
)
def _is_noise_line(line: str) -> bool:
    return any(p in line for p in _TELEMETRY_NOISE_PATTERNS)


def _set_default_env() -> None:
    os.environ.setdefault("RAG_API_HOST", "0.0.0.0")
    os.environ.setdefault("RAG_API_PORT", "6006")
    os.environ.setdefault("RAG_EMBED_BACKEND", "vllm")

    embed_port = os.getenv("EMBED_PORT", "6010")
    rerank_port = os.getenv("RERANK_PORT", "6009")

    os.environ.setdefault("RAG_VLLM_EMBED_URL", f"http://127.0.0.1:{embed_port}/v1/embeddings")
    os.environ.setdefault("RAG_VLLM_EMBED_MODEL", str(ROOT_DIR / "Qwen3-VL-Embedding-8B"))
    os.environ.setdefault("RAG_VLLM_TIMEOUT_S", "600")
    os.environ.setdefault("RAG_VLLM_EMBED_MAX_MODEL_LEN", "8192")
    os.environ.setdefault("RAG_VLLM_EMBED_GPU_MEMORY_UTILIZATION", "0.60")

    os.environ.setdefault("RAG_RERANK_ENABLED", "true")
    os.environ.setdefault(
        "RAG_RERANK_INSTRUCTION",
        "Given a web search query, retrieve relevant passages that answer the query",
    )
    os.environ.setdefault("RAG_VLLM_RERANK_URL", f"http://127.0.0.1:{rerank_port}/v1/rerank")
    os.environ.setdefault("RAG_VLLM_RERANK_MODEL", str(ROOT_DIR / "Qwen3-Reranker-0.6B-seq-cls"))
    os.environ.setdefault(
        "RAG_RERANK_TEMPLATE",
        str(ROOT_DIR / "rag_service" / "templates" / "qwen3_reranker_score.jinja"),
    )
    os.environ.setdefault("RAG_RERANK_RECALL_K", "50")
    os.environ.setdefault("RAG_RERANK_FUSION_WEIGHT", "0.7")
    os.environ.setdefault("RAG_VECTOR_FUSION_WEIGHT", "0.3")
    os.environ.setdefault("RAG_RERANK_SOURCE_MODEL", str(ROOT_DIR / "Qwen3-Reranker-0.6B"))
    os.environ.setdefault("RAG_VLLM_RERANK_MAX_MODEL_LEN", "4096")
    os.environ.setdefault("RAG_VLLM_RERANK_GPU_MEMORY_UTILIZATION", "0.35")

    # Chroma 使用本地持久化目录时，单 worker 更稳，避免多进程状态抖动。
    os.environ.setdefault("WORKERS", "4")
    os.environ.setdefault("INPUT_DIR", "uploaded_docs")


def _print_env() -> None:
    keys = [
        "RAG_API_HOST",
        "RAG_API_PORT",
        "RAG_EMBED_BACKEND",
        "RAG_VLLM_EMBED_URL",
        "RAG_VLLM_EMBED_MODEL",
        "RAG_VLLM_EMBED_MAX_MODEL_LEN",
        "RAG_VLLM_EMBED_GPU_MEMORY_UTILIZATION",
        "RAG_RERANK_ENABLED",
        "RAG_RERANK_INSTRUCTION",
        "RAG_VLLM_RERANK_URL",
        "RAG_VLLM_RERANK_MODEL",
        "RAG_RERANK_TEMPLATE",
        "RAG_RERANK_RECALL_K",
        "RAG_RERANK_FUSION_WEIGHT",
        "RAG_VECTOR_FUSION_WEIGHT",
        "RAG_RERANK_SOURCE_MODEL",
        "RAG_VLLM_RERANK_MAX_MODEL_LEN",
        "RAG_VLLM_RERANK_GPU_MEMORY_UTILIZATION",
        "WORKERS",
        "INPUT_DIR",
    ]
    print("== Runtime env ==")
    for k in keys:
        print(f"{k}={os.getenv(k, '')}")


def _run(cmd: list[str], check: bool = True) -> int:
    print("[run_rag.py] Exec:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if _is_noise_line(line):
            continue
        print(line, end="")
    proc.wait()
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode or 0)


def _spawn_bg(cmd: list[str], log_path: Path) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab")
    print(f"[run_rag.py] Start background -> {log_path}")
    return subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=os.environ.copy(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def _vllm_entry_cmd() -> str | None:
    return shutil.which("vllm")


def _build_embed_cmd() -> list[str]:
    embed_port = os.getenv("EMBED_PORT", "6010")
    model = os.environ["RAG_VLLM_EMBED_MODEL"]
    max_len = os.environ["RAG_VLLM_EMBED_MAX_MODEL_LEN"]
    gpu_util = os.environ.get("RAG_VLLM_EMBED_GPU_MEMORY_UTILIZATION", "")

    vllm_bin = _vllm_entry_cmd()
    if vllm_bin:
        cmd = [
            vllm_bin,
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            embed_port,
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--trust-remote-code",
            "--max-model-len",
            max_len,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            embed_port,
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--trust-remote-code",
            "--max-model-len",
            max_len,
        ]

    if gpu_util:
        cmd.extend(["--gpu-memory-utilization", gpu_util])
    return cmd


def _build_rerank_cmd() -> list[str]:
    rerank_port = os.getenv("RERANK_PORT", "6009")
    model = os.environ["RAG_VLLM_RERANK_MODEL"]
    max_len = os.environ["RAG_VLLM_RERANK_MAX_MODEL_LEN"]
    gpu_util = os.environ.get("RAG_VLLM_RERANK_GPU_MEMORY_UTILIZATION", "")

    vllm_bin = _vllm_entry_cmd()
    if vllm_bin:
        cmd = [
            vllm_bin,
            "serve",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            rerank_port,
            "--runner",
            "pooling",
            "--max-model-len",
            max_len,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            rerank_port,
            "--runner",
            "pooling",
            "--max-model-len",
            max_len,
        ]

    if gpu_util:
        cmd.extend(["--gpu-memory-utilization", gpu_util])
    return cmd


def _wait_http_ready(url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2):
                print(f"[run_rag.py] Ready: {url}")
                return
        except (URLError, TimeoutError, OSError):
            time.sleep(1)
    raise TimeoutError(f"Timeout waiting for {url}")


def cmd_convert_rerank() -> None:
    out_dir = Path(os.environ["RAG_VLLM_RERANK_MODEL"])
    if out_dir.exists() and (out_dir / "config.json").exists():
        print(f"[run_rag.py] Rerank seq-cls model already exists: {out_dir}")
        return
    _run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "convert_qwen3_reranker_to_seq_cls.py"),
            "--model_name",
            os.environ["RAG_RERANK_SOURCE_MODEL"],
            "--path",
            str(out_dir),
        ]
    )


def cmd_ingest(reset: bool) -> None:
    cmd = [sys.executable, "-m", "rag_service.ingest", "--input", os.environ["INPUT_DIR"]]
    if reset:
        cmd.append("--reset")
    _run(cmd)


def cmd_gateway() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rag_service.server:app",
            "--host",
            os.environ["RAG_API_HOST"],
            "--port",
            os.environ["RAG_API_PORT"],
            "--workers",
            os.environ["WORKERS"],
        ]
    )


def cmd_vllm_embed() -> None:
    _run(_build_embed_cmd())


def cmd_vllm_rerank() -> None:
    _run(_build_rerank_cmd())


def cmd_all() -> None:
    cmd_convert_rerank()
    _spawn_bg(_build_embed_cmd(), LOG_DIR / "vllm_embed.log")
    _spawn_bg(_build_rerank_cmd(), LOG_DIR / "vllm_rerank.log")

    embed_port = os.getenv("EMBED_PORT", "6010")
    rerank_port = os.getenv("RERANK_PORT", "6009")
    _wait_http_ready(f"http://127.0.0.1:{embed_port}/v1/models")
    _wait_http_ready(f"http://127.0.0.1:{rerank_port}/v1/models")

    cmd_gateway()


def main() -> None:
    _set_default_env()

    parser = argparse.ArgumentParser(description="Python launcher for RAG services.")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("env")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--reset", action="store_true")

    sub.add_parser("convert-rerank")
    sub.add_parser("vllm-embed")
    sub.add_parser("vllm-rerank")
    sub.add_parser("gateway")
    sub.add_parser("all")
    args = parser.parse_args()

    _print_env()

    if args.mode == "env":
        return
    if args.mode == "ingest":
        cmd_ingest(reset=args.reset)
        return
    if args.mode == "convert-rerank":
        cmd_convert_rerank()
        return
    if args.mode == "vllm-embed":
        cmd_vllm_embed()
        return
    if args.mode == "vllm-rerank":
        cmd_vllm_rerank()
        return
    if args.mode == "gateway":
        cmd_gateway()
        return
    if args.mode == "all":
        cmd_all()
        return

    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
