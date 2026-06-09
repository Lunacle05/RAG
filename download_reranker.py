from pathlib import Path

from modelscope import snapshot_download

ROOT = Path(__file__).resolve().parent
model_dir = snapshot_download(
    "Qwen/Qwen3-Reranker-0.6B",
    local_dir=str(ROOT / "Qwen3-Reranker-0.6B"),
)

print("Reranker model downloaded to:", model_dir)