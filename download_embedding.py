from pathlib import Path

from modelscope import snapshot_download

ROOT = Path(__file__).resolve().parent
model_dir = snapshot_download(
    "Qwen/Qwen3-VL-Embedding-8B",
    local_dir=str(ROOT / "Qwen3-VL-Embedding-8B"),
)

print("Embedding model downloaded to:", model_dir)