import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_service.config import RagConfig
from rag_service.chroma_store import ChromaConfig, ChromaStore

cfg = RagConfig()
store = ChromaStore(
    ChromaConfig(
        persist_directory=cfg.persist_directory,
        collection_name=cfg.collection_name,
    )
)
col = store._collection

res = col.get(
    where={"$and": [{"type": "image"}, {"source": "data.pdf"}]},
    include=["documents", "metadatas", "embeddings"],
)

print("image count:", len(res["ids"]))

for i, _id in enumerate(res["ids"]):
    doc = res["documents"][i]
    md = res["metadatas"][i]
    emb = res["embeddings"][i]

    print("\n" + "=" * 80)
    print(f"[{i}] id={_id}")
    print("meta:", md)
    print("embedding_dim:", len(emb) if emb else None)
    print("-" * 80)
    print(doc[:3000] if isinstance(doc, str) else doc)

