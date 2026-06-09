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

if not res["ids"]:
    raise RuntimeError("数据库里没有 data.pdf 的 image 记录")

# 先拿 page 23 image 1 这条来测
target_idx = None
for i, md in enumerate(res["metadatas"]):
    if md.get("page") == 23 and md.get("image_index") == 0:
        target_idx = i
        break

if target_idx is None:
    target_idx = 0

anchor_id = res["ids"][target_idx]
anchor_doc = res["documents"][target_idx]
anchor_meta = res["metadatas"][target_idx]
anchor_emb = res["embeddings"][target_idx]

print("=== anchor ===")
print("id:", anchor_id)
print("meta:", anchor_meta)
print("doc:", anchor_doc[:500])

q = col.query(
    query_embeddings=[anchor_emb],
    where={"$and": [{"type": "image"}, {"source": "data.pdf"}]},
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

print("\n=== self retrieval ===")
for i, (doc, md, dist) in enumerate(zip(q["documents"][0], q["metadatas"][0], q["distances"][0])):
    print(f"\nrank={i+1}, distance={dist}")
    print("meta:", md)
    print("doc:", doc[:500] if isinstance(doc, str) else doc)
