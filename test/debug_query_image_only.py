import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_service.config import RagConfig
from rag_service.chroma_store import ChromaConfig, ChromaStore
from rag_service.embeddings import get_global_embedder

cfg = RagConfig()

embedder = get_global_embedder(
    model_path=cfg.model_path,
    query_prompt_name=cfg.query_prompt_name,
)

store = ChromaStore(
    ChromaConfig(
        persist_directory=cfg.persist_directory,
        collection_name=cfg.collection_name,
    )
)
col = store._collection

queries = [
    "GPA评分体系",
    "字母评分等级",
    "学术表现 对照表",
    "GPA评分体系的图表",
]

for query in queries:
    print("\n" + "=" * 100)
    print("query:", query)

    qvec = embedder.embed_query(query)

    res = col.query(
        query_embeddings=[qvec],
        where={"$and": [{"type": "image"}, {"source": "data.pdf"}]},
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    for i, (doc, md, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
        print(f"\nrank={i+1}, distance={dist}")
        print("meta:", md)
        print("doc:", doc[:500] if isinstance(doc, str) else doc)
