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
    include=["documents", "metadatas"],
)

print("image count:", len(res["ids"]))

for i, _id in enumerate(res["ids"]):
    doc = res["documents"][i] or ""
    md = res["metadatas"][i]

    print("\n" + "=" * 100)
    print(f"[{i}] id={_id}")
    print("meta:", md)

    has_ocr = "Image OCR:\n" in doc
    print("has_ocr:", has_ocr)

    if "Page context:\n" in doc:
        page_context = doc.split("Page context:\n", 1)[1]
        if "\n\nImage OCR:\n" in page_context:
            page_context = page_context.split("\n\nImage OCR:\n", 1)[0]
        print("\n[PAGE CONTEXT]")
        print(page_context[:1000])
    else:
        print("\n[PAGE CONTEXT]")
        print("[NONE]")

    if "Image OCR:\n" in doc:
        ocr_part = doc.split("Image OCR:\n", 1)[1]
        print("\n[IMAGE OCR]")
        print(ocr_part[:2000])
    else:
        print("\n[IMAGE OCR]")
        print("[EMPTY]")
