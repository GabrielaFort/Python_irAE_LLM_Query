# debug_index.py
from index_manager import IndexManager
im = IndexManager()
im.load()
print("Loaded", len(im.metadatas), "chunks.")
results = im.search("first-line therapy for metastatic melanoma", top_k=3)
for r in results:
    print(r['doc_key'], r['section_title'][:80], r['score'])