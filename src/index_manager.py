import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class IndexManager:
    def __init__(self, kb_dir="src/knowledge_base", model_name="NeuML/pubmedbert-base-embeddings"):
        self.kb_dir = kb_dir
        self.model_name = model_name
        self.index = None
        self.metadatas = None
        self.model = None
        self.embed_dim = None

    def load(self):
        idx_path = os.path.join(self.kb_dir, "faiss.index")
        meta_path = os.path.join(self.kb_dir, "metadatas.jsonl")
        cfg_path = os.path.join(self.kb_dir, "index_config.json")

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Knowledge base not built. Run build_pdf_kb.py first.")

        self.index = faiss.read_index(idx_path)

        metas = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                metas.append(json.loads(line.strip()))
        self.metadatas = metas

        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
                self.embed_dim = cfg.get("embed_dim", None)

        return self
    
    def init_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            self.embed_dim = self.model.get_sentence_embedding_dimension()
        return self  

    def embed_fn(self, text):
        """
        Input: single string (user query)
        Returns: normalized numpy vector
        """
        self.init_model()
        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb  
    
    def search(self, query, top_k=10):
        """
        Embed query and search FAISS. Returns list of dicts with metadata + score.
        """
        if self.index is None or self.metadatas is None:
            raise RuntimeError("Index not loaded. Call load() first.")
        qv = self.embed_fn(query).reshape(1, -1)
        # Check dtype and shape
        if qv.dtype != 'float32':
            qv = qv.astype('float32')
        if qv.shape[1] != self.index.d:
            raise ValueError(f"Query vector dimension {qv.shape[1]} does not match index dimension {self.index.d}")
        D, I = self.index.search(qv, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.metadatas[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results