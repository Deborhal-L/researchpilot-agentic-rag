import json, pickle
import numpy as np, faiss
from pathlib import Path
from .config import EMBED_MODEL, RERANK_MODEL, TOP_N, TOP_K
from .ingest import tokenize
from rank_bm25 import BM25Okapi
IDX = Path("data/index")

class Retriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self.chunks = json.loads((IDX / "chunks.json").read_text())
        self.bm25 = pickle.load(open(IDX / "bm25.pkl", "rb"))
        self.index = faiss.read_index(str(IDX / "faiss.index"))
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)

    def bm25_search(self, q, n=TOP_N):
        scores = self.bm25.get_scores(tokenize(q))
        return list(np.argsort(scores)[::-1][:n])

    def vector_search(self, q, n=TOP_N):
     v = self.embedder.encode([q], normalize_embeddings=True).astype("float32")
     _, ids = self.index.search(v, min(n, self.index.ntotal))
     return [int(i) for i in ids[0] if i >= 0]
 
    @staticmethod
    def rrf(rank_lists, k=60):
        scores = {}
        for rl in rank_lists:
            for rank, idx in enumerate(rl):
                scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def search(self, q, top_k=TOP_K, mode="hybrid_rerank"):
        if mode == "vector":
            ids = self.vector_search(q)[:top_k]
            return [self.chunks[i] for i in ids]
        ids = self.rrf([self.bm25_search(q), self.vector_search(q)])[:TOP_N]
        cands = [self.chunks[i] for i in ids]
        if mode == "hybrid":
            return cands[:top_k]
        s = self.reranker.predict([(q, c["text"]) for c in cands])
        order = np.argsort(s)[::-1][:top_k]
        return [{**cands[i], "rerank_score": float(s[i])} for i in order]
    def search_in_document(
        self,
        q,
        doc_id,
        top_k=TOP_K,
        mode="hybrid_rerank",
    ):
        """Retrieve and rank chunks from one document only."""

        global_indices = [
            i
            for i, chunk in enumerate(self.chunks)
            if chunk.get("doc_id") == doc_id
        ]

        if not global_indices:
            return []

        chunks = [self.chunks[i] for i in global_indices]
        limit = min(TOP_N, len(chunks))

        # BM25 keyword retrieval within this document.
        local_bm25 = BM25Okapi([
            tokenize(chunk["text"]) for chunk in chunks
        ])
        keyword_scores = local_bm25.get_scores(tokenize(q))
        keyword_order = list(
            np.argsort(keyword_scores)[::-1][:limit]
        )

        # Vector retrieval within this document.
        query_vector = self.embedder.encode(
            [q],
            normalize_embeddings=True,
        ).astype("float32")[0]

        document_vectors = np.vstack([
            self.index.reconstruct(i)
            for i in global_indices
        ]).astype("float32")

        vector_scores = document_vectors @ query_vector
        vector_order = list(
            np.argsort(vector_scores)[::-1][:limit]
        )

        if mode == "vector":
            return [chunks[i] for i in vector_order[:top_k]]

        # Combine keyword and vector rankings.
        ranked_ids = self.rrf([
            keyword_order,
            vector_order,
        ])[:limit]

        candidates = [chunks[i] for i in ranked_ids]

        if mode == "hybrid":
            return candidates[:top_k]

        # Rerank only the selected document's passages.
        rerank_scores = self.reranker.predict([
            (q, chunk["text"]) for chunk in candidates
        ])

        order = np.argsort(rerank_scores)[::-1][:top_k]

        return [
            {
                **candidates[i],
                "rerank_score": float(rerank_scores[i]),
            }
            for i in order
        ]