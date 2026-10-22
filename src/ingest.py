"""Chunk docs in data/raw/*.txt and build BM25 + vector indexes."""
import json, pickle, re
from pathlib import Path
import numpy as np, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL, CHUNK_TOKENS, CHUNK_OVERLAP

RAW, OUT = Path("data/raw"), Path("data/index")

def tokenize(t): return re.findall(r"\w+", t.lower())

def chunk_doc(doc_id: str, text: str):
    words = text.split()
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        piece = " ".join(words[start:start + CHUNK_TOKENS])
        if piece.strip():
            yield {"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "text": piece}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = []
    for p in sorted(RAW.glob("*.txt")):
        text = p.read_text(encoding="utf-8-sig")


        chunks += list(chunk_doc(p.stem, text))
    print(f"{len(chunks)} chunks")

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

    (OUT / "chunks.json").write_text(json.dumps(chunks))
    pickle.dump(bm25, open(OUT / "bm25.pkl", "wb"))
    faiss.write_index(index, str(OUT / "faiss.index"))

if __name__ == "__main__":
    main()

