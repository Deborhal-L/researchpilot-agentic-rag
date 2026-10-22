import os
from dotenv import load_dotenv

load_dotenv(override=True)

# LLM backend: mock (free), ollama (local), or anthropic (API)
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_HOPS = int(os.getenv("MAX_HOPS", 2))

TOP_N = 10
TOP_K = 3
CHUNK_TOKENS = 300
CHUNK_OVERLAP = 50

# USD per 1 million tokens; mock and Ollama cost zero
PRICE_IN = float(os.getenv("PRICE_IN", 0.0))
PRICE_OUT = float(os.getenv("PRICE_OUT", 0.0))
