from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "index"
TESTSET_DIR = DATA_DIR / "testset"
RESULTS_DIR = ROOT_DIR / "results"

ALL_SECTIONS_PATH = PROCESSED_DIR / "all_sections.json"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
DENSE_EMBEDDINGS_PATH = INDEX_DIR / "dense_embeddings.npy"
INDEX_META_PATH = INDEX_DIR / "meta.json"

# Dense embedding model — small and fast, good enough for a corpus of a
# few thousand sections. Swap for "BAAI/bge-base-en-v1.5" for a quality
# bump at the cost of speed, if your machine can handle it.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder reranker — reads (question, passage) pairs together and
# rescores the fused candidate list. Much slower than the retrievers
# above, so only run it over the top N fused candidates, not the whole
# corpus.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates each retriever contributes before fusion, and how
# many survive fusion to be handed to the reranker / LLM.
BM25_TOP_N = 20
DENSE_TOP_N = 20
RERANK_TOP_N = 5      # how many fused candidates the reranker rescores
FINAL_TOP_K = 3        # how many sections actually go to the LLM

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

REFUSAL_PHRASE = "Not covered in the legislation I have access to"
