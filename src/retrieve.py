"""
retrieve.py — the retrieval half of the pipeline:

    question
       |
       +--> BM25 (keyword)        --> top-N ranked list
       +--> dense (semantic)      --> top-N ranked list
       |
       v
    Reciprocal Rank Fusion (combine by RANK, not raw score — BM25 scores
    and cosine similarities aren't on the same scale, so averaging them
    directly would be misleading)
       |
       v
    Cross-encoder reranker (optional — rescores the fused top-N by
    actually reading question+passage together; slower, so only run it
    on a shortlist)
       |
       v
    final top-k sections
"""

import json
import pickle
from functools import lru_cache

import numpy as np

import config
from build_index import section_text, tokenize


def load_sections():
    with open(config.ALL_SECTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_bm25():
    with open(config.BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_dense_embeddings():
    return np.load(config.DENSE_EMBEDDINGS_PATH)


@lru_cache(maxsize=1)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL_NAME)


def bm25_search(query: str, sections: list[dict], top_n: int) -> list[int]:
    """Return the top_n section INDICES ranked by BM25 score, best first."""
    bm25 = _load_bm25()
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_n]
    return list(ranked)


def dense_search(query: str, sections: list[dict], top_n: int) -> list[int]:
    """Return the top_n section INDICES ranked by cosine similarity."""
    model = _load_embedding_model()
    embeddings = _load_dense_embeddings()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # cosine sim, since both are normalized
    ranked = np.argsort(scores)[::-1][:top_n]
    return list(ranked)


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Combine several ranked lists of section indices into one ranking.

    Standard RRF: score(doc) = sum over lists of 1 / (k + rank_in_list).
    `k` is a smoothing constant (60 is the usual default in the IR
    literature) — it just softens the difference between rank 1 and
    rank 2 so one list can't totally dominate.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def rerank(query: str, candidate_indices: list[int], sections: list[dict],
           top_k: int) -> list[int]:
    """Rescore candidates with a cross-encoder, return the best top_k."""
    reranker = _load_reranker()
    pairs = [(query, section_text(sections[i])) for i in candidate_indices]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1][:top_k]
    return [candidate_indices[i] for i in order]


def hybrid_search(
    query: str,
    top_k: int = config.FINAL_TOP_K,
    use_dense: bool = True,
    use_rerank: bool = True,
) -> list[dict]:
    """The main entry point: question -> list of section dicts, best first.

    Each returned dict is a section from all_sections.json, unmodified
    (act / part / division / section / heading / text).
    """
    sections = load_sections()

    ranked_lists = [bm25_search(query, sections, config.BM25_TOP_N)]
    if use_dense and config.DENSE_EMBEDDINGS_PATH.exists():
        ranked_lists.append(dense_search(query, sections, config.DENSE_TOP_N))

    fused = reciprocal_rank_fusion(ranked_lists)

    final_indices = None
    if use_rerank:
        try:
            shortlist = fused[:max(config.RERANK_TOP_N, top_k)]
            final_indices = rerank(query, shortlist, sections, top_k)
        except ImportError:
            print("  ! sentence-transformers not installed — skipping rerank. "
                  "Install with: pip install sentence-transformers")
    if final_indices is None:
        final_indices = fused[:top_k]

    return [sections[i] for i in final_indices]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the penalty for dangerous driving causing death?"
    results = hybrid_search(q)
    for r in results:
        print(f"[{r['act']} s.{r['section']}] {r['heading']}")
