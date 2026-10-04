"""
build_index.py — build the BM25 keyword index and dense embedding index
over the section-level chunks produced by extract.py.

Run this after extract.py, and again any time you re-run extract.py
with different Acts.
"""

import json
import pickle
import sys

import numpy as np

import config


def load_sections():
    if not config.ALL_SECTIONS_PATH.exists():
        print(f"{config.ALL_SECTIONS_PATH} not found — run extract.py first.")
        sys.exit(1)
    with open(config.ALL_SECTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def section_text(section: dict) -> str:
    """The text actually indexed for a section: heading + body.

    Including the heading helps both BM25 (headings are often close to
    how people phrase questions, e.g. "dangerous driving") and dense
    retrieval (the heading gives the embedding model a topic anchor).
    """
    return f"{section['heading']}. {section['text']}"


def tokenize(text: str) -> list[str]:
    """Simple whitespace/lowercase tokenizer, good enough for BM25."""
    return text.lower().split()


def build_bm25(sections: list[dict]):
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(section_text(s)) for s in sections]
    bm25 = BM25Okapi(corpus)
    return bm25


def build_dense(sections: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model {config.EMBEDDING_MODEL_NAME} "
          f"(first run downloads it, ~90MB)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    texts = [section_text(s) for s in sections]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity == dot product
    )
    return embeddings


def main():
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    sections = load_sections()
    print(f"Loaded {len(sections)} sections.")

    print("Building BM25 index...")
    bm25 = build_bm25(sections)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  -> wrote {config.BM25_INDEX_PATH}")

    try:
        print("Building dense embedding index...")
        embeddings = build_dense(sections)
        np.save(config.DENSE_EMBEDDINGS_PATH, embeddings)
        print(f"  -> wrote {config.DENSE_EMBEDDINGS_PATH} "
              f"(shape {embeddings.shape})")
        dense_ok = True
    except ImportError:
        print("  ! sentence-transformers not installed — skipping dense "
              "index. BM25-only retrieval will still work. Install with:\n"
              "    pip install sentence-transformers")
        dense_ok = False

    meta = {
        "n_sections": len(sections),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "dense_index_built": dense_ok,
    }
    with open(config.INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  -> wrote {config.INDEX_META_PATH}")
    print("\nDone. Run `python cli.py ask \"your question\"` to try it.")


if __name__ == "__main__":
    main()
