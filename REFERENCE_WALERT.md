# How this project maps onto Walert

Walert (Pathiyan Cherumanal et al., CHIIR '24 — see `reference/walert/`
for the original poster, README and citation) compared an Intent-Based
chatbot against RAG (BM25+Falcon and DPR+Falcon separately) for
answering FAQs about RMIT programs of study. This project borrows its
test-design and evaluation methodology directly, adapted for legal
text:

| Walert                                            | This project |
|----------------------------------------------------|--------------|
| FAQ passages, RMIT School of Computing            | Section-level chunks of 7 Victorian Acts |
| `topics.csv` / `Collection.csv` / `groundtruth.csv` / `qrels.txt` / `gold_summaries.csv` | `data/testset/topics.csv` / `data/processed/all_sections.json` / `data/testset/groundtruth.csv` / `data/testset/gold_summaries.csv` (simplified — no separate qrels.txt, groundtruth.csv covers that role) |
| Known / Inferred / Out-of-KB question categories  | Same three categories, same intent |
| BM25 **or** DPR, compared separately              | BM25 **and** dense, fused (RRF) + reranked — a step beyond Walert's own comparison |
| Falcon-7B-instruct                                | Ollama (llama3.1:8b by default) — same role, lighter to run locally |
| NDCG (known/inferred), % unanswered (out-of-KB), BERTScore, ROUGE-1 | NDCG, % unanswered, ROUGE-1 (BERTScore optional, see requirements.txt) |
| Pyserini (Lucene BM25 + Faiss dense)               | `rank_bm25` + `sentence-transformers` — much lighter dependency footprint, no JDK required |

The one genuine architectural improvement over Walert's own setup is
**fusing** BM25 and dense retrieval (via Reciprocal Rank Fusion) plus a
**cross-encoder reranking** stage, rather than treating them as two
separate systems to compare. Walert's own results (see
`reference/walert/Poster.pdf`) show neither retriever wins outright —
DPR tends to edge out BM25 on known-answer questions, while both trade
places on inferred questions — which is exactly the failure mode
hybrid retrieval is meant to cover.
