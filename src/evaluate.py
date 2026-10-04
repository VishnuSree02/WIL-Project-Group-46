"""
evaluate.py — evaluate the pipeline the same way the Walert paper did:

  Known / Inferred questions  -> retrieval quality via NDCG@k
  Out-of-KB questions         -> % unanswered (higher is better here —
                                  see the note below, this is the one
                                  metric where "unanswered" is success)
  All answered questions      -> answer quality via ROUGE-1 (and
                                  BERTScore if installed)

IMPORTANT about "% unanswered": for out-of-KB questions there is, by
definition, no correct answer available in the knowledge base. So a
well-calibrated system should refuse ("not covered...") on close to
100% of them. A LOW % unanswered on out-of-KB questions is a BAD sign
— it means the system is attempting (and likely hallucinating) an
answer instead of admitting it doesn't know. This is the opposite of
known/inferred, where you obviously want a high answer rate.
"""

import argparse
import csv
import json
import math
from collections import defaultdict

import config
from rag import ask

try:
    from rouge import Rouge
    _rouge = Rouge()
except ImportError:
    _rouge = None

try:
    from bert_score import score as bert_score
    _bert_score_available = True
except ImportError:
    _bert_score_available = False


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_testset():
    topics = load_csv(config.TESTSET_DIR / "topics.csv")
    groundtruth_rows = load_csv(config.TESTSET_DIR / "groundtruth.csv")
    gold_rows = load_csv(config.TESTSET_DIR / "gold_summaries.csv")

    groundtruth = defaultdict(dict)  # question_id -> {(act,section): relevance}
    for row in groundtruth_rows:
        key = (row["act"], row["section"])
        groundtruth[row["question_id"]][key] = int(row["relevance_judgment"])

    gold = {row["question_id"]: row["summary"] for row in gold_rows}
    return topics, groundtruth, gold


def ndcg_at_k(retrieved_keys: list[tuple], relevance: dict, k: int) -> float:
    """Standard NDCG@k with graded relevance (0/1/2)."""
    def dcg(keys):
        return sum(
            relevance.get(key, 0) / math.log2(i + 2)
            for i, key in enumerate(keys[:k])
        )
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_order))
    if idcg == 0:
        return 0.0
    return dcg(retrieved_keys) / idcg


def is_refusal(answer: str) -> bool:
    return config.REFUSAL_PHRASE.lower() in answer.lower()


def evaluate(top_k: int = 5, use_dense: bool = True, use_rerank: bool = True,
             llm: str = config.OLLAMA_MODEL, extractive: bool = False,
             run_name: str = "baseline"):
    topics, groundtruth, gold = load_testset()

    ndcg_scores = defaultdict(list)   # category -> [ndcg, ...]
    unanswered_flags = []             # for out_of_kb
    rouge_scores = []
    per_question_results = []

    for row in topics:
        qid, category, question = row["question_id"], row["category"], row["question"]
        result = ask(question, top_k=top_k, use_dense=use_dense,
                      use_rerank=use_rerank, llm=llm, extractive=extractive)
        retrieved_keys = [(s["act"], s["section"]) for s in result["sources"]]

        record = {"question_id": qid, "category": category, "question": question,
                   "answer": result["answer"], "retrieved": retrieved_keys}

        if category in ("known", "inferred"):
            score = ndcg_at_k(retrieved_keys, groundtruth.get(qid, {}), top_k)
            ndcg_scores[category].append(score)
            record["ndcg"] = score

        if category == "out_of_kb":
            refused = is_refusal(result["answer"])
            unanswered_flags.append(refused)
            record["refused"] = refused

        if qid in gold and _rouge is not None:
            try:
                rouge_result = _rouge.get_scores(result["answer"], gold[qid])
                r1 = rouge_result[0]["rouge-1"]["f"]
                rouge_scores.append(r1)
                record["rouge_1"] = r1
            except Exception:
                pass  # rouge chokes on some empty/short strings — skip, don't crash the run

        per_question_results.append(record)
        print(f"[{qid}] {category}: {record.get('ndcg', record.get('refused', ''))}")

    summary = {
        "run_name": run_name,
        "ndcg_known": sum(ndcg_scores["known"]) / len(ndcg_scores["known"]) if ndcg_scores["known"] else None,
        "ndcg_inferred": sum(ndcg_scores["inferred"]) / len(ndcg_scores["inferred"]) if ndcg_scores["inferred"] else None,
        "pct_unanswered_out_of_kb": (sum(unanswered_flags) / len(unanswered_flags) * 100) if unanswered_flags else None,
        "rouge_1_avg": sum(rouge_scores) / len(rouge_scores) if rouge_scores else None,
    }

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out_dir = config.RESULTS_DIR / run_name
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_question.json", "w", encoding="utf-8") as f:
        json.dump(per_question_results, f, indent=2)

    print("\n=== Summary ===")
    print(f"NDCG@{top_k} (known):        {summary['ndcg_known']}")
    print(f"NDCG@{top_k} (inferred):     {summary['ndcg_inferred']}")
    print(f"% unanswered (out-of-KB):   {summary['pct_unanswered_out_of_kb']}  "
          f"(higher = better here, see module docstring)")
    print(f"ROUGE-1 (avg, answered qs): {summary['rouge_1_avg']}")
    print(f"\nWrote results to {out_dir}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--extractive", action="store_true",
                         help="skip the LLM, just return the top retrieved section")
    parser.add_argument("--llm", default=config.OLLAMA_MODEL)
    args = parser.parse_args()

    evaluate(top_k=args.top_k, use_dense=not args.no_dense,
              use_rerank=not args.no_rerank, llm=args.llm,
              extractive=args.extractive, run_name=args.name)
