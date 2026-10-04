import json
import sys
from pathlib import Path


RESULTS_DIR = Path("results")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_summary(run_name):
    path = RESULTS_DIR / run_name / "summary.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    return load_json(path)


def get_questions(run_name):
    path = RESULTS_DIR / run_name / "per_question.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    return load_json(path)


def is_refused(item):
    """
    Determine whether the answer was treated as a refusal.

    Prefer the explicit 'refused' field if available.
    Otherwise check for the project's refusal phrase.
    """
    if "refused" in item:
        return bool(item["refused"])

    answer = str(item.get("answer", "")).lower()

    refusal_phrases = [
        "not covered in the legislation i have access to",
        "not provided in the text",
        "not enough information",
        "i'm not able to find",
        "i don't see any sections",
    ]

    return any(phrase in answer for phrase in refusal_phrases)


def calculate_extra_metrics(questions):
    """
    Calculate the metrics that are not present in summary.json.

    The exact field names can vary between versions of the evaluator,
    so this function checks several likely names.
    """

    # ------------------------------------------------------------
    # Out-of-KB refusal
    # ------------------------------------------------------------

    out_of_kb = [
        q for q in questions
        if q.get("category") == "out_of_kb"
    ]

    refused_out_of_kb = [
        q for q in out_of_kb
        if is_refused(q)
    ]

    pct_refused_out_of_kb = (
        100 * len(refused_out_of_kb) / len(out_of_kb)
        if out_of_kb else 0
    )

    # ------------------------------------------------------------
    # Answerable questions
    # ------------------------------------------------------------

    answerable = [
        q for q in questions
        if q.get("category") in ("known", "inferred")
    ]

    wrongly_refused = [
        q for q in answerable
        if is_refused(q)
    ]

    pct_wrongly_refused = (
        100 * len(wrongly_refused) / len(answerable)
        if answerable else 0
    )

    # ------------------------------------------------------------
    # Citation metrics
    # ------------------------------------------------------------

    citation_match_values = []
    ground_truth_citation_values = []

    for q in questions:

        # Try several possible field names used by evaluators.

        citation_match = None

        for key in [
            "citation_matches_retrieved",
            "citation_match",
            "citation_match_retrieved",
            "citations_match_retrieved",
        ]:
            if key in q:
                citation_match = q[key]
                break

        if citation_match is not None:
            citation_match_values.append(bool(citation_match))

        ground_truth = None

        for key in [
            "citation_matches_ground_truth",
            "ground_truth_citation",
            "cites_ground_truth",
            "citation_ground_truth",
            "citation_matches_gt",
        ]:
            if key in q:
                ground_truth = q[key]
                break

        if ground_truth is not None:
            ground_truth_citation_values.append(bool(ground_truth))

    pct_citations_match_retrieved = (
        100 * sum(citation_match_values) /
        len(citation_match_values)
        if citation_match_values else None
    )

    pct_answers_citing_ground_truth = (
        100 * sum(ground_truth_citation_values) /
        len(ground_truth_citation_values)
        if ground_truth_citation_values else None
    )

    return {
        "pct_refused_out_of_kb": pct_refused_out_of_kb,
        "pct_wrongly_refused": pct_wrongly_refused,
        "pct_citations_match_retrieved": pct_citations_match_retrieved,
        "pct_answers_citing_ground_truth":
            pct_answers_citing_ground_truth,
    }


def print_table(runs):
    rows = []

    for run in runs:
        summary = get_summary(run)
        questions = get_questions(run)

        extra = calculate_extra_metrics(questions)

        rows.append({
            "run": run,
            "ndcg_known": summary.get("ndcg_known"),
            "ndcg_inferred": summary.get("ndcg_inferred"),

            # Prefer summary.json because this is the official
            # evaluator value.
            "refused_out_of_kb":
                summary.get(
                    "pct_unanswered_out_of_kb",
                    extra["pct_refused_out_of_kb"]
                ),

            "wrongly_refused":
                extra["pct_wrongly_refused"],

            "rouge_1":
                summary.get("rouge_1_avg"),

            "citation_retrieved":
                extra["pct_citations_match_retrieved"],

            "citation_ground_truth":
                extra["pct_answers_citing_ground_truth"],
        })

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print()
    print("POLICE CODE RAG EVALUATION COMPARISON")
    print("=" * 100)

    header = (
        f"{'Metric':45}"
        f"{runs[0]:>16}"
        f"{runs[1]:>16}"
        f"{runs[2]:>20}"
    )

    print(header)
    print("-" * 100)

    # ------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------

    def value(row, key, percent=False):
        v = row[key]

        if v is None:
            return "N/A"

        if percent:
            return f"{v:.2f}%"

        return f"{v:.4f}"

    metric_rows = [
        (
            "NDCG@3 for known questions",
            "ndcg_known",
            False,
        ),
        (
            "NDCG@3 for inferred questions",
            "ndcg_inferred",
            False,
        ),
        (
            "% refused on out-of-scope questions",
            "refused_out_of_kb",
            True,
        ),
        (
            "% wrongly refused on answerable questions",
            "wrongly_refused",
            True,
        ),
        (
            "ROUGE-1",
            "rouge_1",
            False,
        ),
        (
            "% citations matching retrieved section",
            "citation_retrieved",
            True,
        ),
        (
            "% answers citing ground-truth section",
            "citation_ground_truth",
            True,
        ),
    ]

    for label, key, percent in metric_rows:

        print(
            f"{label:45}"
            f"{value(rows[0], key, percent):>16}"
            f"{value(rows[1], key, percent):>16}"
            f"{value(rows[2], key, percent):>20}"
        )

    print("=" * 100)
    print()


def main():

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python compare_results.py "
            "bm25_only hybrid hybrid_rerank"
        )
        sys.exit(1)

    runs = sys.argv[1:4]

    try:
        print_table(runs)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()