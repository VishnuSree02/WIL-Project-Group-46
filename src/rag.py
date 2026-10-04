"""
rag.py — the full pipeline: question -> retrieved sections -> answer.
"""

import config
from retrieve import hybrid_search
from generate import generate_answer


def ask(question: str, top_k: int = config.FINAL_TOP_K,
        use_dense: bool = True, use_rerank: bool = True,
        llm: str = config.OLLAMA_MODEL, extractive: bool = False) -> dict:
    """Run the full pipeline for one question.

    Returns {"question", "answer", "sources": [section dicts used]}.
    Set extractive=True to skip the LLM entirely and just return the
    top matching section (useful if Ollama isn't installed yet, or for
    a fast sanity check that retrieval is working).
    """
    sections = hybrid_search(question, top_k=top_k, use_dense=use_dense,
                              use_rerank=use_rerank)

    if extractive:
        if not sections:
            answer = f"This is {config.REFUSAL_PHRASE}."
        else:
            top = sections[0]
            answer = f"[{top['act']}, s. {top['section']} — {top['heading']}]\n{top['text']}"
    else:
        answer = generate_answer(question, sections, model=llm)

    return {"question": question, "answer": answer, "sources": sections}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the penalty for dangerous driving causing death?"
    result = ask(q)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['act']}, s. {s['section']} ({s['heading']})")
