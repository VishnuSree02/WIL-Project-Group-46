"""
generate.py — turn retrieved sections + a question into a grounded,
cited answer using a local Ollama model.

The prompt is deliberately strict about two things, which matter a lot
for a legal-answers tool:
  1. Always cite the Act + section number the answer is drawn from.
  2. Say so explicitly — and only that — when the retrieved sections
     don't actually answer the question, instead of guessing.
"""

import requests

import config

SYSTEM_PROMPT = f"""You are a legal information assistant answering questions about \
Victorian legislation. You are NOT a lawyer and this is not legal advice.

Answer ONLY using the numbered sections provided below. For every claim, cite the \
Act and section number it comes from, like this: (Crimes Act 1958, s. 319).

If the provided sections do not contain enough information to answer the question, \
you MUST respond with exactly this sentence and nothing else: \
"This is {config.REFUSAL_PHRASE}."

Do not use any outside knowledge of the law. Do not guess."""


def build_prompt(question: str, sections: list[dict]) -> str:
    doc_blocks = []
    for i, s in enumerate(sections, start=1):
        doc_blocks.append(
            f"Section {i} — {s['act']}, s. {s['section']} ({s['heading']}):\n{s['text']}"
        )
    docs_text = "\n\n".join(doc_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"{docs_text}\n\n"
        f"Answer (with citations):"
    )


def call_ollama(prompt: str, model: str = config.OLLAMA_MODEL, timeout: int = 120) -> str:
    try:
        response = requests.post(
            config.OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Couldn't reach Ollama at "
            f"{config.OLLAMA_URL}. Is it running? (open the Ollama app, or "
            "run `ollama serve` in another terminal)"
        )


def generate_answer(question: str, sections: list[dict],
                     model: str = config.OLLAMA_MODEL) -> str:
    if not sections:
        return f"This is {config.REFUSAL_PHRASE}."
    prompt = build_prompt(question, sections)
    return call_ollama(prompt, model=model)
