# Police Code RAG — Victorian Legislation Assistant

A retrieval-augmented (RAG) question-answering system over seven pieces
of Victorian legislation:

- Crimes Act 1958
- Drugs, Poisons and Controlled Substances Act 1981
- Infringements Act 2006
- Road Safety Act 1986
- Sentencing Act 1991
- Summary Offences Act 1966
- Victoria Police Act 2013

Built following the architecture and evaluation methodology of
**Walert** (Pathiyan Cherumanal et al., CHIIR '24) — see
`REFERENCE_WALERT.md` for how this project maps onto that paper — but
adapted for legal text: hybrid retrieval (keyword + semantic) with
fusion and reranking, rather than a single retriever.

## Architecture

```
                    USER QUESTION
                          |
              +-----------+-----------+
              v                       v
            BM25              Dense Vector Search
       keyword retrieval       semantic retrieval
       (rank_bm25)              (sentence-transformers)
              |                       |
              +-----------+-----------+
                          v
                Reciprocal Rank Fusion
                          |
                          v
              Cross-encoder Reranker
                          |
                          v
                Top relevant sections
                          |
                          v
                    LLM (via Ollama)
                          |
                          v
              Grounded Answer + Citation
```

Each stage is toggle-able (`--no-dense`, `--no-rerank`, `--extractive`)
so you can run ablations — e.g. BM25-only vs hybrid vs hybrid+rerank —
and measure the difference with `eval`, which is exactly the kind of
comparison Walert's own paper makes.

## Project layout

```
police-code-rag/
├── data/
│   ├── raw/            the 7 Act PDFs
│   ├── processed/      extracted section-level JSON (one file per Act,
│   │                   plus all_sections.json combining them)
│   │                   and index/ (built BM25 + embedding indexes)
│   └── testset/        topics.csv, groundtruth.csv, gold_summaries.csv
│                       — a seed test set, see "Extending the test set"
├── src/
│   ├── extract.py      PDF -> section-aware chunks
│   ├── build_index.py  builds the BM25 + dense indexes
│   ├── retrieve.py      hybrid search + RRF fusion + reranking
│   ├── generate.py      Ollama prompt + call
│   ├── rag.py           ties retrieve + generate together
│   └── evaluate.py      Walert-style evaluation (NDCG / % unanswered / ROUGE-1)
├── cli.py               command-line entry point for everything above
├── requirements.txt
└── results/             written by `eval`, one folder per run name
```

The `data/processed/*.json` files already in this zip are the **real,
already-extracted output** — running `extract` again will just
regenerate them from the PDFs, so you can start from `build` /`ask`
straight away if you just want to try it, or re-run `extract` after
editing the parser.

## Setup (Windows + VS Code)

1. **Python 3.10+** from python.org (tick "Add python.exe to PATH" on
   install).
2. **Poppler** (for best-quality PDF extraction — the parser falls
   back to `pypdf` automatically if this isn't installed, but poppler
   gives cleaner results): download a Windows build from
   https://github.com/oschwartz10612/poppler-windows/releases, unzip
   it somewhere, and add its `bin` folder to your PATH. Check it
   worked with `pdftotext -v` in a new terminal.
3. **Ollama** from https://ollama.com, then:
   ```powershell
   ollama pull llama3.1:8b
   ```
   (or `llama3.2:3b` if your machine has 8GB RAM or less — pass
   `--llm llama3.2:3b` to `ask`/`eval` if so).
4. Open this folder in VS Code, open a terminal (Ctrl+`), then:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running it

```powershell
# 1. Extract the PDFs into section-level chunks (already done in this
#    zip, but re-run any time you change data/raw/ or extract.py)
python cli.py extract

# 2. Build the BM25 + dense indexes
python cli.py build

# 3. Ask it something
python cli.py ask "What is the penalty for dangerous driving causing death?"

# 4. Spot-check that a specific section was extracted cleanly
python cli.py inspect "Crimes" 319

# 5. Run the evaluation
python cli.py eval --name hybrid_rerank
python cli.py eval --name bm25_only --no-dense --no-rerank
```

Compare `results/hybrid_rerank/summary.json` against
`results/bm25_only/summary.json` — that comparison IS your ablation
study.

If Ollama isn't installed yet, or you just want to sanity-check that
retrieval alone is working, add `--extractive` to `ask`/`eval` to skip
the LLM and just return the top retrieved section verbatim.

## The test set (`data/testset/`)

Seeded with 18 questions (6 known-answer, 6 inferred-answer, 6
out-of-KB) across all 7 Acts, following Walert's three-category
design — see `topics.csv`, `groundtruth.csv`, `gold_summaries.csv`.
This is a **starting point, not a finished test set** — Walert's own
test set had 43 topics with several question paraphrases each. Extend
it the same way:

- `topics.csv` — add a row per question: `question_id,category,question`
  (category is `known`, `inferred`, or `out_of_kb`)
- `groundtruth.csv` — for known/inferred questions, add a row per
  relevant section: `question_id,act,section,relevance_judgment` (2 =
  fully answers it, 1 = partially relevant). Leave out_of_kb questions
  out of this file entirely — that's what marks them as out-of-KB.
- `gold_summaries.csv` — the ideal answer for each question, used for
  ROUGE-1 scoring.

Use `python cli.py inspect <act> <section>` to check the exact section
number/heading before adding a groundtruth row — the parser's section
numbers come straight from the Act (e.g. `319`, `71AB`, `50AAA`).

## An important note on the "% unanswered" metric

For **out-of-KB** questions there is, by definition, no right answer
in these 7 Acts. So a well-calibrated system should refuse (say "not
covered") on close to 100% of them — for this one category, a HIGH %
unanswered is the good outcome, not a bad one. (It's easy to misread
this the same way I initially did when walking through Walert's own
chart on this — worth stating explicitly in your report, since it's a
genuinely counter-intuitive metric direction the first time you see
it.) Known/inferred questions are the opposite, as usual: you want a
high answer rate there.

## Known limitations (worth a line in your report)

- **Margin-note leakage.** The parser strips the vast majority of
  Victorian legislation's amendment-history margin notes, but a small
  number of note fragments (e.g. a single lowercase word that's the
  tail of a note like "S. 35(1) def. of *vagina*") aren't caught by
  the filter and occasionally leak a stray word into a section's text.
  Always spot-check with `inspect` before trusting a section wholesale.
- **BM25-only retrieval is noticeably weaker on some known-answer
  questions** than you might expect — e.g. asking about "community
  correction order" without hitting the exact defining section by
  keyword overlap alone. This is real, reproducible evidence for why
  hybrid retrieval + reranking matters, not just a theoretical claim —
  worth including as a concrete example in your report.
- **Extractive mode (`--extractive`) always returns its best guess,**
  even for out-of-KB questions — it has no LLM judgement step to
  decide to refuse. It's meant for quickly sanity-checking retrieval
  quality without needing Ollama running, not for real refusal
  behaviour; that's the LLM path's job (see the system prompt in
  `generate.py`).
