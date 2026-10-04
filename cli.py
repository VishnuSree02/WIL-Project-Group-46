"""
cli.py — single entry point for the whole pipeline.

    python cli.py extract              # PDFs in data/raw/ -> data/processed/*.json
    python cli.py build                # build BM25 + dense indexes
    python cli.py inspect <act> <sec>  # print one extracted section, for spot-checking
    python cli.py ask "your question"
    python cli.py eval [--name run_name] [--no-dense] [--no-rerank] [--extractive]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config  # noqa: E402


def cmd_extract(args):
    import extract
    extract.main()


def cmd_build(args):
    import build_index
    build_index.main()


def cmd_inspect(args):
    if not config.ALL_SECTIONS_PATH.exists():
        print("Run `python cli.py extract` first.")
        return
    sections = json.load(open(config.ALL_SECTIONS_PATH, encoding="utf-8"))
    matches = [s for s in sections
               if args.act.lower() in s["act"].lower() and s["section"] == args.section]
    if not matches:
        print(f"No section {args.section} found matching act '{args.act}'. "
              f"Try `python cli.py inspect --list <act>` to see what's there.")
        return
    for s in matches:
        print(json.dumps(s, indent=2, ensure_ascii=False))


def cmd_ask(args):
    from rag import ask
    question = " ".join(args.question)
    result = ask(question, top_k=args.top_k, use_dense=not args.no_dense,
                 use_rerank=not args.no_rerank, llm=args.llm,
                 extractive=args.extractive)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['act']}, s. {s['section']} ({s['heading']})")


def cmd_chat(args):
    """Interactive REPL — the actual 'chatbot' experience."""
    from rag import ask
    print("Police Code RAG — ask a question about the 7 loaded Acts.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye.")
            break
        result = ask(question, top_k=args.top_k, use_dense=not args.no_dense,
                     use_rerank=not args.no_rerank, llm=args.llm,
                     extractive=args.extractive)
        print(f"\nBot: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['act']}, s. {s['section']} ({s['heading']})")
        print()


def cmd_chat(args):
    from rag import ask
    print("Police Code RAG — type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        result = ask(question, top_k=args.top_k, use_dense=not args.no_dense,
                      use_rerank=not args.no_rerank, llm=args.llm,
                      extractive=args.extractive)
        print(f"\nBot: {result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s['act']}, s. {s['section']} ({s['heading']})")
        print()


def cmd_eval(args):
    import evaluate
    evaluate.evaluate(top_k=args.top_k, use_dense=not args.no_dense,
                       use_rerank=not args.no_rerank, llm=args.llm,
                       extractive=args.extractive, run_name=args.name)


def main():
    parser = argparse.ArgumentParser(description="Victorian legislation RAG pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("extract", help="extract PDFs in data/raw/ into section chunks")
    sub.add_parser("build", help="build BM25 + dense indexes")

    p_inspect = sub.add_parser("inspect", help="print one extracted section")
    p_inspect.add_argument("act", help="substring of the Act name, e.g. 'Crimes'")
    p_inspect.add_argument("section", help="section number, e.g. 319")

    p_ask = sub.add_parser("ask", help="ask the RAG pipeline a single question")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--top-k", type=int, default=config.FINAL_TOP_K)
    p_ask.add_argument("--no-dense", action="store_true")
    p_ask.add_argument("--no-rerank", action="store_true")
    p_ask.add_argument("--extractive", action="store_true")
    p_ask.add_argument("--llm", default=config.OLLAMA_MODEL)

    p_chat = sub.add_parser("chat", help="interactive chat loop (the actual chatbot)")
    p_chat.add_argument("--top-k", type=int, default=config.FINAL_TOP_K)
    p_chat.add_argument("--no-dense", action="store_true")
    p_chat.add_argument("--no-rerank", action="store_true")
    p_chat.add_argument("--extractive", action="store_true")
    p_chat.add_argument("--llm", default=config.OLLAMA_MODEL)

    p_eval = sub.add_parser("eval", help="run the Walert-style evaluation")
    p_eval.add_argument("--name", default="baseline")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--no-dense", action="store_true")
    p_eval.add_argument("--no-rerank", action="store_true")
    p_eval.add_argument("--extractive", action="store_true")
    p_eval.add_argument("--llm", default=config.OLLAMA_MODEL)

    args = parser.parse_args()
    {
        "extract": cmd_extract,
        "build": cmd_build,
        "inspect": cmd_inspect,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "eval": cmd_eval,
    }[args.command](args)


if __name__ == "__main__":
    main()
