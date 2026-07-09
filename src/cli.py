"""
CLI for local RAG.

Usage:
    python cli.py index --docs ./my-docs      # build/rebuild the index
    python cli.py chat                        # chat with the indexed docs
    python cli.py ask "What is the return policy?"   # one-shot question
"""

import argparse
import sys

import src.core as rag


def cmd_index(args):
    rag.build_index(args.docs, persist_dir=args.db)


def cmd_chat(args):
    try:
        vectorstore = rag.load_index(persist_dir=args.db)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print("Chatting with your documents. Type 'quit' or Ctrl+C to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        answer, sources = rag.answer_query(vectorstore, question, top_k=args.top_k)
        print(f"\nAssistant: {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()


def cmd_ask(args):
    try:
        vectorstore = rag.load_index(persist_dir=args.db)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    answer, sources = rag.answer_query(vectorstore, args.question, top_k=args.top_k)
    print(answer)
    if sources:
        print(f"\nSources: {', '.join(sources)}")


def main():
    parser = argparse.ArgumentParser(description="Chat with a local directory of documents.")
    parser.add_argument("--db", default=rag.PERSIST_DIR, help="Path to the vector DB directory")
    parser.add_argument("--top-k", type=int, default=rag.TOP_K, help="Number of chunks to retrieve")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build/rebuild the index from a folder of documents")
    p_index.add_argument("--docs", required=True, help="Path to your documents folder")
    p_index.set_defaults(func=cmd_index)

    p_chat = sub.add_parser("chat", help="Interactive chat loop")
    p_chat.set_defaults(func=cmd_chat)

    p_ask = sub.add_parser("ask", help="Ask a single question and exit")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()