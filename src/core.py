"""
Core RAG logic: index a directory of documents, then query it.

"""

import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Config (override via CLI flags or env vars) ----
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
PERSIST_DIR = os.environ.get("RAG_DB_DIR", "./chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

LOADERS = {
    "**/*.txt": TextLoader,
    "**/*.md": TextLoader,
    "**/*.pdf": PyPDFLoader,
    "**/*.docx": UnstructuredWordDocumentLoader,
}


def load_documents(docs_dir: str):
    """Load every supported file under docs_dir (recursively)."""
    docs = []
    for glob_pattern, loader_cls in LOADERS.items():
        loader = DirectoryLoader(
            docs_dir,
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True,  # skip unreadable files instead of crashing the whole run
        )
        docs.extend(loader.load())
    return docs


def build_index(docs_dir: str, persist_dir: str = PERSIST_DIR):
    """Load, chunk, embed, and persist a directory of documents to a local Chroma DB."""
    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(
            f"No supported documents found in {docs_dir!r} "
            f"(looking for {', '.join(LOADERS)})"
        )
    print(f"Loaded {len(docs)} documents from {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Wipe any previous index so re-running index is idempotent
    if Path(persist_dir).exists():
        import shutil

        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )
    print(f"Index built and saved to {persist_dir}")
    return vectorstore


def load_index(persist_dir: str = PERSIST_DIR):
    """Load an already-built Chroma index from disk."""
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No index found at {persist_dir!r}. Run `python cli.py index --docs <folder>` first."
        )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the answer isn't in the context, say you don't know — don't make anything up.
Cite the source file name(s) you used at the end of your answer.

Context:
{context}

Question: {question}

Answer:"""


def answer_query(vectorstore, question: str, top_k: int = TOP_K):
    """Retrieve relevant chunks and generate a grounded answer with sources."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[{Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
        for d in retrieved_docs
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)
    response = llm.invoke(prompt)

    sources = sorted({Path(d.metadata.get("source", "unknown")).name for d in retrieved_docs})
    return response.content, sources