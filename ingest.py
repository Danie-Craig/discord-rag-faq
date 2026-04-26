"""Ingestion pipeline: read every .md/.txt file in data/, chunk it, embed
each chunk with sentence-transformers AND build a BM25 lexical index.
Saves both to index.npz. Run once whenever knowledge base changes.

Hybrid retrieval (BM25 + dense) is industry standard — dense embeddings catch
semantic matches, BM25 catches literal keyword matches. Combining them with
weighted score fusion gives both strengths.

Usage:
    python ingest.py
"""

import os
import pickle
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from logger_config import configure_logging, get_logger

load_dotenv()
configure_logging()
log = get_logger("ingest")

DATA_DIR = Path("data")
INDEX_PATH = Path("index.npz")
BM25_PATH = Path("bm25.pkl")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware splitter targeting `size` chars with `overlap` overlap."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > size:
                for i in range(0, len(para), size - overlap):
                    chunks.append(para[i : i + size])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase + word-split tokenizer for BM25.

    Strips markdown punctuation but keeps alphanumeric+hyphens.
    Fine for English FAQ content; for multilingual KBs you'd swap in a
    language-aware tokenizer.
    """
    text = text.lower()
    # Keep alphanumeric, hyphen (compounds), apostrophe (contractions).
    tokens = re.findall(r"[a-z0-9][a-z0-9'\-]*", text)
    return tokens


def load_documents() -> list[tuple[str, str]]:
    """Return list of (source_filename, full_text)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")

    docs: list[tuple[str, str]] = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            docs.append((path.name, text))
            log.info("loaded_document", file=path.name, chars=len(text))
    if not docs:
        raise RuntimeError(
            f"No .md or .txt files found in {DATA_DIR}/. Add knowledge base content first."
        )
    return docs


def main() -> None:
    log.info("ingest_start", model=EMBEDDING_MODEL, chunk_size=CHUNK_SIZE)

    docs = load_documents()
    all_chunks: list[str] = []
    all_sources: list[str] = []

    for source, text in docs:
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_sources.append(source)

    log.info("chunked", total_chunks=len(all_chunks))

    # --- Dense embeddings ---
    log.info("loading_embedding_model")
    model = SentenceTransformer(EMBEDDING_MODEL)

    log.info("embedding_chunks", count=len(all_chunks))
    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True,  # so dot product == cosine similarity
        show_progress_bar=True,
    ).astype(np.float32)

    np.savez(
        INDEX_PATH,
        embeddings=embeddings,
        chunks=np.array(all_chunks, dtype=object),
        sources=np.array(all_sources, dtype=object),
    )
    log.info("dense_index_saved", path=str(INDEX_PATH), dim=embeddings.shape[1])

    # --- BM25 lexical index ---
    log.info("building_bm25_index")
    tokenized = [tokenize_for_bm25(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    log.info("bm25_index_saved", path=str(BM25_PATH))

    log.info(
        "ingest_complete",
        chunks=len(all_chunks),
        dense_index=str(INDEX_PATH),
        bm25_index=str(BM25_PATH),
    )


if __name__ == "__main__":
    main()
