"""RAG pipeline with HYBRID retrieval (dense + BM25) and optional
conversation memory.

Loads both indices built by ingest.py once at startup. Hybrid retrieval is
industry standard:
  - Dense embeddings catch semantic matches ("office hours scheduled" -> "Office Hours")
  - BM25 catches literal keyword matches ("deadline" -> chunks with the word "deadline")
  - Score fusion: normalized weighted sum, alpha=HYBRID_ALPHA (0.0=BM25 only, 1.0=dense only)

Conversation memory: callers can pass a `history` list of prior {role, content}
messages to `answer()`. The history is prepended to the current question in
the LLM call so follow-ups like "tell me more" have context. The retrieval
query itself is still the latest user question alone — we don't embed history
because that would dilute the signal.

Swap to MongoDB Atlas Vector Search later by replacing only the `retrieve()` function.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ingest import tokenize_for_bm25
from logger_config import get_logger

log = get_logger("rag")

INDEX_PATH = Path("index.npz")
BM25_PATH = Path("bm25.pkl")
TOP_K = int(os.getenv("TOP_K", 6))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))  # 0.0=BM25 only, 1.0=dense only
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-R1")

SYSTEM_PROMPT = """You are the AI Bootcamp FAQ assistant for the PM Accelerator program.
Answer the user's question using ONLY the context provided below. If the context does
not contain the answer, say "I don't have that information in the bootcamp documentation —
please ask in the appropriate Discord channel or check Office Hours." Be concise and direct.
Cite sources by file name in square brackets like [intern_faq.md] when relevant.
If the user's question is a follow-up to a prior turn, you may use the conversation
history to interpret references like "it" or "that," but only retrieve facts from
the provided context."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    chunks: list[RetrievedChunk]


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize an array of scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


class RAGEngine:
    def __init__(self) -> None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Dense index not found at {INDEX_PATH}. Run `python ingest.py` first."
            )
        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run `python ingest.py` first."
            )

        log.info("loading_dense_index", path=str(INDEX_PATH))
        data = np.load(INDEX_PATH, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"]
        self.chunks: np.ndarray = data["chunks"]
        self.sources: np.ndarray = data["sources"]

        log.info("loading_bm25_index", path=str(BM25_PATH))
        with BM25_PATH.open("rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]

        log.info("loading_embedding_model", model=EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        if not LLM_API_KEY:
            log.warning("llm_api_key_missing")
        self.llm_client = OpenAI(
            api_key=LLM_API_KEY or "missing",
            base_url=LLM_BASE_URL or None,
        )
        log.info(
            "rag_ready",
            chunks=len(self.chunks),
            dim=self.embeddings.shape[1],
            llm_model=LLM_MODEL,
            hybrid_alpha=HYBRID_ALPHA,
        )

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        """Hybrid retrieval: weighted fusion of dense cosine + BM25 scores."""
        q_vec = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)[0]
        dense_scores = self.embeddings @ q_vec

        q_tokens = tokenize_for_bm25(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

        dense_norm = _minmax_normalize(dense_scores)
        bm25_norm = _minmax_normalize(bm25_scores)
        fused = HYBRID_ALPHA * dense_norm + (1.0 - HYBRID_ALPHA) * bm25_norm

        top_idx = np.argsort(-fused)[:k]
        return [
            RetrievedChunk(
                text=str(self.chunks[i]),
                source=str(self.sources[i]),
                score=float(fused[i]),
            )
            for i in top_idx
        ]

    def build_prompt(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Build the LLM messages, optionally with prior conversation history.

        Structure: system prompt -> prior history -> current user message
        (with retrieved context inlined). The current message always carries the
        retrieved context so the LLM grounds the new turn in fresh evidence.
        """
        context_blocks = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in chunks
        )
        user_message = (
            f"Context from the AI Bootcamp documentation:\n\n{context_blocks}\n\n"
            f"Question: {query}"
        )
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(self, messages: list[dict]) -> str:
        completion = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        return completion.choices[0].message.content or ""

    def answer(
        self, query: str, history: Optional[list[dict]] = None
    ) -> RAGResponse:
        log.info(
            "rag_query_received", query=query, history_turns=len(history or [])
        )
        chunks = self.retrieve(query)
        log.info(
            "retrieved",
            count=len(chunks),
            top_score=chunks[0].score if chunks else None,
            sources=[c.source for c in chunks],
        )
        messages = self.build_prompt(query, chunks, history=history)
        answer_text = self.generate(messages)
        unique_sources = list(dict.fromkeys(c.source for c in chunks))
        log.info("rag_query_complete", answer_chars=len(answer_text))
        return RAGResponse(
            answer=answer_text, sources=unique_sources, chunks=chunks
        )
