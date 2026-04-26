# Discord RAG FAQ Chatbot

A Retrieval-Augmented Generation Discord bot that answers questions about the **PM Accelerator AI Bootcamp** using the cohort's official documentation as a knowledge base.

**Role:** Backend Engineer + Data Scientist (dual role)
**Cohort:** 8
**Project:** Discord RAG FAQ Chatbot — Weeks 1–3

---

## What it does

When an intern asks a question in Discord (e.g. *"When are office hours?"*, *"How do I submit my assignment?"*), the bot:

1. Embeds the question into a vector and tokenizes it for lexical search
2. Runs **hybrid retrieval** — dense semantic + BM25 keyword — against the indexed knowledge base, then fuses the scores
3. Sends the question + retrieved chunks (plus optional conversation history) to an LLM
4. Replies in Discord with a grounded answer + thumbs-up/thumbs-down reactions for feedback

Slash commands:
- `/ask question:<text>` — ask a question. Remembers your last 3 exchanges.
- `/forget` — clear your conversation history with the bot.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and rationale.

```
Discord user ──/ask──► Discord Bot (discord.py)
                            │
                            ▼ HTTP (with optional history)
                    FastAPI Backend
                    ├── /api/rag-query   ── hybrid RAG ──► LLM
                    ├── /api/feedback     (logs to JSONL)
                    ├── /api/ingest       (rebuild indices on demand)
                    └── /health           (counters + status)
                            │
                            ▼
        Hybrid Index: index.npz (dense) + bm25.pkl (lexical)
                            │
                            ▼
                    Knowledge Base (data/*.md)
```

---

## Tech choices

| Component | Choice | Why |
|---|---|---|
| Backend | FastAPI | Async, auto OpenAPI docs, fastest to ship in Python |
| Bot | discord.py | Mature, slash commands, reaction events |
| Chunking | Paragraph-aware splitter (~800 chars w/ 50 overlap) | Preserves semantic coherence at chunk boundaries |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dim, ~80MB |
| Lexical retrieval | BM25 (rank_bm25) | Catches literal keyword matches dense embeddings miss |
| Vector store | NumPy + pickled BM25 | FAQ corpus is small (<1k chunks); FAISS / Atlas unneeded but documented as swap targets |
| Score fusion | Min-max normalize → weighted sum (`HYBRID_ALPHA`, default 0.5) | Industry-standard hybrid retrieval |
| LLM | Llama 3.1 8B via Groq (OpenAI-compatible API) | Free, fast; assignment-recommended DeepSeek-R1 via Azure AI Foundry is a 2-env-var swap |
| Logging | `structlog` with JSON output | Structured logs, request IDs, latency on every event |
| Container | Dockerfile + .dockerignore + healthcheck | Reproducible deploy, tested locally |

---

## Setup (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/Danie-Craig/discord-rag-faq.git
cd discord-rag-faq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Required values:

- `DISCORD_BOT_TOKEN` — from https://discord.com/developers/applications (create app → Bot → Reset Token)
- `LLM_API_KEY` — Groq API key from https://console.groq.com/, or any OpenAI-compatible key
- `LLM_BASE_URL` — `https://api.groq.com/openai/v1` (or your provider's endpoint)
- `LLM_MODEL` — `llama-3.1-8b-instant` (or any chat model your provider supports)

Optional tuning knobs:

- `HYBRID_ALPHA` — fusion weight (0.0 = BM25 only, 1.0 = dense only, 0.5 = balanced)
- `TOP_K` — number of retrieved chunks (default 6)
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunking parameters
- `MEMORY_TURNS` — per-user conversation memory window in exchanges (default 3)
- `INGEST_ADMIN_TOKEN` — if set, `POST /api/ingest` requires `X-Admin-Token` header

⚠️ **`.env` is in `.gitignore`. Never commit secrets.**

### 3. Add knowledge base content

Drop the source documents into `data/` as `.md` or `.txt` files. The included KB has:

- `data/bootcamp_journey.md`
- `data/intern_faq.md`
- `data/project_assignment.md`
- `data/training.md`

Add or replace these with your own docs to point the bot at a different corpus.

### 4. Build the indices

```bash
python ingest.py
```

This chunks every doc in `data/`, builds dense embeddings (saved to `index.npz`) and a BM25 lexical index (saved to `bm25.pkl`).

### 5. Run the API + bot

In two terminals (or use Docker, see below):

```bash
# Terminal 1: backend
uvicorn api:app --port 8000

# Terminal 2: Discord bot
python bot.py
```

In your Discord server: `/ask question: When are office hours?`

---

## Evaluation

A small hand-labeled eval suite (`eval.py`, `eval_set.py`) measures retrieval precision, faithfulness, and answer correctness via LLM-as-judge. Run with:

```bash
python eval.py
```

Current scores on 10 examples:

| Metric | Score |
|---|---|
| Retrieval Precision@k | 0.60 |
| Faithfulness | 4.90 / 5 |
| Answer Correctness | 4.50 / 5 |

See `architecture.md` §11 for the full methodology and findings.

---

## Docker

```bash
docker build -t discord-rag-faq .
docker run --rm -p 8000:8000 --env-file .env discord-rag-faq
```

The image runs the FastAPI service only. The Discord bot is intentionally not bundled — in production it lives on a separate host that maintains the long-lived Discord WebSocket. See `architecture.md` §12 for the deployment rationale.

---

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/rag-query` | `{ "query": str, "user_id": str?, "history": [{"role": "user"\|"assistant", "content": str}]? }` | `{ "answer": str, "sources": [str], "latency_ms": int, "request_id": str }` |
| POST | `/api/feedback` | `{ "query": str, "answer": str, "rating": "up"\|"down", "user_id": str?, "comment": str? }` | `{ "ok": true }` |
| POST | `/api/ingest` | (empty) | `{ "ok": bool, "chunks": int, "duration_ms": int, "message": str }` |
| GET | `/health` | — | `{ "status": "ok", "metrics": {...} }` |

Auto-generated OpenAPI docs at `http://localhost:8000/docs`.

---

## Project structure

```
discord-rag-faq/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── ingest.py          # Chunk + embed + build dense + BM25 indices
├── rag.py             # Hybrid retrieval + prompt + LLM call + memory
├── api.py             # FastAPI server (4 endpoints + middleware)
├── bot.py             # Discord bot (slash commands + reactions + memory)
├── logger_config.py   # Structured JSON logging
├── eval.py            # Eval runner (LLM-as-judge for faithfulness + correctness)
├── eval_set.py        # 10 hand-labeled test cases
├── eval_report.json   # Last eval run results
├── data/              # Knowledge base source documents
│   ├── bootcamp_journey.md
│   ├── intern_faq.md
│   ├── project_assignment.md
│   └── training.md
└── docs/
    └── architecture.md
```

---

## Highlights beyond the base requirements

- **Hybrid retrieval** — BM25 lexical + dense semantic, score-fused. Improved answer correctness from 4.0/5 to 4.5/5 on the eval set.
- **RAG evaluation suite** — 10 hand-labeled examples scoring retrieval precision, faithfulness, and answer correctness. Surfaced and fixed two real failure modes.
- **Conversation memory** — per-user sliding window of 3 exchanges with a `/forget` Discord command.
- **`/api/ingest` endpoint** — refresh the knowledge base without redeploying. Spawns ingestion as a subprocess and hot-reloads the in-process engine.
- **Docker** — Dockerfile + `.dockerignore` + HEALTHCHECK, tested locally end-to-end.

## What's left as future work

- Move the index to **MongoDB Atlas Vector Search** (currently NumPy + BM25 pickle)
- Wire metrics into **Prometheus + Grafana** (currently logged as JSON, dashboard-ready)
- Add **graded relevance labels** in the eval set for more rigorous precision scoring
- Use a **stronger separate judge model** (e.g., GPT-4o) to remove self-evaluation bias in the eval

---

## Submission

GitHub repo: https://github.com/Danie-Craig/discord-rag-faq
Video walkthrough: submitted via the official Google Form.
