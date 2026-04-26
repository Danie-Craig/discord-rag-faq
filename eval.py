"""Run the RAG evaluation suite.

For each example in eval_set.py, this script:
1. Runs the question through the RAG pipeline
2. Computes Retrieval Precision@k (was the expected source retrieved?)
3. Asks the LLM to judge Faithfulness (did the answer use only retrieved context?)
4. Asks the LLM to judge Answer Correctness (does the answer match the expected one?)

Prints a per-example breakdown plus aggregate scores. Saves a JSON report.

Usage:
    python eval.py

The judge model is the same LLM used for generation — pragmatic for a small
eval, but in a more rigorous setup you'd use a stronger separate judge model
to avoid the "self-evaluation bias" where a model rates its own outputs higher.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from eval_set import EVAL_SET
from logger_config import configure_logging, get_logger
from rag import RAGEngine

load_dotenv()
configure_logging()
log = get_logger("eval")

REPORT_PATH = Path("eval_report.json")

JUDGE_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
judge_client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is FAITHFUL to its retrieved context.

Faithfulness = the answer makes no factual claims beyond what the context supports.

Retrieved context:
{context}

Answer:
{answer}

Score from 1 to 5:
- 5 = every claim in the answer is directly supported by the context
- 4 = answer is mostly grounded with one minor unsupported detail
- 3 = answer is partially grounded; some claims unsupported
- 2 = answer makes several claims not supported by context
- 1 = answer is largely fabricated or contradicts the context

A correct refusal ("I don't have that information") when context is irrelevant is a 5.

Respond with ONLY a single integer 1-5. No explanation."""


CORRECTNESS_PROMPT = """You are evaluating whether an AI answer matches a gold-standard expected answer.

Question: {question}

Expected answer: {expected}

AI's answer: {answer}

Score from 1 to 5:
- 5 = AI answer fully matches the expected answer's key facts
- 4 = AI answer is mostly correct with minor omissions
- 3 = AI answer has the right idea but is incomplete or partially wrong
- 2 = AI answer is mostly wrong but contains some correct elements
- 1 = AI answer is wrong, contradicts expected, or fails to address the question

If the expected answer is "REFUSAL_EXPECTED", score 5 if the AI refused (e.g., said it doesn't have the information), else 1.

Respond with ONLY a single integer 1-5. No explanation."""


def llm_judge(prompt: str) -> int:
    """Ask the judge LLM for a 1-5 score. Robust to non-numeric output."""
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Find the first digit 1-5 in the response.
        for ch in text:
            if ch in "12345":
                return int(ch)
        log.warning("judge_unparseable_response", response=text)
        return 0
    except Exception as e:
        log.exception("judge_call_failed", error=str(e))
        return 0


def retrieval_precision(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Fraction of retrieved chunks whose source is in the expected sources.
    
    If expected_sources is empty (out-of-scope question), returns 1.0 if no
    sources are *strongly* retrieved, else the score reflects irrelevance.
    For simplicity here we just return 1.0 if expected is empty — the judge
    handles the "did it refuse" check via the correctness metric.
    """
    if not expected_sources:
        return 1.0
    if not retrieved_sources:
        return 0.0
    hits = sum(1 for s in retrieved_sources if s in expected_sources)
    return hits / len(retrieved_sources)


def main() -> None:
    log.info("eval_start", examples=len(EVAL_SET), judge_model=JUDGE_MODEL)
    rag = RAGEngine()

    results = []
    for ex in EVAL_SET:
        log.info("eval_example_start", id=ex["id"], question=ex["question"])
        start = time.perf_counter()

        try:
            response = rag.answer(ex["question"])
        except Exception as e:
            log.exception("rag_call_failed", id=ex["id"])
            results.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "error": str(e),
                    "precision": 0.0,
                    "faithfulness": 0,
                    "correctness": 0,
                    "latency_ms": 0,
                }
            )
            continue

        latency_ms = int((time.perf_counter() - start) * 1000)

        retrieved_sources = [c.source for c in response.chunks]
        precision = retrieval_precision(retrieved_sources, ex["expected_sources"])

        context_text = "\n\n".join(c.text for c in response.chunks)
        faithfulness = llm_judge(
            FAITHFULNESS_PROMPT.format(context=context_text, answer=response.answer)
        )
        correctness = llm_judge(
            CORRECTNESS_PROMPT.format(
                question=ex["question"],
                expected=ex["expected_answer"],
                answer=response.answer,
            )
        )

        result = {
            "id": ex["id"],
            "question": ex["question"],
            "expected_answer": ex["expected_answer"],
            "actual_answer": response.answer,
            "expected_sources": ex["expected_sources"],
            "retrieved_sources": retrieved_sources,
            "precision": round(precision, 2),
            "faithfulness": faithfulness,
            "correctness": correctness,
            "latency_ms": latency_ms,
        }
        results.append(result)
        log.info(
            "eval_example_done",
            id=ex["id"],
            precision=precision,
            faithfulness=faithfulness,
            correctness=correctness,
        )

    # Aggregate.
    n = len(results)
    avg_precision = sum(r["precision"] for r in results) / n
    avg_faithfulness = sum(r["faithfulness"] for r in results) / n
    avg_correctness = sum(r["correctness"] for r in results) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n

    summary = {
        "n_examples": n,
        "avg_retrieval_precision": round(avg_precision, 2),
        "avg_faithfulness_1_to_5": round(avg_faithfulness, 2),
        "avg_correctness_1_to_5": round(avg_correctness, 2),
        "avg_latency_ms": round(avg_latency, 0),
    }

    report = {"summary": summary, "results": results}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Pretty print to console.
    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)
    print(f"Examples evaluated:     {summary['n_examples']}")
    print(f"Retrieval Precision@k:  {summary['avg_retrieval_precision']:.2f}  (1.0 = perfect)")
    print(f"Faithfulness:           {summary['avg_faithfulness_1_to_5']:.2f} / 5")
    print(f"Answer Correctness:     {summary['avg_correctness_1_to_5']:.2f} / 5")
    print(f"Avg latency:            {summary['avg_latency_ms']:.0f} ms")
    print("=" * 70)
    print("\nPer-example breakdown:\n")
    for r in results:
        marker = "✓" if r.get("correctness", 0) >= 4 else "✗"
        print(f"  {marker} [{r['id']}] precision={r['precision']:.2f} "
              f"faith={r['faithfulness']} correct={r['correctness']} "
              f"({r['latency_ms']} ms)")
    print(f"\nFull report written to {REPORT_PATH}")
    log.info("eval_complete", **summary)


if __name__ == "__main__":
    main()
