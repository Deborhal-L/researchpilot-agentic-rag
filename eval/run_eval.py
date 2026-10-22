
"""Usage: python -m eval.run_eval [vector|hybrid|hybrid_rerank|agentic]"""

import json
import sys
import math
import numpy as np

from src.retrieve import Retriever
from src.agent import run_agent

K = 5

mode = sys.argv[1] if len(sys.argv) > 1 else "agentic"

with open("eval/questions.json", encoding="utf-8-sig") as f:
    questions = json.load(f)

retriever = Retriever()


def unique_in_order(items):
    """Remove duplicates without changing the ranking order."""
    return list(dict.fromkeys(items))


def metrics(retrieved, expected):
    """Calculate ranking metrics using unique chunk IDs."""
    retrieved = unique_in_order(retrieved)
    expected = set(expected)

    top_k = retrieved[:K]

    hit = int(any(chunk in expected for chunk in top_k))
    recall = (
        len(set(top_k) & expected) / len(expected)
        if expected else 0.0
    )

    reciprocal_rank = next(
        (
            1 / rank
            for rank, chunk in enumerate(retrieved[:K], start=1)
            if chunk in expected
        ),
        0.0,
    )

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk in enumerate(top_k, start=1)
        if chunk in expected
    )

    ideal_relevant = min(len(expected), K)
    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return hit, recall, reciprocal_rank, ndcg


rows = []
latencies = []
costs = []
abstention_correct = []

for question in questions:
    if mode == "agentic":
        trace = run_agent(
            question["question"],
            retriever,
            document_id=question.get("document_id"),
        )

        retrieved = [
            chunk
            for hop in trace["hops"]
            for chunk in hop["retrieved"]
        ]

        latencies.append(trace["latency"])
        costs.append(trace["cost"])

        predicted_abstention = trace["status"] == "abstain"
        expected_abstention = not question["answerable"]

        abstention_correct.append(
            predicted_abstention == expected_abstention
        )

    else:
        results = retriever.search(
            question["question"],
            mode=mode,
        )
        retrieved = [chunk["chunk_id"] for chunk in results]

    if question["answerable"]:
        rows.append(
            metrics(
                retrieved,
                question["expected_chunks"],
            )
        )

if rows:
    averages = np.mean(rows, axis=0)

    print(f"mode={mode}  n={len(rows)}")
    print(
        f"Hit@{K}={averages[0]:.2f}  "
        f"Recall@{K}={averages[1]:.2f}  "
        f"MRR@{K}={averages[2]:.2f}  "
        f"nDCG@{K}={averages[3]:.2f}"
    )
else:
    print("No answerable questions found; ranking metrics unavailable.")

if latencies:
    print(
        f"p95 latency={np.percentile(latencies, 95):.1f}s  "
        f"avg cost=${np.mean(costs):.4f}  "
        f"abstention correct={np.mean(abstention_correct):.2f}"
    )
