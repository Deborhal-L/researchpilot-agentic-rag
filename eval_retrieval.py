import json
import math
from pathlib import Path

from src.retrieve import Retriever

DATASET = Path("data/eval/questions.json")
OUTPUT = Path("data/eval/retrieval_report.json")
K_VALUES = (1, 3, 5)


def reciprocal_rank(relevant, ranked):
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant, ranked, k):
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


def main():
    questions = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    retriever = Retriever()
    results = []

    for item in questions:
        # Evaluate retrieval using the original question.
        chunks = retriever.search(item["question"], top_k=5)
        ranked_ids = list(dict.fromkeys(
            chunk["chunk_id"] for chunk in chunks
        ))
        relevant = set(item["expected_chunk_ids"])

        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": ranked_ids,
            "hit": bool(relevant.intersection(ranked_ids[:5]))
                if relevant else None,
            "reciprocal_rank": reciprocal_rank(relevant, ranked_ids)
                if relevant else None,
            "recall_at_k": {
                str(k): (
                    len(relevant.intersection(ranked_ids[:k])) / len(relevant)
                    if relevant else None
                )
                for k in K_VALUES
            },
            "ndcg_at_k": {
                str(k): ndcg_at_k(relevant, ranked_ids, k)
                if relevant else None
                for k in K_VALUES
            },
        }
        results.append(row)

        print(f"\n{item['id']}: {item['question']}")
        print("Expected:", sorted(relevant))
        print("Retrieved:", ranked_ids)
        print("Hit@5:", row["hit"])

    answerable = [r for r in results if r["expected_chunk_ids"]]
    summary = {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "hit_at_5": (
            sum(r["hit"] for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "mrr": (
            sum(r["reciprocal_rank"] for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "recall_at_k": {
            str(k): (
                sum(r["recall_at_k"][str(k)] for r in answerable)
                / len(answerable)
            )
            if answerable else 0.0
            for k in K_VALUES
        },
        "ndcg_at_k": {
            str(k): (
                sum(r["ndcg_at_k"][str(k)] for r in answerable)
                / len(answerable)
            )
            if answerable else 0.0
            for k in K_VALUES
        },
        "note": (
            "Initial evaluation on a two-document corpus. "
            "Unanswerable questions are excluded from retrieval relevance "
            "averages. These results do not measure answer faithfulness."
        ),
        "results": results,
    }

    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n--- Retrieval Evaluation Summary ---")
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "results"},
        indent=2,
    ))
    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
