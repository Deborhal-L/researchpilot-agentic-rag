
import json
import statistics
from pathlib import Path

from src.retrieve import Retriever
from src.agent import run_agent


DATASET = Path("data/eval/questions.json")
REPORT = Path("data/eval/agent_report.json")

def main():
    questions = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    retriever = Retriever()

    results = []

    print(f"Running {len(questions)} end-to-end tests...\n")

    for item in questions:
        print(f"[{item['id']}] {item['question']}")

        try:
            trace = run_agent(item["question"], retriever)

            expected_status = item["expected_status"]
            expected_ids = set(item["expected_chunk_ids"])
            actual_ids = set(trace.get("citations", []))

            status_correct = trace["status"] == expected_status
            citation_recall = (
                len(expected_ids & actual_ids) / len(expected_ids)
                if expected_ids
                else None
            )
            citations_valid = (
                not actual_ids
                or actual_ids.issubset(expected_ids)
            )

            result = {
                "id": item["id"],
                "question": item["question"],
                "expected_status": expected_status,
                "actual_status": trace["status"],
                "status_correct": status_correct,
                "expected_chunk_ids": sorted(expected_ids),
                "actual_citations": sorted(actual_ids),
                "citation_recall": citation_recall,
                "citations_valid": citations_valid,
                "latency_seconds": round(trace.get("latency", 0), 2),
                "tokens": trace.get("tokens", 0),
                "cost": trace.get("cost", 0),
                "trace_id": trace.get("id"),
                "answer": trace.get("answer", ""),
                "faithfulness": trace.get("faithfulness"),
            }

        except Exception as exc:
            result = {
                "id": item["id"],
                "question": item["question"],
                "error": str(exc),
                "status_correct": False,
                "citations_valid": False,
            }

        results.append(result)

        print(
            f"  Status: {result.get('actual_status', 'ERROR')} "
            f"| Correct: {result['status_correct']}"
        )

    completed = [r for r in results if "error" not in r]
    latencies = [
        r["latency_seconds"] for r in completed
    ]

    summary = {
        "total_questions": len(questions),
        "completed": len(completed),
        "status_accuracy": (
            sum(r["status_correct"] for r in completed) / len(completed)
            if completed else 0
        ),
        "citation_validity_rate": (
            sum(r["citations_valid"] for r in completed) / len(completed)
            if completed else 0
        ),
        "mean_latency_seconds": (
            round(statistics.mean(latencies), 2)
            if latencies else 0
        ),
        "results": results,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Evaluation Summary ---")
    print(f"Completed: {summary['completed']}/{summary['total_questions']}")
    print(f"Status accuracy: {summary['status_accuracy']:.1%}")
    print(f"Citation validity rate: {summary['citation_validity_rate']:.1%}")
    print(f"Mean latency: {summary['mean_latency_seconds']} seconds")
    print(f"Report saved to: {REPORT}")


if __name__ == "__main__":
    main()
