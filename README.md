# Production Agentic-RAG Research System

## Problem and corpus

<domain, N documents, why this domain>

## Architecture

Question → Planner → Hybrid retrieve + rerank → Evidence check → Next hop / synthesis → Faithfulness check → Answer or abstain

## Results (eval set: N labeled questions)

| Variant         | Hit@5 | Recall@5 | MRR | nDCG | Faithfulness | p95 latency | $/query |
| --------------- | ----- | -------- | --- | ---- | ------------ | ----------- | ------- |
| Vector only     |       |          |     |      |              |             |         |
| Hybrid + rerank |       |          |     |      |              |             |         |
| Full agentic    |       |          |     |      |              |             |         |

## Trade-offs

- Why hybrid over pure vector:
- Why abstain instead of guess:
- Why a hop cap:

## Failure modes (5 real cases)

1.

## What I changed from the reference repo

-

## What breaks at 10x traffic

-

## Run

```
pip install -r requirements.txt
python -m src.ingest
python -m eval.run_eval agentic
uvicorn api:app
```
