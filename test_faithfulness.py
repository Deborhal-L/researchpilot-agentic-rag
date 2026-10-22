from src.retrieve import Retriever
from src.llm import call_json
from src import prompts as P

retriever = Retriever()
ids = ["sample_a#0", "sample_b#0"]
chunks = [
    chunk for chunk in retriever.chunks
    if chunk["chunk_id"] in ids
]

evidence = "\n\n".join(
    f"[{chunk['chunk_id']}]\n{chunk['text']}"
    for chunk in chunks
)

answer = (
    "Sample A recommends validating and normalizing file paths, "
    "restricting file access to the permitted directory, and applying "
    "the vendor update [sample_a#0]. "
    "Sample B recommends strong passwords, MFA where supported, "
    "restricting network access, and applying the vendor update "
    "[sample_b#0]."
)

prompt = f"""
Original question:
How do the two sample documents differ in their mitigation recommendations?

Answer:
{answer}

Evidence:
{evidence}

Allowed citation IDs:
{ids}
"""

result, usage = call_json(
    P.FAITHFULNESS,
    prompt,
    task_name="faithfulness_test",
)

print("VERDICT:", result)
print("USAGE:", usage)
