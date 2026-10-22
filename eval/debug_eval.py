
import json
from src.retrieve import Retriever
from src.agent import run_agent

with open("eval/questions.json", encoding="utf-8-sig") as f:
    questions = json.load(f)

retriever = Retriever()

for i, q in enumerate(questions, start=1):
    trace = run_agent(q["question"], retriever)

    expected = q["answerable"]
    predicted = trace["status"] != "abstain"
    result = "PASS" if predicted == expected else "FAIL"

    print(f"\n{'=' * 65}")
    print(f"Question {i}: {q['question']}")
    print(f"Expected answerable: {expected}")
    print(f"Status: {trace['status']}")
    print(f"Result: {result}")
    print(f"Answer: {trace['answer']}")
    print(f"Citations: {trace['citations']}")
