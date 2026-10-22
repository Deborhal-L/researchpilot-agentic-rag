
"""LLM wrapper with mock, Ollama, and Anthropic backends."""

import json
import re
import time
import urllib.request

from .config import (
    LLM_BACKEND,
    LLM_MODEL,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PRICE_IN,
    PRICE_OUT,
)


STOP = set(
    "a an the of to in on for and or is are was were be by with as at from "
    "that this it its what which how do does did between about differ "
    "difference compare list according assigned the two both documents "
    "document recommendations recommendation".split()
)


def words(text: str) -> set[str]:
    """Return meaningful lowercase words."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP and len(word) > 2
    }


def passages(text: str) -> list[tuple[str, str]]:
    """Parse passages formatted as [chunk_id] on its own line."""
    parts = re.split(r"^\[([^\]\n]+)\]\s*$", text, flags=re.M)

    return [
        (parts[i].strip(), parts[i + 1].strip())
        for i in range(1, len(parts) - 1, 2)
        if parts[i].strip() and parts[i + 1].strip()
    ]


def is_comparison_question(query: str) -> bool:
    """Detect questions that require evidence from multiple sources."""
    q = query.lower()

    return any(
        term in q
        for term in (
            "compare",
            "differ",
            "difference",
            "both documents",
            "between the two",
            "differences between",
        )
    )


def is_mitigation_question(query: str) -> bool:
    """Detect mitigation or recommendation questions."""
    q = query.lower()

    return any(
        term in q
        for term in (
            "mitigation",
            "recommend",
            "prevent",
            "fix",
            "remediation",
            "remedy",
        )
    )


def _field_text(passage: str, field: str) -> str:
    """Extract a labeled field up to the next known field label."""
    match = re.search(
        rf"(?is)\b{re.escape(field)}:\s*(.*?)(?=\s+"
        r"(?:Detection|Product|Issue|Impact|Mitigation):|$)",
        passage,
    )

    return match.group(1).strip() if match else ""


def mock(system: str, user: str) -> str:
    """Keyword-based mock for testing the RAG pipeline."""
    prompt = system.lower()

    # 1. Query planning
    if "query planner" in prompt:
        return json.dumps({
            "sub_queries": [user.strip()],
            "reasoning": "mock: single sub-query",
        })

    # 2. Evidence checking
    if (
        "judge whether retrieved passages" in prompt
        or "supplied passages contain enough evidence" in prompt
    ):
        query = (
            user.split("Passages:", 1)[0]
            .replace("Sub-query:", "")
            .strip()
        )

        passage_text = user.split("Passages:", 1)[-1]
        passage_blocks = passages(passage_text)
        query_words = words(query)

        # Do not invent a CVSS score when none is supplied.
        asks_for_cvss = (
            "cvss" in query.lower()
            and "score" in query.lower()
        )

        has_cvss_score = any(
            re.search(
                r"\bCVSS\b.{0,30}\b\d+(?:\.\d+)?\b",
                passage,
                re.I,
            )
            for _, passage in passage_blocks
        )

        if asks_for_cvss and not has_cvss_score:
            return json.dumps({
                "sufficient": False,
                "supporting_chunk_ids": [],
                "missing": "The documents do not provide a CVSS score.",
                "rewrite": "",
            })

        # Reject questions about topics absent from retrieved passages.
        topic_terms = {
            "ransomware": {
                "ransomware",
                "ransom",
                "encrypting files",
            },
            "directory traversal": {
                "directory traversal",
                "path traversal",
            },
            "weak authentication": {
                "weak authentication",
                "weak credentials",
                "authentication configuration",
            },
        }

        query_lower = query.lower()
        combined_passages = "\n".join(
            passage.lower() for _, passage in passage_blocks
        )

        for topic, indicators in topic_terms.items():
            if topic in query_lower:
                topic_present = any(
                    indicator in combined_passages
                    for indicator in indicators
                )

                if not topic_present:
                    return json.dumps({
                        "sufficient": False,
                        "supporting_chunk_ids": [],
                        "missing": (
                            "The retrieved passages do not contain "
                            f"evidence about {topic}."
                        ),
                        "rewrite": "",
                    })

        # A comparison needs at least two distinct source chunks.
        if is_comparison_question(query):
            relevant = []

            for chunk_id, passage in passage_blocks:
                if is_mitigation_question(query):
                    if _field_text(passage, "Mitigation"):
                        relevant.append(chunk_id)
                elif len(query_words & words(passage)) >= 2:
                    relevant.append(chunk_id)

            good = sorted(set(relevant))
            sufficient = len(good) >= 2

        else:
            good = [
                chunk_id
                for chunk_id, passage in passage_blocks
                if (
                    len(query_words & words(passage))
                    / max(len(query_words), 1)
                    >= 0.34
                )
            ]

            sufficient = bool(good)

        return json.dumps({
            "sufficient": sufficient,
            "supporting_chunk_ids": sorted(set(good)),
            "missing": (
                ""
                if sufficient
                else "The retrieved passages do not provide enough evidence."
            ),
            "rewrite": "" if sufficient else " ".join(sorted(query_words)),
        })

    # 3. Answer generation
    if (
        "concise research assistant" in prompt
        or "answer the question using only" in prompt
    ):
        query = (
            user.split("Evidence:", 1)[0]
            .replace("Question:", "")
            .strip()
        )

        evidence_blocks = passages(user.split("Evidence:", 1)[-1])
        query_lower = query.lower()
        comparison = is_comparison_question(query)
        mitigation = is_mitigation_question(query)

        if any(
            term in query_lower
            for term in ("detect", "detected", "detection", "identify")
        ):
            preferred = (
                "detection",
                "logs",
                "monitor",
                "request",
                "review",
            )

        elif mitigation:
            preferred = (
                "mitigation",
                "validate",
                "restrict",
                "password",
                "authentication",
                "update",
                "multi-factor",
                "network access",
            )

        elif any(
            term in query_lower
            for term in ("impact", "effect", "risk", "harm")
        ):
            preferred = (
                "impact",
                "attacker",
                "unauthorized",
                "access",
            )

        else:
            preferred = ()

        candidates = []

        for chunk_id, passage in evidence_blocks:
            sentences = re.split(
                r"(?<=[.!?])\s+|\n+",
                passage,
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if (
                    not sentence
                    or len(sentence) < 20
                    or sentence.startswith("#")
                    or "illustrative training document" in sentence.lower()
                ):
                    continue

                # When the question asks for mitigation, use that field.
                if mitigation:
                    mitigation_text = _field_text(
                        passage,
                        "Mitigation",
                    )

                    if mitigation_text:
                        for mitigation_sentence in re.split(
                            r"(?<=[.!?])\s+",
                            mitigation_text,
                        ):
                            mitigation_sentence = mitigation_sentence.strip()

                            if (
                                mitigation_sentence
                                and len(mitigation_sentence) >= 10
                            ):
                                candidates.append(
                                    (mitigation_sentence, chunk_id)
                                )

                        continue

                lower_sentence = sentence.lower()

                if preferred and not any(
                    term in lower_sentence
                    for term in preferred
                ):
                    continue

                candidates.append((sentence, chunk_id))

        # Fallback to keyword overlap when needed.
        if not candidates:
            query_words = words(query)
            scored = []

            for chunk_id, passage in evidence_blocks:
                sentences = re.split(
                    r"(?<=[.!?])\s+|\n+",
                    passage,
                )

                for sentence in sentences:
                    sentence = sentence.strip()

                    if (
                        len(sentence) < 20
                        or sentence.startswith("#")
                        or "illustrative training document" in sentence.lower()
                    ):
                        continue

                    overlap = len(query_words & words(sentence))

                    if overlap:
                        scored.append((overlap, sentence, chunk_id))

            scored.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            candidates = [
                (sentence, chunk_id)
                for _, sentence, chunk_id in scored
            ]

        selected = []
        seen_sentences = set()

        if comparison:
            # Select evidence from each source separately.
            for chunk_id, passage in evidence_blocks:
                source_candidates = [
                    (sentence, source_id)
                    for sentence, source_id in candidates
                    if source_id == chunk_id
                ]

                if mitigation:
                    field_text = _field_text(
                        passage,
                        "Mitigation",
                    )

                    source_candidates = [
                        (sentence.strip(), chunk_id)
                        for sentence in re.split(
                            r"(?<=[.!?])\s+",
                            field_text,
                        )
                        if sentence.strip()
                    ] if field_text else source_candidates

                for sentence, source_id in source_candidates:
                    key = sentence.lower()

                    if key not in seen_sentences:
                        seen_sentences.add(key)
                        selected.append((sentence.strip(), source_id))

        else:
            for sentence, chunk_id in candidates:
                key = sentence.lower()

                if key in seen_sentences:
                    continue

                seen_sentences.add(key)
                selected.append((sentence, chunk_id))

                if len(selected) >= 2:
                    break

        citations = sorted({
            chunk_id for _, chunk_id in selected
        })

        if comparison and len(citations) < 2:
            return json.dumps({
                "answer": (
                    "The supplied evidence does not contain enough "
                    "information to compare both documents."
                ),
                "citations": [],
            })

        answer = " ".join(
            f"{sentence} [{chunk_id}]"
            for sentence, chunk_id in selected
        )

        if not answer:
            answer = (
                "The supplied evidence does not contain "
                "a relevant answer."
            )

        return json.dumps({
            "answer": answer,
            "citations": citations,
        })

    # 4. Faithfulness checking
    if (
        "verify an answer" in prompt
        or "verify an answer against" in prompt
    ):
        answer_match = re.search(
            r"Answer:\s*(.*?)(?:\n\nPassages:|$)",
            user,
            re.S,
        )

        answer = answer_match.group(1) if answer_match else ""
        evidence_text = user.split("Passages:", 1)[-1]
        evidence_blocks = passages(evidence_text)
        evidence_by_id = dict(evidence_blocks)

        citation_pattern = r"\[([^\]\n]+)\]"
        cited_ids = set(re.findall(citation_pattern, answer))

        claims = []
        all_supported = True

        sentences = re.split(
            r"(?<=\])\s+|(?<=[.!?])\s+(?!\[)",
            answer,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_citations = re.findall(
                citation_pattern,
                sentence,
            )

            claim = re.sub(
                r"\s*\[[^\]\n]+\]",
                "",
                sentence,
            ).strip()

            if not claim:
                continue

            # Remove field labels without changing the substantive claim.
            claim_for_check = re.sub(
                r"^(?:Answer|Detection|Mitigation|Impact|Issue|Product):\s*",
                "",
                claim,
                flags=re.I,
            ).strip()

            supported = False
            matched_chunk_id = ""

            claim_normalized = re.sub(
                r"\s+",
                " ",
                claim_for_check,
            ).strip().lower()

            # A claim must be supported by its cited evidence passage.
            for chunk_id in sentence_citations:
                passage = evidence_by_id.get(chunk_id, "")

                if not passage:
                    continue

                passage_normalized = re.sub(
                    r"\s+",
                    " ",
                    passage,
                ).strip().lower()

                if (
                    claim_normalized
                    and claim_normalized in passage_normalized
                ):
                    supported = True
                    matched_chunk_id = chunk_id
                    break

                # Permit claims copied from a labeled field.
                for field in (
                    "Mitigation",
                    "Detection",
                    "Impact",
                    "Issue",
                    "Product",
                ):
                    field_text = _field_text(passage, field)

                    field_normalized = re.sub(
                        r"\s+",
                        " ",
                        field_text,
                    ).strip().lower()

                    if (
                        claim_normalized
                        and field_normalized
                        and claim_normalized in field_normalized
                    ):
                        supported = True
                        matched_chunk_id = chunk_id
                        break

                if supported:
                    break

                # Conservative fallback for lightly reformatted claims.
                claim_words = words(claim_for_check)
                passage_words = words(passage)

                if claim_words and (
                    len(claim_words & passage_words)
                    / len(claim_words)
                    >= 0.70
                ):
                    supported = True
                    matched_chunk_id = chunk_id
                    break

            claims.append({
                "claim": claim,
                "chunk_id": matched_chunk_id or (
                    sentence_citations[0]
                    if sentence_citations
                    else ""
                ),
                "supported": supported,
            })

            if not supported:
                all_supported = False

        # Every citation must refer to supplied evidence.
        if not cited_ids.issubset(set(evidence_by_id)):
            all_supported = False

           
    unsupported_claims = [
    claim["claim"]
    for claim in claims
    if not claim["supported"]
]
    return json.dumps({
      "faithful": all_supported,
      "unsupported_claims": unsupported_claims,
      "claims": claims,
       "all_supported": all_supported,
})

    # 5. Clarification
    if "clarifying" in prompt:
        return json.dumps({
            "clarifying_question": "Which document or topic do you mean?",
        })

    return "{}"


def ollama(
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Call Ollama and report generation speed."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        body,
        {"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)

    eval_duration_ns = result.get("eval_duration", 0)
    eval_count = result.get("eval_count", 0)

    eval_duration_seconds = eval_duration_ns / 1_000_000_000

    tokens_per_second = (
        eval_count / eval_duration_seconds
        if eval_duration_seconds > 0
        else 0.0
    )

    print(
        f"Ollama generated {eval_count} tokens "
        f"in {eval_duration_seconds:.2f}s "
        f"({tokens_per_second:.2f} tokens/sec)"
    )

    return (
        result["message"]["content"],
        result.get("prompt_eval_count", 0),
        eval_count,
    )


_client = None


def anthropic_call(
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Call the Anthropic API."""
    global _client

    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()

    response = _client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    return (
        response.content[0].text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


def call_llm(
    system: str,
    user: str,
    max_tokens: int = 800,
) -> dict:
    """Run the configured backend and return usage information."""
    start = time.time()

    if LLM_BACKEND == "mock":
        text = mock(system, user)
        tokens_in = (len(system) + len(user)) // 4
        tokens_out = len(text) // 4
        cost = 0.0

    elif LLM_BACKEND == "ollama":
        text, tokens_in, tokens_out = ollama(
            system,
            user,
            max_tokens,
        )
        cost = 0.0

    elif LLM_BACKEND == "anthropic":
        text, tokens_in, tokens_out = anthropic_call(
            system,
            user,
            max_tokens,
        )

        cost = (
            tokens_in * PRICE_IN + tokens_out * PRICE_OUT
        ) / 1_000_000

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND={LLM_BACKEND!r}. "
            "Choose mock, ollama, or anthropic."
        )

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": cost,
        "latency": time.time() - start,
    }



def call_json(
    system: str,
    user: str,
    task_name: str = "other",
) -> tuple[dict, dict]:
    """Call the configured backend with a task-specific output limit."""
    prompt = system.lower()

    if "planner" in prompt:
        max_tokens = 128
    elif "evidence checker" in prompt or "supplied passages contain enough evidence" in prompt:
        max_tokens = 160
    elif "answer questions using retrieved evidence" in prompt or "concise research assistant" in prompt:
        max_tokens = 128
    elif "faithfulness" in prompt or "factuality verifier" in prompt or "verify an answer against" in prompt:
        max_tokens = 128
    elif "clarif" in prompt:
        max_tokens = 64
    else:
        max_tokens = 128

    result = call_llm(
        system + "\nReturn ONLY valid JSON, no markdown fences.",
        user,
        max_tokens=max_tokens,
    )

    result["task_name"] = task_name
    if task_name == "evidence_check":
       print("EVIDENCE CHECK RAW OUTPUT:", result["text"])
    print(
        f"LLM task={task_name} "
        f"latency={result['latency']:.2f}s "
        f"tokens_in={result['tokens_in']} "
        f"tokens_out={result['tokens_out']}"
    )

    output = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        result["text"].strip(),
        flags=re.M,
    ).strip()

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        result["json_error"] = str(exc)
        result["raw_output"] = output
        parsed = {}

    return parsed, result
